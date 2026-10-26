import importlib
import importlib.util
import json
import subprocess
import sys
from unittest.mock import patch

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import pysam

from cnn_mamba.common import splice_candidate_masks, start_stop_candidate_masks
from cnn_mamba.evaluate_gated_phase import BinnedBinaryMetrics
from cnn_mamba.prepare_human import parse_cds_phase_records as parse_human_phase
from cnn_mamba.model import SpliceMamba, model_from_checkpoint
from cnn_mamba.mamba3_torch import siso_scan
from cnn_mamba.mlm_pretrain import CNNMambaMLM, corrupt_for_mlm
from cnn_mamba.prepare_droso import (
    build_cds_boundary_mask, build_cds_track, build_phase_track,
)
from cnn_mamba.score_chromosome import keep_bounds
from cnn_mamba.train import (
    PUDataset, boundary_weighted_cds_loss, candidate_loss,
    cosine_learning_rate, masked_phase_loss, group_balanced_joint_frame_loss,
    shared_backbone_gradient_norm,
)
from cnn_mamba.temperature_scaling import (
    calibration_metrics, fit_temperature, joint_frame_targets,
)
from cnn_mamba.uniann_evaluate import parse_stats


class CoreTests(unittest.TestCase):
    def test_cosine_learning_rate_stays_at_floor_after_horizon(self):
        base = 3e-4
        self.assertAlmostEqual(cosine_learning_rate(0, base, 1000, 100), 3e-6)
        self.assertAlmostEqual(cosine_learning_rate(1000, base, 1000, 100), 3e-5)
        self.assertAlmostEqual(cosine_learning_rate(2000, base, 1000, 100), 3e-5)

    def test_human_phase_parser_normalizes_accession_and_chr_names(self):
        eviann = (
            "NC_000001.11\tEviAnn\tCDS\t2\t5\t.\t+\t1\tParent=x\n"
        )
        chess = (
            'chr1\tCHESS\tCDS\t8\t10\t.\t-\t2\ttranscript_id "y";\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            e_path = Path(directory) / "e.gff"
            c_path = Path(directory) / "c.gtf"
            e_path.write_text(eviann)
            c_path.write_text(chess)
            e_records = parse_human_phase(e_path, "eviann")
            c_records = parse_human_phase(c_path, "chess")
        self.assertEqual(e_records["chr1"]["+"], [(1, 5, 1)])
        self.assertEqual(c_records["chr1"]["-"], [(7, 10, 2)])

    def test_candidate_masks(self):
        # A C G T G T A G A T G T A A
        sequence = torch.tensor([[0, 1, 2, 3, 2, 3, 0, 2, 0, 3, 2, 3, 0, 0]])
        donor, acceptor = splice_candidate_masks(sequence)
        start, stop = start_stop_candidate_masks(sequence)
        self.assertEqual(torch.where(donor[0])[0].tolist(), [2, 4, 10])
        self.assertEqual(torch.where(acceptor[0])[0].tolist(), [6])
        self.assertEqual(torch.where(start[0])[0].tolist(), [8])
        self.assertEqual(torch.where(stop[0])[0].tolist(), [5, 11])

    def test_candidate_negative_weight_preserves_legacy_at_one(self):
        sequence = torch.tensor([[2, 3, 2, 3, 0, 2]])  # GT, GT, AG
        labels = torch.tensor([[1, 0, 0, 0, 2, 0]])
        logits = torch.tensor(
            [[[0.1, 1.2, -0.2], [0.0, 0.0, 0.0],
              [1.0, -0.1, -0.2], [0.0, 0.0, 0.0],
              [-0.3, 0.2, 1.1], [0.0, 0.0, 0.0]]]
        )
        legacy = candidate_loss(logits, labels, sequence, "splice")
        explicit = candidate_loss(
            logits, labels, sequence, "splice", negative_weight=1.0
        )
        self.assertTrue(torch.equal(legacy, explicit))

    def test_candidate_negative_weight_uses_weighted_mean(self):
        sequence = torch.tensor([[2, 3, 2, 3, 0, 2]])  # GT, GT, AG
        labels = torch.tensor([[1, 0, 0, 0, 2, 0]])
        logits = torch.tensor(
            [[[0.1, 1.2, -0.2], [0.0, 0.0, 0.0],
              [1.0, -0.1, -0.2], [0.0, 0.0, 0.0],
              [-0.3, 0.2, 1.1], [0.0, 0.0, 0.0]]],
            requires_grad=True,
        )
        loss = candidate_loss(
            logits, labels, sequence, "splice", negative_weight=0.25
        )
        selected_logits = logits[0, [0, 2, 4]]
        selected_labels = labels[0, [0, 2, 4]]
        per_candidate = torch.nn.functional.cross_entropy(
            selected_logits, selected_labels, reduction="none"
        )
        weights = torch.tensor([1.0, 0.25, 1.0])
        expected = (per_candidate * weights).sum() / weights.sum()
        self.assertAlmostEqual(loss.item(), expected.item(), places=7)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_model_shapes_without_mamba_layers(self):
        model = SpliceMamba(d_model=16, d_state=8, n_layers=0, dropout=0.0,
                            architecture="cnn_mamba", cnn_kernel_size=7)
        outputs = model(torch.randint(0, 5, (2, 101)))
        self.assertEqual(outputs[0].shape, (2, 101, 3))
        self.assertEqual(outputs[1].shape, (2, 101, 3))

    def test_mlm_corruption_masks_only_acgt_and_uses_mask_token(self):
        sequence = torch.tensor([[0, 1, 2, 3, 4] * 200])
        generator = torch.Generator().manual_seed(7)
        corrupted, targets = corrupt_for_mlm(
            sequence, mask_rate=0.15, span_length=3, generator=generator
        )
        chosen = targets != -100
        self.assertTrue(chosen.any())
        self.assertTrue((sequence[chosen] < 4).all())
        self.assertTrue((targets[sequence == 4] == -100).all())
        self.assertTrue((corrupted <= 5).all())
        self.assertTrue((corrupted == 5).any())

    def test_mlm_transfer_discards_mask_embedding_and_task_heads(self):
        model = CNNMambaMLM(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            cnn_kernel_size=7,
        )
        transferred = model.transfer_state_dict()
        self.assertEqual(transferred["embedding.weight"].shape, (5, 16))
        self.assertFalse(any("head" in name for name in transferred))
        downstream = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
        )
        incompatible = downstream.load_state_dict(transferred, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all("head" in name for name in incompatible.missing_keys))

    def test_frame_branch_and_phase_head_shapes(self):
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, phase_auxiliary=True,
            cds_auxiliary=True,
        )
        outputs = model(
            torch.randint(0, 5, (2, 101)),
            return_phase=True, return_cds=True,
        )
        self.assertEqual(len(outputs), 4)
        for output in outputs[:3]:
            self.assertEqual(output.shape, (2, 101, 3))
        self.assertEqual(outputs[3].shape, (2, 101, 2))

    def test_four_state_joint_frame_head_shape(self):
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, joint_frame_auxiliary=True,
        )
        outputs = model(
            torch.randint(0, 5, (2, 101)), return_joint_frame=True
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[2].shape, (2, 101, 4))

    def test_group_balanced_joint_frame_loss(self):
        logits = torch.zeros(1, 6, 4, requires_grad=True)
        phase = torch.tensor([[0, 1, 2, 0, 0, 0]])
        phase_mask = torch.tensor([[True, True, True, False, False, False]])
        cds = torch.tensor([[True, True, True, False, False, True]])
        loss = group_balanced_joint_frame_loss(
            logits, phase, phase_mask, cds, noncoding_weight=0.1
        )
        self.assertAlmostEqual(loss.item(), 1.1 * np.log(4), places=6)
        loss.backward()
        # The final CDS base has conflicting/unknown phase and is excluded.
        self.assertTrue(torch.equal(logits.grad[0, 5], torch.zeros(4)))
        self.assertIsNotNone(logits.grad)

    def test_joint_frame_targets_exclude_ambiguous_cds(self):
        phase = torch.tensor([[0, 1, 2, 0, 0]])
        phase_mask = torch.tensor([[True, True, True, False, False]])
        cds = torch.tensor([[True, True, True, False, True]])
        targets, valid = joint_frame_targets(phase, phase_mask, cds)
        self.assertEqual(targets.tolist(), [[1, 2, 3, 0, 0]])
        self.assertEqual(valid.tolist(), [[True, True, True, True, False]])

    def test_temperature_scaling_softens_without_changing_argmax(self):
        torch.manual_seed(7)
        base = torch.randn(4000, 4)
        targets = torch.distributions.Categorical(logits=base).sample()
        overconfident = base * 3.0
        temperature = fit_temperature(overconfident, targets)
        before = calibration_metrics(overconfident, targets)
        after = calibration_metrics(overconfident, targets, temperature)
        self.assertGreater(temperature, 1.5)
        self.assertLess(after["nll"], before["nll"])
        self.assertLess(
            after["mean_max_probability"], before["mean_max_probability"]
        )
        self.assertTrue(torch.equal(
            overconfident.argmax(-1), (overconfident / temperature).argmax(-1)
        ))

    def test_phase_tracks_follow_transcript_direction(self):
        plus = build_phase_track([(2, 8, 0)], 10, "+")
        minus = build_phase_track([(2, 8, 0)], 10, "-")
        self.assertEqual(plus[2:8].tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(minus[2:8][::-1].tolist(), [0, 1, 2, 0, 1, 2])

    def test_conflicting_phase_tracks_are_masked(self):
        track = build_phase_track([(0, 6, 0), (0, 6, 1)], 6, "+")
        self.assertTrue((track == -2).all())

    def test_conflicting_phase_tracks_remain_cds(self):
        track = build_cds_track([(0, 6, 0), (0, 6, 1)], 8)
        self.assertEqual(track.tolist(), [True] * 6 + [False] * 2)

    def test_cds_boundary_mask_uses_only_real_chromosome_transitions(self):
        track = np.asarray([True, True, True, False, False, False], dtype=bool)
        mask = build_cds_boundary_mask(track, radius=1)
        self.assertEqual(mask.tolist(), [False, False, True, True, False, False])

    def test_boundary_weighted_cds_loss_separately_averages_regions(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[0, 1, 1, 0]])
        boundary = torch.tensor([[False, True, True, False]])
        loss = boundary_weighted_cds_loss(
            logits, labels, boundary, far_weight=0.1
        )
        self.assertAlmostEqual(loss.item(), 1.1 * np.log(2), places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_window_ownership(self):
        self.assertEqual(keep_bounds(0, 3, 10_000, 5_000), (0, 7_500))
        self.assertEqual(keep_bounds(1, 3, 10_000, 5_000), (2_500, 7_500))
        self.assertEqual(keep_bounds(2, 3, 10_000, 5_000), (2_500, 10_000))

    def test_binned_binary_metrics_separates_perfect_scores(self):
        metric = BinnedBinaryMetrics(bins=16)
        metric.add(
            np.asarray([0.01, 0.1, 0.9, 0.99], dtype=np.float32),
            np.asarray([False, False, True, True]),
        )
        result = metric.metrics()
        self.assertAlmostEqual(result["AP"], 1.0)
        self.assertAlmostEqual(result["best_F1"], 1.0)

    def test_joint_phase_cds_dataset_layout_and_backward(self):
        length = 12
        with tempfile.TemporaryDirectory() as directory:
            arrays = {
                "sequence": np.asarray([0, 3, 2, 3] * 3, dtype=np.uint8),
                "labels": np.zeros(length, dtype=np.int8),
                "start_stop_labels": np.zeros(length, dtype=np.int8),
                "chess_labels": np.zeros(length, dtype=np.int8),
                "chess_start_stop_labels": np.zeros(length, dtype=np.int8),
                "phase_labels": np.arange(length, dtype=np.int8) % 3,
                "phase_mask": np.ones(length, dtype=bool),
                "reference_phase_labels": np.arange(length, dtype=np.int8) % 3,
                "reference_phase_mask": np.ones(length, dtype=bool),
                "cds_labels": np.asarray([0, 1] * 6, dtype=bool),
                "reference_cds_labels": np.asarray([1, 0] * 6, dtype=bool),
                "cds_boundary_mask": np.asarray(
                    [False, True, True, False] * 3, dtype=bool
                ),
                "reference_cds_boundary_mask": np.asarray(
                    [True, True, False, False] * 3, dtype=bool
                ),
            }
            np.savez_compressed(Path(directory) / "sample.npz", **arrays)
            batch = PUDataset(
                Path(directory), require_phase=True, require_cds=True,
                require_cds_boundary=True,
            )[0]
        self.assertEqual(len(batch), 15)
        self.assertTrue(torch.equal(batch[11], torch.from_numpy(arrays["cds_labels"])))
        self.assertTrue(torch.equal(
            batch[13], torch.from_numpy(arrays["cds_boundary_mask"])
        ))
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, phase_auxiliary=True, cds_auxiliary=True,
        )
        sequence = batch[0].unsqueeze(0)
        splice, start_stop, phase, cds = model(
            sequence, return_phase=True, return_cds=True
        )
        loss = (
            0.25 * candidate_loss(splice, batch[1].unsqueeze(0), sequence, "splice")
            + candidate_loss(
                start_stop, batch[2].unsqueeze(0), sequence, "start_stop"
            )
            + 0.1 * masked_phase_loss(
                phase, batch[7].unsqueeze(0), batch[8].unsqueeze(0)
            )
            + 0.1 * torch.nn.functional.cross_entropy(
                cds.reshape(-1, 2), batch[11].reshape(-1)
            )
        )
        shared = tuple(
            parameter for name, parameter in model.named_parameters()
            if not name.startswith((
                "splice_head.", "start_stop_head.", "phase_head.",
                "cds_head.", "frame_branch.",
            ))
        )
        cds_norm = shared_backbone_gradient_norm(
            torch.nn.functional.cross_entropy(
                cds.reshape(-1, 2), batch[11].reshape(-1)
            ),
            shared,
        )
        self.assertGreater(cds_norm, 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in shared))
        loss.backward()
        self.assertIsNotNone(model.cds_head.weight.grad)

    def test_stats_parser(self):
        text = (
            "Query mRNAs : 100 in 90 loci (80 multi-exon)\n"
            "Locus level:    75.0     |    80.0\n"
            "Matching loci: 72\nMissed loci: 10/100\nNovel loci: 3/90\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.stats"
            path.write_text(text)
            row = parse_stats(path)
        self.assertEqual(row["f1"], 77.41935483870968)
        self.assertEqual(row["predicted_loci"], 90)


def _recurrent_reference(q, k, v, adt, dt, trap, angles, q_bias, k_bias, skip, z):
    """Independent, per-token float64 trapezoidal recurrence with complex RoPE.

    S_t = exp(ADT_t) (S_{t-1} + dt_t (1-trap_t) K_{t-1} V_{t-1})
          + dt_t trap_t K_t V_t; y_t = (Q_t S_t + D V_t) silu(Z_t).
    """
    q, k = q.double() + q_bias.double(), k.double() + k_bias.double()
    v, dt, adt, trap = v.double(), dt.double(), adt.double(), trap.double().sigmoid()
    batch, length, heads, width = v.shape
    state = torch.zeros(batch, heads, k.shape[-1], width, dtype=torch.float64)
    theta = torch.zeros(batch, heads, angles.shape[-1], dtype=torch.float64)
    previous = torch.zeros_like(state)
    outputs = []
    for t in range(length):
        increment = angles[:, t].double().tanh()
        if increment.ndim == 2:
            increment = increment.unsqueeze(1)
        theta = theta + increment * np.pi * dt[:, t, :, None]
        rotation = torch.polar(torch.ones_like(theta), theta)

        def rotate(value):
            n = 2 * theta.shape[-1]
            pairs = torch.view_as_complex(value[..., :n].reshape(batch, heads, -1, 2).contiguous())
            return torch.cat((torch.view_as_real(pairs * rotation).flatten(-2), value[..., n:]), -1)

        qt, kt = rotate(q[:, t]), rotate(k[:, t])
        current = kt.unsqueeze(-1) * v[:, t].unsqueeze(-2)
        gamma = dt[:, t] * trap[:, t]
        beta = dt[:, t] * (1 - trap[:, t])
        state = adt[:, t, :, None, None].exp() * (
            state + beta[..., None, None] * previous
        ) + gamma[..., None, None] * current
        value = (qt.unsqueeze(-1) * state).sum(-2) + skip.double()[None, :, None] * v[:, t]
        outputs.append(value * torch.nn.functional.silu(z[:, t].double()))
        previous = current
    return torch.stack(outputs, 1)

class PortableInferenceTests(unittest.TestCase):
    def test_scan_matches_independent_float64_recurrence(self):
        torch.manual_seed(123)
        for length in (1, 65, 129, 1025):
            q = torch.randn(1, length, 1, 8)
            v = torch.randn(1, length, 2, 4)
            dt = torch.rand(1, length, 2) * 0.5 + 0.01
            args = (q, torch.randn_like(q), v, -torch.rand_like(dt) * dt, dt,
                    torch.randn_like(dt), torch.randn(1, length, 2),
                    torch.randn(2, 8), torch.randn(2, 8), torch.randn(2), torch.randn_like(v))
            expected = _recurrent_reference(*args)
            for group in (1, 4, 16):
                with self.subTest(length=length, group=group):
                    actual = siso_scan(*args, chunks_per_group=group)
                    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=2e-5)

    def test_checkpoint_roundtrip_preserves_both_architectures(self):
        for architecture in ("cnn_mamba", "mamba"):
            config = dict(d_model=16, d_state=8, n_layers=1, dropout=0., architecture=architecture)
            model = SpliceMamba(**config, backend="torch").eval()
            restored = model_from_checkpoint(dict(config=config, model_state_dict=model.state_dict()), "cpu", backend="torch").eval()
            self.assertEqual(restored.backend, "torch")
            self.assertEqual(tuple(restored.state_dict()), tuple(model.state_dict()))
            sequence = torch.randint(0, 5, (2, 129))
            with torch.inference_mode():
                for before, after in zip(model(sequence), restored(sequence)):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "optional native training package absent")
    def test_native_training_default_and_checkpoint_keys_are_preserved(self):
        native = SpliceMamba(d_model=64, d_state=16, n_layers=1)
        portable = SpliceMamba(d_model=64, d_state=16, n_layers=1, backend="torch")
        self.assertEqual(native.backend, "triton")
        saved = dict(config=dict(d_model=64, d_state=16, n_layers=1), model_state_dict=native.state_dict())
        self.assertEqual(model_from_checkpoint(saved, "cpu").backend, "triton")
        portable.load_state_dict(native.state_dict(), strict=True)
        native.load_state_dict(portable.state_dict(), strict=True)

    def test_inference_does_not_import_native_dependencies(self):
        code = '''
import importlib.abc
import sys
import torch
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mamba_ssm', 'triton', 'causal_conv1d'}:
            raise AssertionError('inference imported ' + fullname)
sys.meta_path.insert(0, RejectNative())
from cnn_mamba.model import SpliceMamba, model_from_checkpoint
config = dict(d_model=16, d_state=8, n_layers=1)
model = SpliceMamba(**config, backend='torch')
model = model_from_checkpoint(dict(config=config, model_state_dict=model.state_dict()), 'cpu', backend='torch').eval()
with torch.inference_mode():
    assert all(y.isfinite().all() for y in model(torch.zeros(1, 65, dtype=torch.long)))
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_all_four_cpu_scorers_preserve_formats_and_batch_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "ATGGTAGTAATGCGTAG" * 9 + "NNN"
            fasta = root / "tiny.fa"
            fasta.write_text(">tiny\n" + text + "\n")
            pysam.faidx(str(fasta))
            model = SpliceMamba(d_model=16, d_state=8, n_layers=1, backend="torch",
                                phase_auxiliary=True, cds_auxiliary=True, joint_frame_auxiliary=True)
            checkpoint = root / "model.pt"
            torch.save(dict(model_state_dict=model.state_dict(), epoch=1,
                            config=dict(d_model=16, d_state=8, n_layers=1, window_size=64,
                                        stride=32, phase_aux_weight=0.1)), checkpoint)
            for name in ("score_chromosome", "score_phase_chromosome",
                         "score_joint_frame_chromosome", "score_gated_phase_chromosome"):
                module = importlib.import_module("cnn_mamba." + name)
                destinations = []
                for batch in (None, 2):
                    destination = root / f"{name}_{batch}"
                    argv = [name, "--checkpoint", str(checkpoint), "--fasta", str(fasta),
                            "--chrom", "tiny", "--output-dir", str(destination),
                            "--device", "cpu", "--threads", "1"]
                    if batch is not None:
                        argv += ["--batch-size", str(batch)]
                    with patch.object(sys, "argv", argv):
                        module.main()
                    metadata = json.loads((destination / "metadata.json").read_text())
                    self.assertEqual(metadata["backend"], "torch")
                    self.assertEqual(metadata["device"], "cpu")
                    self.assertTrue((destination / ".complete.json").is_file())
                    for path in destination.glob("*.npy"):
                        self.assertTrue(np.isfinite(np.load(path)).all())
                    destinations.append(destination)
                for path in destinations[0].glob("*.npy"):
                    np.testing.assert_allclose(np.load(destinations[1] / path.name), np.load(path), rtol=2e-5, atol=2e-6)
                if name == "score_chromosome":
                    expected = [i for i in range(len(text)-1) if text[i:i+2] == "GT"]
                    self.assertEqual(np.load(destinations[0]/"donor_positions0.npy").tolist(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_portable_cuda_matches_cpu(self):
        model = SpliceMamba(d_model=64, d_state=16, n_layers=1, backend="torch", dropout=0.).eval()
        saved = dict(config=dict(d_model=64, d_state=16, n_layers=1, dropout=0.), model_state_dict=model.state_dict())
        accelerator = model_from_checkpoint(saved, "cuda", backend="torch").eval()
        sequence = torch.randint(0, 5, (2, 129))
        with torch.inference_mode():
            for expected, actual in zip(model(sequence), accelerator(sequence.cuda())):
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

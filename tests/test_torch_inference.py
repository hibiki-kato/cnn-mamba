"""Numerical equivalence, checkpoint compatibility and CPU CLI coverage."""

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pysam
import torch
from torch.nn import functional as F

from cnn_mamba.inference import resolve_device, inference_precision
from cnn_mamba.mamba3_torch import Mamba3Torch, siso_scan
from cnn_mamba.model import SpliceMamba, model_from_checkpoint


def recurrent_reference(q, k, v, adt, dt, trap, angles, q_bias, k_bias, skip, z):
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
        outputs.append(value * F.silu(z[:, t].double()))
        previous = current
    return torch.stack(outputs, 1)


class TorchInferenceTests(unittest.TestCase):
    def test_bf16_autocast_preserves_fp32_scan_context_and_checkpoint(self):
        model = SpliceMamba(d_model=64, d_state=16, n_layers=1, dropout=0.0).eval()
        keys = tuple(model.state_dict())
        module = importlib.import_module('cnn_mamba.mamba3_torch')
        scan = module.siso_scan
        contexts = []
        def observed(*args, **kwargs):
            contexts.append(torch.is_autocast_enabled('cpu'))
            return scan(*args, **kwargs)
        with patch.object(module, 'siso_scan', observed), torch.inference_mode(), inference_precision('bf16', torch.device('cpu')):
            outputs = model(torch.randint(0, 5, (2, 129)))
        self.assertEqual(contexts, [False, False])
        self.assertTrue(all(y.dtype == torch.bfloat16 and torch.isfinite(y).all() for y in outputs))
        self.assertEqual(tuple(model.state_dict()), keys)
        with self.assertRaises(ValueError):
            model.enable_bidirectional_fusion()

    def test_execution_group_uses_shape_without_hardware_probes(self):
        layer = Mamba3Torch(d_model=192, d_state=64, headdim=64)
        with patch("torch.cuda.get_device_properties", side_effect=AssertionError("no runtime probe")):
            self.assertEqual(layer.scan_group_size(1, torch.device("cuda")), 64)
            self.assertEqual(layer.scan_group_size(4, torch.device("cuda")), 16)
            self.assertEqual(layer.scan_group_size(16, torch.device("cuda")), 4)
        self.assertEqual(layer.scan_group_size(1, torch.device("cpu")), 16)
        self.assertEqual(layer.scan_group_size(4, torch.device("cpu")), 4)
        self.assertEqual(layer.scan_group_size(1, torch.device("mps")), 16)
        layer.chunks_per_group = 7
        self.assertEqual(layer.scan_group_size(1, torch.device("cuda")), 7)

    def test_scan_matches_independent_recurrence_across_blocks(self):
        torch.manual_seed(123)
        for length in (1, 63, 64, 65, 129, 1025, 1091):
            batch, heads, width, state = 2, 3, 4, 8
            q = torch.randn(batch, length, 1, state)
            k = torch.randn_like(q)
            v = torch.randn(batch, length, heads, width)
            dt = torch.rand(batch, length, heads) * 0.5 + 0.01
            args = (q, k, v, -torch.rand_like(dt) * dt, dt,
                    torch.randn_like(dt), torch.randn(batch, length, state // 4),
                    torch.randn(heads, state), torch.randn(heads, state),
                    torch.randn(heads), torch.randn_like(v))
            expected = recurrent_reference(*args)
            for chunk_size in (16, 64):
                for chunks_per_group in (1, 4, 16):
                    with self.subTest(length=length, chunk_size=chunk_size,
                                      chunks_per_group=chunks_per_group):
                        actual = siso_scan(*args, chunk_size=chunk_size,
                                           chunks_per_group=chunks_per_group)
                        # Parallel block sums reassociate float32 angle/decay
                        # additions; the independent reference accumulates in float64.
                        torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=2e-5)

    def test_scan_stays_finite_under_large_decay_and_angle_increments(self):
        torch.manual_seed(124)
        q = torch.randn(1, 193, 1, 8)
        v = torch.randn(1, 193, 2, 4)
        dt = torch.full((1, 193, 2), 10.0)
        args = (q, q * 0.5, v, -dt * 100, dt, torch.randn_like(dt),
                torch.randn(1, 193, 2) * 5, torch.ones(2, 8), torch.ones(2, 8),
                torch.ones(2), v)
        actual = siso_scan(*args)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.double(), recurrent_reference(*args), rtol=2e-5, atol=3e-4)

    def test_same_model_interface_and_checkpoint_for_both_architectures(self):
        for architecture in ("cnn_mamba", "mamba"):
            with self.subTest(architecture=architecture):
                model = SpliceMamba(
                    d_model=16, d_state=8, n_layers=1, dropout=0.0,
                    architecture=architecture, frame_dilation=3,
                    phase_auxiliary=True, cds_auxiliary=True, joint_frame_auxiliary=True,
                )
                x = torch.randint(0, 5, (2, 65))
                outputs = model(x, return_phase=True, return_cds=True, return_joint_frame=True)
                self.assertEqual([tuple(y.shape) for y in outputs],
                                 [(2, 65, 3), (2, 65, 3), (2, 65, 3), (2, 65, 2), (2, 65, 4)])
                sum(y.square().mean() for y in outputs).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                config = dict(d_model=16, d_state=8, n_layers=1, dropout=0.0,
                              architecture=architecture, frame_dilation=3, phase_aux_weight=0.1)
                restored = model_from_checkpoint({"config": config, "model_state_dict": model.state_dict()}, "cpu")
                self.assertEqual(restored.backend, "torch")
                with torch.inference_mode():
                    for before, after in zip(outputs, restored.eval()(
                            x, return_phase=True, return_cds=True, return_joint_frame=True)):
                        torch.testing.assert_close(before, after)

    def test_inference_never_imports_acceleration_packages(self):
        code = '''
import sys
import importlib.abc
import torch
class RejectAcceleration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"mamba_ssm", "triton", "causal_conv1d", "tilelang"}:
            raise AssertionError("inference imported " + fullname)
sys.meta_path.insert(0, RejectAcceleration())
from cnn_mamba.model import SpliceMamba
model = SpliceMamba(d_model=16, d_state=8, n_layers=1).eval()
with torch.inference_mode():
    model(torch.zeros(1, 65, dtype=torch.long))
assert "mamba_ssm" not in sys.modules
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_device_selection(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
                "torch.backends.mps.is_available", return_value=False):
            self.assertEqual(resolve_device().type, "cpu")
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
            with self.assertRaises(RuntimeError):
                resolve_device("mps")
        with patch("torch.cuda.is_available", return_value=False), patch(
                "torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device().type, "mps")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device().type, "cuda")
            self.assertEqual(str(resolve_device("cuda:1")), "cuda:1")

    def test_all_scoring_commands_on_cpu_with_real_mamba_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "ATGGTAGTAATGCGTAG" * 9 + "NNN"
            fasta = root / "tiny.fa"
            fasta.write_text(">tiny\n" + text + "\n")
            pysam.faidx(str(fasta))
            model = SpliceMamba(d_model=16, d_state=8, n_layers=1,
                                phase_auxiliary=True, cds_auxiliary=True, joint_frame_auxiliary=True)
            checkpoint = root / "model.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": 1,
                        "config": dict(d_model=16, d_state=8, n_layers=1,
                                       window_size=64, stride=32, phase_aux_weight=0.1)}, checkpoint)
            for name, output_file in (
                ("score_chromosome", "donor_scores.npy"),
                ("score_phase_chromosome", "phase_probabilities.npy"),
                ("score_joint_frame_chromosome", "joint_frame_probabilities.npy"),
                ("score_gated_phase_chromosome", "gated_phase_probabilities.npy"),
            ):
                with self.subTest(command=name):
                    destination = root / name
                    argv = [name, "--checkpoint", str(checkpoint), "--fasta", str(fasta),
                            "--chrom", "tiny", "--output-dir", str(destination), "--device", "cpu"]
                    module = importlib.import_module("cnn_mamba." + name)
                    with patch.object(sys, "argv", argv):
                        module.main()
                    metadata = json.loads((destination / "metadata.json").read_text())
                    self.assertEqual(metadata["backend"], "torch")
                    self.assertEqual(metadata["device"], "cpu")
                    self.assertEqual(metadata["batch_size"], 1)
                    self.assertTrue((destination / ".complete.json").is_file())
                    values = np.load(destination / output_file)
                    self.assertTrue(np.isfinite(values).all())
                    if name == "score_chromosome":
                        self.assertEqual(metadata["checkpoint_epoch"], 1)
                        self.assertIsNone(metadata["checkpoint_step"])
                        positions = np.load(destination / "donor_positions0.npy")
                        expected = [i for i in range(len(text) - 1) if text[i:i+2] == "GT"]
                        self.assertEqual(positions.tolist(), expected)
                    else:
                        self.assertEqual(values.shape[0], len(text))
                    auto_destination = root / (name + "_auto")
                    auto_argv = argv.copy()
                    auto_argv[auto_argv.index("--output-dir") + 1] = str(auto_destination)
                    auto_argv += ["--batch-size", "auto", "--memory-gb", "0.5", "--max-batch-size", "2"]
                    with patch.object(sys, "argv", auto_argv):
                        module.main()
                    auto_metadata = json.loads((auto_destination / "metadata.json").read_text())
                    self.assertEqual(auto_metadata["batch_size"], 2)
                    for path in destination.glob("*.npy"):
                        before, after = np.load(path), np.load(auto_destination / path.name)
                        if np.issubdtype(before.dtype, np.floating):
                            np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-6)
                        else:
                            np.testing.assert_array_equal(after, before)
                    bf16_destination = root / (name + '_bf16')
                    bf16_argv = argv.copy()
                    bf16_argv[bf16_argv.index('--output-dir') + 1] = str(bf16_destination)
                    bf16_argv += ['--precision', 'bf16']
                    with patch.object(sys, 'argv', bf16_argv):
                        module.main()
                    self.assertEqual(json.loads((bf16_destination/'metadata.json').read_text())['precision'], 'bf16')
                    wrong_precision = argv + ['--precision', 'bf16']
                    with patch.object(sys, 'argv', wrong_precision), self.assertRaisesRegex(ValueError, 'precision'):
                        module.main()
                    for path in destination.glob('*.npy'):
                        values = np.load(bf16_destination/path.name)
                        self.assertTrue(np.isfinite(values).all())
                        # Reduced precision can change a close class decision;
                        # candidate positions/motif identifiers remain exact.
                        if not np.issubdtype(values.dtype, np.floating) and 'argmax' not in path.name:
                            np.testing.assert_array_equal(values, np.load(path))
                        if path.name == 'joint_frame_argmax.npy':
                            probabilities = np.load(bf16_destination/'joint_frame_probabilities.npy')
                            np.testing.assert_array_equal(values, probabilities.argmax(-1))

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('mamba_ssm'),
                         'native CUDA training environment unavailable')
    def test_native_bidirectional_fusion_preserves_outputs_gradients_and_keys(self):
        if torch.version.hip is not None:
            self.skipTest('native training kernels require NVIDIA CUDA')
        from cnn_mamba.model import BiMamba3Block
        torch.manual_seed(107)
        original = BiMamba3Block(64, 16, 64, backend='triton').cuda()
        fused = BiMamba3Block(64, 16, 64, backend='triton').cuda()
        fused.load_state_dict(original.state_dict(), strict=True)
        fused.fuse_directions = True
        for length in (64, 65, 129):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(length=length, dtype=dtype):
                    original.zero_grad(set_to_none=True)
                    fused.zero_grad(set_to_none=True)
                    x = torch.randn(2, length, 64, device='cuda', requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype==torch.bfloat16):
                        a, b = original(x), fused(y)
                        loss_a, loss_b = a.float().square().mean(), b.float().square().mean()
                    torch.testing.assert_close(b, a, rtol=2e-4, atol=2e-5)
                    loss_a.backward()
                    loss_b.backward()
                    torch.testing.assert_close(y.grad, x.grad, rtol=2e-4, atol=2e-5)
                    self.assertEqual(tuple(fused.state_dict()), tuple(original.state_dict()))
                    for before, after in zip(original.parameters(), fused.parameters()):
                        torch.testing.assert_close(after.grad, before.grad, rtol=2e-4, atol=2e-5)

    @unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "optional Mamba3 training package absent")
    def test_upstream_state_dict_loads_strictly_in_both_directions(self):
        from mamba_ssm import Mamba3
        for d_model, d_state in ((16, 8), (64, 16), (192, 64)):
            with self.subTest(d_model=d_model, d_state=d_state):
                upstream = Mamba3(d_model=d_model, d_state=d_state, headdim=min(64, d_model))
                portable = Mamba3Torch(d_model=d_model, d_state=d_state, headdim=min(64, d_model))
                portable.load_state_dict(upstream.state_dict(), strict=True)
                upstream.load_state_dict(portable.state_dict(), strict=True)

    @unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "optional Mamba3 training package absent")
    def test_forward_matches_upstream_module_with_reference_cpu_kernels(self):
        from mamba_ssm import Mamba3
        upstream_module = importlib.import_module("mamba_ssm.modules.mamba3")

        def reference_kernel(Q, K, V, ADT, DT, Trap, Angles, Q_bias, K_bias, D, Z, **kwargs):
            return recurrent_reference(
                Q, K, V, ADT.transpose(1, 2), DT.transpose(1, 2),
                Trap.transpose(1, 2), Angles, Q_bias, K_bias, D, Z,
            ).to(V.dtype)

        def reference_norm(norm, x, z=None):
            self.assertIsNone(z)
            return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                            + norm.eps) * norm.weight).to(x.dtype)

        torch.manual_seed(125)
        upstream = Mamba3(d_model=64, d_state=16, headdim=64)
        portable = Mamba3Torch(d_model=64, d_state=16, headdim=64)
        portable.load_state_dict(upstream.state_dict(), strict=True)
        x = torch.randn(2, 129, 64)
        with torch.inference_mode(), patch.object(
                upstream_module, "mamba3_siso_combined", reference_kernel), patch.object(
                upstream_module.RMSNormGated, "forward", reference_norm):
            expected = upstream(x)
            actual = portable(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_portable_accelerator_matches_cpu_without_training_kernels(self):
        # ROCm deliberately presents the same torch.cuda API as NVIDIA.
        torch.manual_seed(51)
        cpu = SpliceMamba(d_model=64, d_state=16, n_layers=1,
                          dropout=0.0, joint_frame_auxiliary=True).eval()
        gpu = SpliceMamba(d_model=64, d_state=16, n_layers=1,
                          dropout=0.0, joint_frame_auxiliary=True).cuda().eval()
        gpu.load_state_dict(cpu.state_dict(), strict=True)
        x = torch.randint(0, 5, (2, 1091))
        with torch.inference_mode():
            for reference, actual in zip(cpu(x, return_joint_frame=True), gpu(x.cuda(), return_joint_frame=True)):
                torch.testing.assert_close(actual.cpu(), reference, rtol=3e-4, atol=3e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_matches_original_triton_model(self):
        if torch.version.hip is not None:
            self.skipTest("upstream training kernels require NVIDIA CUDA")
        if not importlib.util.find_spec("mamba_ssm"):
            self.skipTest("optional Mamba3 training package absent")
        if not torch.cuda.is_bf16_supported():
            self.skipTest('native Mamba3 kernels require BF16 support')
        torch.manual_seed(42)
        training = SpliceMamba(d_model=64, d_state=16, n_layers=1,
                               dropout=0.0, backend="triton").cuda().eval()
        portable = SpliceMamba(d_model=64, d_state=16, n_layers=1,
                               dropout=0.0).cuda().eval()
        portable.load_state_dict(training.state_dict(), strict=True)
        cpu = SpliceMamba(d_model=64, d_state=16, n_layers=1,
                          dropout=0.0).eval()
        cpu.load_state_dict(portable.state_dict(), strict=True)
        for length in (64, 65, 256):
            with self.subTest(length=length), torch.inference_mode():
                x = torch.randint(0, 5, (2, length), device="cuda")
                fused, actual, reference = training(x), portable(x), cpu(x.cpu())
                for expected, output, cpu_output in zip(fused, actual, reference):
                    torch.testing.assert_close(output.cpu(), cpu_output, rtol=2e-4, atol=2e-5)
                    # The pinned wrapper forces BF16 scan inputs/intermediates;
                    # the portable scan accumulates in FP32. Bound probability
                    # and RMS logit error without demanding identical arithmetic.
                    self.assertLess((output.softmax(-1) - expected.softmax(-1)).abs().max().item(), 1e-3)
                    self.assertLess((output - expected).square().mean().sqrt().item(), 1e-3)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_forward_matches_cpu(self):
        cpu = SpliceMamba(d_model=16, d_state=8, n_layers=1).eval()
        mps = SpliceMamba(d_model=16, d_state=8, n_layers=1).to("mps").eval()
        mps.load_state_dict(cpu.state_dict())
        x = torch.randint(0, 5, (1, 129))
        with torch.inference_mode():
            for expected, actual in zip(cpu(x), mps(x.to("mps"))):
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()

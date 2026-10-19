import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn_mamba.gene_paths import (
    COMPLEMENT, build_isoform_mask_tracks, build_path_tracks,
    longest_cds_transcripts,
)
from cnn_mamba.gene_hmm import (
    ALL_EVENTS, BOUNDARIES, E0, E1, E2, EVENT_SHIFT, I0, I1, I2, INFEASIBLE, IR,
    N_STATES, UNCONSTRAINED, GeneHMM, label_mask, labels_to_masks,
    log_partition, mask_potentials, path_is_grammatical, transition_potentials,
    viterbi,
)

ENCODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def encode(text):
    return torch.tensor([[ENCODE[base] for base in text]])


def brute_force(emission, transition):
    """Return scores of every path, and the paths, for one sequence."""
    length = emission.shape[0]
    paths = np.array(list(itertools.product(range(N_STATES), repeat=length)))
    score = emission[np.arange(length), paths].sum(1)
    for t in range(1, length):
        score += transition[t, paths[:, t - 1], paths[:, t]]
    return score, paths


def path_score(emission, transition, path):
    score = emission[0, 0, path[0]]
    for t in range(1, len(path)):
        score = score + transition[0, t, path[t - 1], path[t]] + emission[0, t, path[t]]
    return float(score)


def random_inputs(text, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sequence = encode(text)
    length = len(text)
    emission = torch.randn(1, length, N_STATES, generator=generator, dtype=torch.float64)
    boundary = {
        name: torch.randn(1, length, generator=generator, dtype=torch.float64)
        for name in BOUNDARIES
    }
    return sequence, emission, boundary


class GeneHMMTests(unittest.TestCase):
    # Contains ATG, GT, AG and TAG/TAA motifs so every transition is reachable.
    TEXTS = ("ATGTAG", "ATGTAAG", "GTAGTAA")

    def test_log_partition_matches_brute_force(self):
        for text in self.TEXTS:
            sequence, emission, boundary = random_inputs(text)
            transition = transition_potentials(sequence, boundary)
            scores, _ = brute_force(emission[0].numpy(), transition[0].numpy())
            expected = np.logaddexp.reduce(scores[scores > INFEASIBLE])
            got = float(log_partition(emission, transition)[0])
            self.assertAlmostEqual(got, expected, places=8)

    def test_constrained_partition_and_viterbi(self):
        sequence, emission, boundary = random_inputs("ATGTAAG", seed=3)
        transition = transition_potentials(sequence, boundary)
        scores, paths = brute_force(emission[0].numpy(), transition[0].numpy())
        best = paths[np.argmax(scores)]
        self.assertTrue(np.array_equal(viterbi(emission, transition)[0].numpy(), best))
        labels = torch.tensor([[-1, 2, -1, -1, -1, -1, -1]])
        constrained = emission + label_mask(labels, emission.dtype)
        keep = (paths[:, 1] == 2) & (scores > INFEASIBLE)
        expected = np.logaddexp.reduce(scores[keep])
        got = float(log_partition(constrained, transition)[0])
        self.assertAlmostEqual(got, expected, places=8)

    def test_full_gene_is_grammatical_and_single_path(self):
        # ATG AAA G|GTAAG|CC TAA : exon AAAG ends at codon position 0, so the
        # intron is I1 and the next exon resumes at frame 1.
        text = "CCATGAAAGGTAAGCCTAACC"
        path = [IR, IR, E0, E1, E2, E0, E1, E2, E0,
                I1, I1, I1, I1, I1,
                E1, E2, E0, E1, E2, IR, IR]
        self.assertTrue(path_is_grammatical(encode(text)[0], torch.tensor(path)))
        broken = list(path)
        broken[9:14] = [I0] * 5
        self.assertFalse(path_is_grammatical(encode(text)[0], torch.tensor(broken)))
        # An in-frame stop may not be read through.
        readthrough = list(path)
        readthrough[19] = E0
        self.assertFalse(path_is_grammatical(encode(text)[0], torch.tensor(readthrough)))

    def test_layer_loss_gradients_and_posteriors(self):
        torch.manual_seed(0)
        text = "CCATGAAAGGTAAGCCTAACCATGTTTAAGTGA"
        sequence = encode(text).repeat(2, 1)
        length = sequence.shape[1]
        splice = torch.randn(2, length, 3, requires_grad=True)
        start_stop = torch.randn(2, length, 3, requires_grad=True)
        state = torch.randn(2, length, 5, requires_grad=True)
        labels = torch.full((2, length), -1)
        labels[1, 19:21] = IR
        labels[1, 21] = E2  # E2 directly after IR is ungrammatical
        hmm = GeneHMM()
        nll, feasible = hmm(sequence, splice, start_stop, state, labels)
        self.assertEqual(feasible.tolist(), [True, False])
        self.assertAlmostEqual(float(nll[0].detach()), 0.0, places=5)  # unconstrained
        labels[1, 21] = E0  # ATG starts at 21
        nll, feasible = hmm(sequence, splice, start_stop, state, labels)
        self.assertTrue(bool(feasible.all()))
        masked, _ = hmm(sequence, splice, start_stop, state,
                        path_masks=labels_to_masks(labels))
        torch.testing.assert_close(masked, nll)
        with self.assertRaises(ValueError):
            hmm(sequence, splice, start_stop, state)
        nll.sum().backward()
        for tensor in (splice, start_stop, state):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertIsNotNone(hmm.boundary_weight.grad)
        states, events = hmm.posteriors(sequence, splice, start_stop, state)
        torch.testing.assert_close(states.sum(-1), torch.ones(2, length))
        for name in BOUNDARIES:
            self.assertTrue(bool(((events[name] >= -1e-6) & (events[name] <= 1 + 1e-6)).all()))
        # Start events can only occur at ATG.
        atg = [i for i in range(length - 2) if text[i:i + 3] == "ATG"]
        nonzero = torch.nonzero(events["start"][0] > 1e-6).flatten().tolist()
        self.assertTrue(set(nonzero) <= set(atg))


class IsoformMaskTests(unittest.TestCase):
    # Three in-frame ATGs; isoforms start at the first (A) and third (B).
    GENE = "CCATGATGATGAAATAGCC"
    ISOFORMS = {
        "a": {"gene": "g", "chrom": "c", "strand": "+", "cds": [(2, 17)]},
        "b": {"gene": "g", "chrom": "c", "strand": "+", "cds": [(8, 17)]},
    }

    @staticmethod
    def _path(start):
        path = [IR] * 19
        for offset, position in enumerate(range(start, 17)):
            path[position] = E0 + offset % 3
        return path

    def _tracks(self, genome, transcripts):
        fetch = lambda chrom, start, end: genome[start:end]
        return build_isoform_mask_tracks(fetch, transcripts, "c", len(genome))

    def test_labels_to_masks_matches_label_mask(self):
        labels = torch.tensor([[-1, IR, E0, E1, E2, I0, I1, I2]])
        state, transition = mask_potentials(labels_to_masks(labels), torch.float64)
        torch.testing.assert_close(state, label_mask(labels, torch.float64))
        self.assertTrue(bool((transition == 0).all()))

    def test_splice_graph_partition_excludes_unannotated_start(self):
        tracks, stats = self._tracks(self.GENE, self.ISOFORMS)
        self.assertEqual(stats["supervised_genes"], 1)
        self.assertEqual(stats["multi_path_genes"], 1)
        masks = torch.from_numpy(tracks["+"].astype(np.int64)).unsqueeze(0)
        sequence, emission, boundary = random_inputs(self.GENE, seed=5)
        transition = transition_potentials(sequence, boundary)
        scores = {k: path_score(emission, transition, self._path(k)) for k in (2, 5, 8)}
        state_mask, transition_mask = mask_potentials(masks, emission.dtype)
        got = float(log_partition(emission + state_mask, transition + transition_mask)[0])
        self.assertAlmostEqual(got, np.logaddexp(scores[2], scores[8]), places=8)
        # States alone would also admit a start at the unannotated ATG at 5.
        states_only = masks & ((1 << EVENT_SHIFT) - 1) | ALL_EVENTS << EVENT_SHIFT
        state_mask, transition_mask = mask_potentials(states_only, emission.dtype)
        got = float(log_partition(emission + state_mask, transition + transition_mask)[0])
        self.assertAlmostEqual(
            got, np.logaddexp.reduce([scores[2], scores[5], scores[8]]), places=8
        )

    def test_minus_strand_and_invalid_isoform(self):
        plus, _ = self._tracks(self.GENE, self.ISOFORMS)
        genome = self.GENE.translate(COMPLEMENT)[::-1]
        length = len(genome)
        minus = {
            tid: dict(record, strand="-",
                      cds=[(length - end, length - start) for start, end in record["cds"]])
            for tid, record in self.ISOFORMS.items()
        }
        tracks, _ = self._tracks(genome, minus)
        self.assertEqual(tracks["-"][::-1].tolist(), plus["+"].tolist())
        bad = dict(self.ISOFORMS, c={"gene": "g", "chrom": "c", "strand": "+",
                                     "cds": [(3, 18)]})
        tracks, stats = self._tracks(self.GENE, bad)
        self.assertTrue((tracks["+"][2:18] == UNCONSTRAINED).all())
        self.assertEqual(stats["invalid_genes_ungrammatical"], 1)


class GenePathTests(unittest.TestCase):
    GENE = "CCATGAAAGGTAAGCCTAACC"
    PATH = [IR, IR, E0, E1, E2, E0, E1, E2, E0, I1, I1, I1, I1, I1,
            E1, E2, E0, E1, E2, IR, IR]

    def _tracks(self, genome, transcripts):
        chosen = longest_cds_transcripts(transcripts)
        fetch = lambda chrom, start, end: genome[start:end]
        return build_path_tracks(fetch, transcripts, chosen, "c", len(genome))

    def test_plus_and_minus_strand_paths(self):
        # 1-based CDS 3..9 and 15..19 on plus; mirrored on minus.
        plus = {"t": {"gene": "g", "chrom": "c", "strand": "+",
                      "cds": [(2, 9), (14, 19)]}}
        tracks, stats = self._tracks(self.GENE, plus)
        self.assertEqual(tracks["+"].tolist(), self.PATH)
        self.assertEqual(stats["supervised_transcripts"], 1)
        minus_genome = self.GENE.translate(COMPLEMENT)[::-1]
        length = len(minus_genome)
        minus = {"t": {"gene": "g", "chrom": "c", "strand": "-",
                       "cds": [(length - 19, length - 14), (length - 9, length - 2)]}}
        tracks, stats = self._tracks(minus_genome, minus)
        self.assertEqual(tracks["-"][::-1].tolist(), self.PATH)
        self.assertEqual(stats["supervised_transcripts"], 1)

    def test_longest_isoform_and_invalid_masking(self):
        transcripts = {
            "long": {"gene": "g", "chrom": "c", "strand": "+",
                     "cds": [(2, 9), (14, 19)]},
            "short": {"gene": "g", "chrom": "c", "strand": "+",
                      "cds": [(2, 5)]},
        }
        self.assertEqual(longest_cds_transcripts(transcripts), ["long"])
        bad = {"t": {"gene": "g", "chrom": "c", "strand": "+",
                     "cds": [(3, 9), (14, 20)]}}  # not starting at ATG
        tracks, stats = self._tracks(self.GENE, bad)
        self.assertTrue((tracks["+"][3:20] == -1).all())
        self.assertEqual(stats["invalid_ungrammatical"], 1)


class GeneHMMModelTests(unittest.TestCase):
    def test_state_head_checkpoint_round_trip_and_dataset(self):
        from cnn_mamba.model import SpliceMamba, model_from_checkpoint
        from cnn_mamba.train import PUDataset

        config = {"hmm_loss_weight": 0.1, "d_model": 16, "d_state": 8,
                  "n_layers": 0, "dropout": 0.0}
        model = SpliceMamba(d_model=16, d_state=8, n_layers=0, dropout=0.0,
                            gene_hmm=True)
        sequence = torch.randint(0, 4, (2, 12))
        outputs = model(sequence, return_states=True)
        self.assertEqual(tuple(outputs[-1].shape), (2, 12, 5))
        restored = model_from_checkpoint(
            {"config": config, "model_state_dict": model.state_dict()}
        )
        self.assertIsNotNone(restored.gene_hmm)
        with self.assertRaises(RuntimeError):
            SpliceMamba(d_model=16, d_state=8, n_layers=0)(sequence, return_states=True)
        with tempfile.TemporaryDirectory() as directory:
            zeros = np.zeros(12, dtype=np.int8)
            path = np.full(12, -1, dtype=np.int8)
            path[3] = E0
            np.savez(Path(directory) / "w.npz", sequence=zeros, labels=zeros,
                     start_stop_labels=zeros, chess_labels=zeros,
                     chess_start_stop_labels=zeros, path_labels=path)
            item = PUDataset(directory, require_path=True)[0]
            expected = labels_to_masks(torch.from_numpy(path.astype(np.int64)))
            self.assertEqual(item[-1].tolist(), expected.tolist())
            masks = np.full(12, UNCONSTRAINED, dtype=np.int16)
            np.savez(Path(directory) / "w.npz", sequence=zeros, labels=zeros,
                     start_stop_labels=zeros, chess_labels=zeros,
                     chess_start_stop_labels=zeros, path_masks=masks)
            item = PUDataset(directory, require_path=True)[0]
            self.assertEqual(item[-1].tolist(), masks.tolist())
            with self.assertRaises(KeyError):
                np.savez(Path(directory) / "w.npz", sequence=zeros, labels=zeros,
                         start_stop_labels=zeros, chess_labels=zeros,
                         chess_start_stop_labels=zeros)
                PUDataset(directory, require_path=True)[0]


if __name__ == "__main__":
    unittest.main()

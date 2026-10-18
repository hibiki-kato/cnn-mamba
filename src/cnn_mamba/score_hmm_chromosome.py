#!/usr/bin/env python3
"""Score plus-strand candidates with gene-HMM posterior site marginals.

Writes the same arrays as ``score_chromosome`` so ``export_uniann_scores``
consumes them unchanged. ``--score-source raw`` writes the head softmax of the
same checkpoint instead, for a matched comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pysam
import torch

from .checkpoint_provenance import checkpoint_progress
from .gene_hmm import N_STATES
from .model import model_from_checkpoint
from .score_chromosome import (
    G, LENGTHS, TASKS, atomic_json, count_candidates, encode, keep_bounds,
    masks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True, help="FASTA accession/sequence name")
    parser.add_argument("--display-chrom", default="chrX")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--score-source", choices=("hmm", "raw"), default="hmm")
    parser.add_argument(
        "--save-state-posteriors", action="store_true",
        help="Also write float16 state_posteriors.npy of shape [length, 7].",
    )
    args = parser.parse_args()
    if (args.output_dir / ".complete.json").is_file():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    window = int(checkpoint["config"]["window_size"])
    stride = int(checkpoint["config"]["stride"])
    batch_size = args.batch_size or ({5000: 8, 10000: 4, 20000: 2}.get(window, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    if model.gene_hmm is None:
        raise RuntimeError("checkpoint has no gene HMM; use score_chromosome")
    fasta = pysam.FastaFile(str(args.fasta))
    chrom_len = fasta.get_reference_length(args.chrom)
    counts = count_candidates(fasta, args.chrom, chrom_len)
    arrays = {}
    for name, count in counts.items():
        arrays[name] = {
            "positions": np.lib.format.open_memmap(args.output_dir / f"{name}_positions0.npy", mode="w+", dtype=np.uint32, shape=(count,)),
            "scores": np.lib.format.open_memmap(args.output_dir / f"{name}_scores.npy", mode="w+", dtype=np.float32, shape=(count,)),
        }
    arrays["stop"]["motif_code"] = np.lib.format.open_memmap(args.output_dir / "stop_motif_codes.npy", mode="w+", dtype=np.uint8, shape=(counts["stop"],))
    state_posteriors = (
        np.lib.format.open_memmap(
            args.output_dir / "state_posteriors.npy", mode="w+",
            dtype=np.float16, shape=(chrom_len, N_STATES),
        )
        if args.save_state_posteriors else None
    )
    offsets = {name: 0 for name in TASKS}
    starts = list(range(0, chrom_len, stride))
    started = time.time()
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start:batch_start + batch_size]
        encoded = []
        for start in batch_starts:
            text = fasta.fetch(args.chrom, start, min(chrom_len, start + window)).upper()
            encoded.append(encode(text + "N" * (window - len(text))))
        encoded = np.stack(encoded)
        tensor = torch.from_numpy(encoded).long().to(device)
        with torch.no_grad():
            splice_logits, ss_logits, state_logits = model(tensor, return_states=True)
        states, events = model.gene_hmm.posteriors(
            tensor, splice_logits, ss_logits, state_logits
        )
        if args.score_source == "hmm":
            scores = {name: events[name].clamp(0, 1).cpu().numpy() for name in TASKS}
        else:
            splice = splice_logits.float().softmax(-1).cpu().numpy()
            start_stop = ss_logits.float().softmax(-1).cpu().numpy()
            scores = {
                "donor": splice[..., 1], "acceptor": splice[..., 2],
                "start": start_stop[..., 1], "stop": start_stop[..., 2],
            }
        states = states.clamp(0, 1).cpu().numpy() if state_posteriors is not None else None
        for local_index, (start, sequence) in enumerate(zip(batch_starts, encoded)):
            index = batch_start + local_index
            left, right = keep_bounds(index, len(starts), window, stride)
            if states is not None:
                state_right = min(right, chrom_len - start)
                state_posteriors[start + left:start + state_right] = states[local_index, left:state_right]
            local_masks = masks(sequence)
            for name in TASKS:
                valid_right = min(right, chrom_len - start - LENGTHS[name] + 1)
                if valid_right <= left:
                    continue
                positions = np.flatnonzero(local_masks[name][left:valid_right]) + left
                count = len(positions)
                if not count:
                    continue
                destination = slice(offsets[name], offsets[name] + count)
                arrays[name]["positions"][destination] = start + positions
                arrays[name]["scores"][destination] = scores[name][local_index, positions]
                if name == "stop":
                    middle, last = sequence[positions + 1], sequence[positions + 2]
                    arrays[name]["motif_code"][destination] = np.where(middle == G, 2, np.where(last == G, 1, 0)).astype(np.uint8)
                offsets[name] += count
        done = min(batch_start + batch_size, len(starts))
        if done % 500 < batch_size or done == len(starts):
            print(f"processed={done:,}/{len(starts):,}", flush=True)
    fasta.close()
    for name in TASKS:
        if offsets[name] != counts[name]:
            raise RuntimeError(f"{name}: wrote {offsets[name]}, expected {counts[name]}")
        for array in arrays[name].values():
            array.flush()
    if state_posteriors is not None:
        state_posteriors.flush()
    metadata = {
        "schema": "droso_chrx_all_candidate_scores_v1", "checkpoint": str(args.checkpoint.resolve()),
        **checkpoint_progress(checkpoint),
        "fasta": str(args.fasta.resolve()), "chromosome": args.display_chrom,
        "fasta_sequence_name": args.chrom, "chromosome_length": chrom_len,
        "strand": "+", "window_size": window, "stride": stride,
        "ownership": "midpoint partition of 50%-overlap windows; chromosome edges retained",
        "position_convention": "zero-based genomic start of motif",
        "score_source": (
            "gene-HMM posterior boundary-event marginal (free start/end per window)"
            if args.score_source == "hmm" else "candidate-head softmax"
        ),
        "candidate_counts": counts, "written_counts": offsets,
        "regime": checkpoint["config"].get("regime", "unspecified"),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    atomic_json(args.output_dir / ".complete.json", {"schema": "droso_score_complete_v1", "counts": offsets})
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

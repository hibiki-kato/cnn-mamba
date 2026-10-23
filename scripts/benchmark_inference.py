#!/usr/bin/env python3
"""Measure portable inference latency and whole-process peak RAM on Unix."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import torch

from cnn_mamba.inference import (
    add_batch_arguments, add_device_argument, hardware_memory,
    configure_batch_size, resolve_device,
)
from cnn_mamba.model import model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    add_batch_arguments(parser, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    add_device_argument(parser)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1:
        parser.error("batch size, threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(saved, device).eval()
    length = int(saved["config"]["window_size"])
    memory_capacity = hardware_memory(device) if args.memory_gb is None else args.memory_gb * 1024**3
    requested_batch = args.batch_size
    args.batch_size = configure_batch_size(args, model, length, device)
    sequence = torch.randint(0, 5, (args.batch_size, length)).to(device)
    extra_outputs = dict(
        return_phase=model.phase_head is not None,
        return_cds=model.cds_head is not None,
        return_joint_frame=model.joint_frame_head is not None,
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    timings = []
    with torch.inference_mode():
        model(sequence, **extra_outputs)  # warm up
        synchronize()
        for _ in range(args.repeats):
            started = time.perf_counter()
            outputs = model(sequence, **extra_outputs)
            synchronize()
            timings.append(time.perf_counter() - started)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    print(json.dumps({
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest.hexdigest(),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "device": str(device), "backend": model.backend,
        "threads": args.threads, "batch_size": args.batch_size,
        "requested_batch_size": requested_batch,
        "memory_fraction": args.memory_fraction,
        "memory_capacity_bytes": memory_capacity,
        "window_size": length, "stride": saved["config"].get("stride"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds_per_batch": timings, "median_seconds_per_batch": statistics.median(timings),
        "process_peak_rss_mib": peak_mib,
        "memory_scope": "whole process including Python, PyTorch, checkpoint and activations; excludes device VRAM",
        "input": "seed-42 random encoded DNA; no FASTA I/O or output writes",
        "output_shapes": [list(output.shape) for output in outputs],
        "outputs_finite": all(torch.isfinite(output).all().item() for output in outputs),
        "mamba_ssm_imported": "mamba_ssm" in sys.modules,
    }, indent=2))


if __name__ == "__main__":
    main()

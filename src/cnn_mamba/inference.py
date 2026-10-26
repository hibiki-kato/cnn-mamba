"""Shared device, thread and batch defaults for portable inference."""

import os
import sys

import torch



def add_device_argument(parser, threads_default=None):
    parser.add_argument(
        "--device", default="auto",
        help="auto (CUDA/ROCm, then MPS, then CPU), cpu, mps, cuda, or cuda:N. "
             "All devices use pure PyTorch inference.",
    )
    parser.add_argument("--threads", type=int, default=threads_default,
                        help="CPU threads; overrides OMP/MKL settings. Defaults respect job allocation.")


def configure_threads(requested, device):
    """Apply explicit thread counts or a fixed default, respecting HPC settings."""
    if requested is not None:
        if requested < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(requested)
    elif not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        limit = 4 if sys.platform == "darwin" else 8
        if device.type != "cpu":
            limit = 1
        if hasattr(os, "sched_getaffinity"):
            limit = min(limit, len(os.sched_getaffinity(0)))
        allocated = os.environ.get("SLURM_CPUS_PER_TASK")
        if allocated is not None:
            limit = min(limit, int(allocated))
        torch.set_num_threads(max(1, min(limit, torch.get_num_threads())))
    return torch.get_num_threads()


def resolve_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported inference device: {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA/ROCm is unavailable; use --device cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
    return device


def resolve_batch_size(requested, device, accelerator_default):
    size = requested if requested is not None else (1 if device.type == "cpu" else accelerator_default)
    if size < 1:
        raise ValueError("batch size must be positive")
    return size

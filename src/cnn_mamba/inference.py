"""Device and memory-aware batching shared by PyTorch inference commands."""

import argparse
import os
import subprocess
import sys

import torch


def _batch_size(value):
    if value == "auto":
        return value
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size must be 'auto' or a positive integer") from error
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return size


def add_batch_arguments(parser, default=None):
    parser.add_argument(
        "--batch-size", type=_batch_size, default=default,
        help="Positive integer or auto. Default: auto on accelerators, 1 on CPU.",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.5,
                        help="Fraction of hardware capacity used for auto batch sizing (default: 0.5).")
    parser.add_argument("--memory-gb", type=float,
                        help="Override hardware capacity in GiB, e.g. an HPC job's allocation.")
    parser.add_argument("--max-batch-size", type=int, default=32,
                        help="Upper bound for auto batch sizing (default: 32).")


def hardware_memory(device):
    """Read fixed device capacity; never inspect current memory usage."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    try:
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True))
        else:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError, subprocess.CalledProcessError):
        return 0
    if device.type == "mps":
        total = min(total, torch.mps.recommended_max_memory())
    return total


def configure_batch_size(args, model, window, device):
    """Choose a batch once from hardware capacity and model/window dimensions.

    The conservative estimate includes full-length projections, residuals and
    FP32 scan intermediates. Explicit sizes take priority. No runtime memory
    monitoring, trial forwards, or automatic retries are performed.
    """
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    if args.max_batch_size < 1:
        raise ValueError("max batch size must be positive")
    if args.memory_gb is not None and not args.memory_gb > 0:
        raise ValueError("memory capacity must be positive")
    if isinstance(args.batch_size, int):
        return args.batch_size
    if args.batch_size is None and device.type == "cpu":
        return 1
    d_model = model.embedding.embedding_dim
    d_state = max((getattr(layer, "d_state", 0) for layer in model.modules()), default=0)
    per_window = 4 * (window * (32 * d_model + 8 * d_state)
                      + 16 * d_model * d_state)
    memory = hardware_memory(device) if args.memory_gb is None else args.memory_gb * 1024**3
    capacity = max(1, int(memory * args.memory_fraction) // per_window)
    capacity = min(capacity, args.max_batch_size)
    return 1 << (capacity.bit_length() - 1)


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


def add_device_argument(parser, threads_default=None):
    parser.add_argument(
        "--device", default="auto",
        help="auto (CUDA/ROCm, then MPS, then CPU), cpu, mps, cuda, or cuda:N. "
             "All devices use pure PyTorch inference.",
    )
    parser.add_argument("--threads", type=int, default=threads_default,
                        help="CPU threads; overrides OMP/MKL settings. Defaults respect job allocation.")
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32',
                        help='BF16 projections/CNN with FP32 scan accumulation; scores can differ. Default: fp32.')


def inference_precision(precision, device):
    """Explicit optional mixed precision; never switch dtype based on timing."""
    if precision not in {'fp32', 'bf16'}:
        raise ValueError(f'unsupported inference precision: {precision}')
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16')


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

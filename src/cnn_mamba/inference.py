"""Device selection shared by the PyTorch inference commands."""

import torch


def add_device_argument(parser):
    parser.add_argument(
        "--device", default="auto",
        help="auto (CUDA, then MPS, then CPU), cpu, mps, cuda, or cuda:N. "
             "All devices use pure PyTorch inference.",
    )


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
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
    return device

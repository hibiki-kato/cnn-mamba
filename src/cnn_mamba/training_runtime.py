"""Static training batch selection and optional mixed precision settings."""

import math

import torch

from .inference import _batch_size, hardware_memory


def add_training_runtime_arguments(parser):
    parser.add_argument('--batch-size', type=_batch_size, default=4,
                        help='Microbatch size or auto, selected once from total VRAM.')
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--effective-batch-size', type=int,
                        help='Samples per optimizer step. Auto defaults to 4 x --grad-accum (8).')
    parser.add_argument('--memory-fraction', type=float, default=.8,
                        help='Fraction of total VRAM budgeted for auto training batches (default: 0.8).')
    parser.add_argument('--memory-gb', type=float,
                        help='Override allocated hardware memory capacity in GiB.')
    parser.add_argument('--max-batch-size', type=int, default=32)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp-dtype', choices=('float16', 'bfloat16'), default='float16',
                        help='Mixed precision dtype with --amp; BF16 does not use gradient scaling.')
    parser.add_argument('--fuse-bidirectional', action='store_true',
                        help='Fuse independent native scan heads across directions; uses more VRAM.')


def configure_training_runtime(args, device, window, d_model=192, d_state=64, n_layers=8):
    """Set microbatch/accumulation once; preserve samples per optimizer step.

    A conservative activation estimate covers native forward/backward tensors
    for the supported CNN-Mamba architecture. Checkpointing is not credited with
    a memory discount. Capacity alone cannot guarantee fit or maximize speed.
    No free-memory queries, trial steps, OOM backoff or timing probes are used.
    Explicit batch sizes bypass the capacity calculation.
    """
    if args.grad_accum < 1 or args.max_batch_size < 1:
        raise ValueError('grad accumulation and max batch size must be positive')
    if min(window, d_model, d_state, n_layers) < 1:
        raise ValueError('model and window dimensions must be positive')
    if not 0 < args.memory_fraction <= 1:
        raise ValueError('memory fraction must be in (0, 1]')
    if args.memory_gb is not None and (not args.memory_gb > 0 or not math.isfinite(args.memory_gb)):
        raise ValueError('memory capacity must be positive')
    if args.effective_batch_size is not None and args.effective_batch_size < 1:
        raise ValueError('effective batch size must be positive')
    requested = args.batch_size
    effective = args.effective_batch_size
    memory = per_window = None
    if requested == 'auto':
        effective = 4 * args.grad_accum if effective is None else effective
        memory = hardware_memory(device) if args.memory_gb is None else args.memory_gb * 1024**3
        # FP32 parameter/gradient/Adam states and CUDA workspaces are reserved
        # separately. AMP still retains FP32 parameters and optimizer state.
        estimated_parameters = n_layers * (14*d_model**2 + 8*d_model*d_state) + 4*d_model**2
        reserved = 512 * 1024**2 + 16 * estimated_parameters
        activation_bytes = 2 if args.amp else 4
        per_window = int(1.2 * window * n_layers * (40*d_model + 16*d_state) * activation_bytes)
        if args.fuse_bidirectional:
            per_window = int(1.2 * per_window)
        capacity = max(1, int(max(0, memory * args.memory_fraction - reserved)) // per_window)
        capacity = min(capacity, args.max_batch_size, effective)
        microbatch = 1 << (capacity.bit_length() - 1)
        while effective % microbatch:
            microbatch //= 2
        args.batch_size = microbatch
    if effective is not None:
        if effective % args.batch_size:
            raise ValueError('effective batch size must be divisible by microbatch size')
        args.grad_accum = effective // args.batch_size
    return {'requested_batch_size': requested, 'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'effective_batch_size': args.batch_size * args.grad_accum,
            'memory_capacity_bytes': memory, 'estimated_activation_bytes_per_window': per_window,
            'amp_dtype': args.amp_dtype if args.amp else None}


def amp_dtype(name):
    return {'float16': torch.float16, 'bfloat16': torch.bfloat16}[name]

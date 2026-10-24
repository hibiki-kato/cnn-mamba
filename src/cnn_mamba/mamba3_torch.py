"""Pure PyTorch Mamba3 SISO used by this project's inference models.

Matches the parameter layout and equations of state-spaces/mamba at
be0303b971cd79a4fbbcfc411a01b33f6b0e3602 (modules/mamba3.py and
ops/triton/mamba3/{angle_dt,mamba3_siso_fwd}.py). Only the configuration
used by SpliceMamba is implemented: expand=2, ngroups=1, rope_fraction=0.5,
SISO, without output RMSNorm or streaming caches.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        if x.device.type == "cuda":
            return F.rms_norm(x.float(), (self.weight.numel(),),
                              self.weight.float(), self.eps).to(x.dtype)
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)


def _rotate(x, angles):
    """Rotate adjacent pairs in the first half of the state dimensions."""
    size = 2 * angles.shape[-1]
    pairs = x[..., :size].reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs.unbind(-1)
    cos, sin = angles.cos(), angles.sin()
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
    return torch.cat((rotated.flatten(-2), x[..., size:]), -1)


def siso_scan(q, k, v, adt, dt, trap, angles, q_bias, k_bias, skip, z,
              chunk_size=64, chunks_per_group=16):
    """Mamba3 scan with batched local blocks and parallel block-state propagation.

    All tensor arithmetic stays on the input device. The Python loop handles
    groups of blocks; within-block work uses batched matrix operations. Block
    propagation uses batched matrices on GPU and an associative scan on CPU.
    Scratch memory is bounded by chunk_size and chunks_per_group, not the
    square of the sequence length.
    """
    if chunk_size < 1 or chunks_per_group < 1:
        raise ValueError("chunk_size and chunks_per_group must be positive")
    batch, length, heads, width = v.shape
    if length == 0:
        return v
    dt, adt = dt.float(), adt.float()
    trap = trap.float().sigmoid()
    gamma = dt * trap
    # The next token's trapezoidal contribution affects later outputs only;
    # keep it out of the current token's diagonal below.
    scale = gamma + F.pad((dt * (1 - trap))[:, 1:], (0, 0, 0, 1))
    state = torch.zeros(batch, heads, k.shape[-1], width,
                        device=v.device, dtype=torch.float32)
    angle_state = torch.zeros(batch, heads, angles.shape[-1],
                              device=v.device, dtype=torch.float32)
    cuda_ops = v.device.type == "cuda"
    local_noncausal = None if cuda_ops else torch.ones(
        chunk_size, chunk_size, device=v.device, dtype=torch.bool,
    ).triu()
    outputs = []
    group_size = chunk_size * chunks_per_group
    for start in range(0, length, group_size):
        end = min(start + group_size, length)
        blocks = (end - start + chunk_size - 1) // chunk_size
        padding = blocks * chunk_size - (end - start)

        def pack(value):
            if padding:
                value = F.pad(value, (0, 0) * (value.ndim - 2) + (0, padding))
            return value.reshape(batch, blocks, chunk_size, *value.shape[2:])

        angle_values = angles[:, start:end].float().tanh()
        if angle_values.ndim == 3:
            angle_values = angle_values.unsqueeze(2)
        increments = pack(angle_values * math.pi * dt[:, start:end].unsqueeze(-1))
        angle_sums = increments.sum(2)
        angle_prefix = F.pad(angle_sums.cumsum(1)[:, :-1], (0, 0, 0, 0, 1, 0))
        theta = (increments.cumsum(2) + angle_prefix.unsqueeze(2)
                 + angle_state[:, None, None]).remainder(2 * math.pi)
        angle_state = (angle_state + angle_sums.sum(1)).remainder(2 * math.pi)
        qb = pack(q[:, start:end].float() + q_bias.float())
        kb = pack(k[:, start:end].float() + k_bias.float())
        diagonal = ((qb * kb).sum(-1) * pack(gamma[:, start:end])).permute(0, 3, 1, 2)
        qr = _rotate(qb, theta).permute(0, 3, 1, 2, 4)
        kr = _rotate(kb, theta).permute(0, 3, 1, 2, 4)
        kr = kr * pack(scale[:, start:end]).permute(0, 3, 1, 2).unsqueeze(-1)
        value = pack(v[:, start:end].float()).permute(0, 3, 1, 2, 4)
        cumulative = pack(adt[:, start:end]).permute(0, 3, 1, 2).cumsum(-1)
        decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp(max=0).exp()
        weights = (qr @ kr.transpose(-1, -2)) * decay
        weights = weights.tril_(-1) if cuda_ops else weights.masked_fill(local_noncausal, 0)

        # End-of-block state contributions; independent across all local blocks.
        total = cumulative[..., -1]
        weighted_k = kr * (total.unsqueeze(-1) - cumulative).exp().unsqueeze(-1)
        block_states = weighted_k.transpose(-1, -2) @ value

        # S_i = exp(A_i) S_{i-1} + U_i. A lower triangular transition
        # matrix propagates U_j into every later block in one batched matmul.
        prefix = total.cumsum(-1)
        before = F.pad(prefix[..., :-1], (1, 0))
        if v.device.type == "cpu":
            # Avoid very skinny GEMMs on CPU BLAS. A logarithmic associative
            # scan computes the same block prefix with vector operations.
            completed, alpha = block_states, total.exp()
            offset = 1
            while offset < blocks:
                tail = (completed[:, :, offset:]
                        + alpha[:, :, offset:, None, None] * completed[:, :, :-offset])
                completed = torch.cat((completed[:, :, :offset], tail), dim=2)
                alpha = torch.cat((alpha[:, :, :offset],
                                   alpha[:, :, offset:] * alpha[:, :, :-offset]), dim=2)
                offset *= 2
            incoming = F.pad(completed[:, :, :-1], (0, 0, 0, 0, 1, 0))
            incoming = incoming + before.exp()[..., None, None] * state.unsqueeze(2)
        else:
            transition = (before.unsqueeze(-1) - prefix.unsqueeze(-2)).clamp(max=0).exp()
            if cuda_ops:
                transition = transition.tril_(-1)
            else:
                causal = torch.ones(blocks, blocks, device=v.device, dtype=torch.bool).tril(-1)
                transition = transition.masked_fill(~causal, 0)
            incoming = transition @ block_states.flatten(-2)
            if cuda_ops:
                incoming = torch.addcmul(incoming, before.exp().unsqueeze(-1),
                                        state.flatten(-2).unsqueeze(-2))
            else:
                incoming = incoming + before.exp().unsqueeze(-1) * state.flatten(-2).unsqueeze(-2)
            incoming = incoming.reshape_as(block_states)

        if cuda_ops:
            out = torch.addcmul(weights @ value, qr @ incoming, cumulative.exp().unsqueeze(-1))
            out = torch.addcmul(out, (diagonal + skip.float()[None, :, None, None]).unsqueeze(-1), value)
        else:
            out = (qr @ incoming) * cumulative.exp().unsqueeze(-1) + weights @ value
            out = out + (diagonal + skip.float()[None, :, None, None]).unsqueeze(-1) * value
        out = out * F.silu(pack(z[:, start:end].float()).permute(0, 3, 1, 2, 4))
        out = out.permute(0, 2, 3, 1, 4).reshape(batch, blocks * chunk_size, heads, width)
        outputs.append(out[:, :end-start].to(v.dtype))
        if cuda_ops:
            state = torch.addcmul(block_states[:, :, -1], incoming[:, :, -1],
                                 total[:, :, -1, None, None].exp())
        else:
            state = incoming[:, :, -1] * total[:, :, -1, None, None].exp() + block_states[:, :, -1]
    return torch.cat(outputs, dim=1)


class Mamba3Torch(nn.Module):
    """Checkpoint-compatible replacement for SpliceMamba's Mamba3 layers."""

    def __init__(self, d_model, d_state=128, headdim=64, chunk_size=64,
                 chunks_per_group=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = 2 * d_model
        self.chunk_size = chunk_size
        self.chunks_per_group = chunks_per_group
        if self.d_inner % headdim:
            raise ValueError("2 * d_model must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        self.num_rope_angles = (d_state // 2) // 2
        if self.num_rope_angles < 1:
            raise ValueError("d_state must be at least 4")
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * d_state + 3 * self.nheads
            + self.num_rope_angles, bias=False,
        )
        dt = torch.exp(torch.rand(self.nheads) * math.log(0.1 / 0.001) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state))
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def scan_group_size(self, batch_size, device):
        """Choose deterministic launch geometry from input shape, without probes."""
        if self.chunks_per_group is not None:
            return self.chunks_per_group
        if device.type == "cuda":
            blocks = min(64, max(4, 384 // max(1, batch_size * self.nheads)))
            return 1 << (blocks.bit_length() - 1)
        if device.type == "cpu":
            blocks = min(16, max(4, 16 // max(1, batch_size)))
            return 1 << (blocks.bit_length() - 1)
        return 16

    def prepare(self, projected):
        """Prepare scan inputs without changing checkpoint parameter layout."""
        batch, length, _ = projected.shape
        z, x, b, c, dd_dt, dd_a, trap, angles = projected.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles], -1,
        )
        x = x.reshape(batch, length, self.nheads, self.headdim)
        z = z.reshape_as(x)
        b = self.B_norm(b).unsqueeze(2)
        c = self.C_norm(c).unsqueeze(2)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = (-F.softplus(dd_a.float())).clamp(max=-1e-4) * dt
        return (c, b, x, adt, dt, trap, angles,
                self.C_bias.squeeze(1), self.B_bias.squeeze(1), self.D, z)

    def forward(self, u):
        batch, length, _ = u.shape
        values = self.prepare(self.in_proj(u))
        # Outer autocast may use BF16 for projections, but recurrent state and
        # scan matrix operations retain FP32 accumulation on every device.
        with torch.autocast(u.device.type, enabled=False):
            y = siso_scan(*values, self.chunk_size, self.scan_group_size(batch, u.device))
        return self.out_proj(y.reshape(batch, length, self.d_inner))

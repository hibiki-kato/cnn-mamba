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
              chunk_size=64):
    """Mamba3 trapezoidal scan, with bounded block scratch space.

    q/k: [batch, length, 1, state]; v/z: [batch, length, heads, headdim].
    dt/adt/trap: [batch, length, heads]; angles: [batch, length, angles].
    Accumulation uses float32, including on MPS. The Python loop is over
    blocks, never individual bases; no length-by-length matrix is allocated.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    batch, length, heads, width = v.shape
    if length == 0:
        return v
    dt, adt = dt.float(), adt.float()
    trap = trap.float().sigmoid()
    gamma = dt * trap
    # Each K_j contributes gamma_j at j, and dt_{j+1}(1-trap_{j+1})
    # to later outputs. The latter must not leak into the diagonal term.
    scale = gamma + F.pad((dt * (1 - trap))[:, 1:], (0, 0, 0, 1))
    state = torch.zeros(batch, heads, k.shape[-1], width,
                        device=v.device, dtype=torch.float32)
    angle_state = torch.zeros(batch, heads, angles.shape[-1],
                              device=v.device, dtype=torch.float32)
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        # Match upstream's bounded, chunkwise angle accumulation.
        increments = (angles[:, start:end].float().tanh().unsqueeze(2)
                      * math.pi * dt[:, start:end].unsqueeze(-1))
        theta = (increments.cumsum(1) + angle_state.unsqueeze(1)).remainder(2 * math.pi)
        angle_state = (angle_state + increments.sum(1)).remainder(2 * math.pi)
        qb = q[:, start:end].float() + q_bias.float()
        kb = k[:, start:end].float() + k_bias.float()
        diagonal = (qb * kb).sum(-1) * gamma[:, start:end]
        qr = _rotate(qb, theta).transpose(1, 2)
        kr = _rotate(kb, theta).transpose(1, 2)
        kr = kr * scale[:, start:end].transpose(1, 2).unsqueeze(-1)
        value = v[:, start:end].float().transpose(1, 2)
        cumulative = adt[:, start:end].transpose(1, 2).cumsum(-1)
        decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp(max=0).exp()
        causal = torch.ones(end - start, end - start, device=v.device,
                            dtype=torch.bool).tril(-1)
        weights = (qr @ kr.transpose(-1, -2)) * decay
        weights = weights.masked_fill(~causal, 0)
        out = (qr @ state) * cumulative.exp().unsqueeze(-1) + weights @ value
        out = out + (diagonal.transpose(1, 2) + skip.float()[None, :, None]).unsqueeze(-1) * value
        out = out * F.silu(z[:, start:end].float().transpose(1, 2))
        outputs.append(out.transpose(1, 2).to(v.dtype))
        final_decay = cumulative[..., -1:]
        weighted_k = kr * (final_decay - cumulative).exp().unsqueeze(-1)
        state = state * final_decay.exp().unsqueeze(-1) + weighted_k.transpose(-1, -2) @ value
    return torch.cat(outputs, dim=1)


class Mamba3Torch(nn.Module):
    """Checkpoint-compatible replacement for SpliceMamba's Mamba3 layers."""

    def __init__(self, d_model, d_state=128, headdim=64, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = 2 * d_model
        self.chunk_size = chunk_size
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

    def forward(self, u):
        batch, length, _ = u.shape
        z, x, b, c, dd_dt, dd_a, trap, angles = self.in_proj(u).split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles], -1,
        )
        x = x.reshape(batch, length, self.nheads, self.headdim)
        z = z.reshape_as(x)
        b = self.B_norm(b).unsqueeze(2)
        c = self.C_norm(c).unsqueeze(2)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = (-F.softplus(dd_a.float())).clamp(max=-1e-4) * dt
        y = siso_scan(c, b, x, adt, dt, trap, angles,
                      self.C_bias.squeeze(1), self.B_bias.squeeze(1),
                      self.D, z, self.chunk_size)
        return self.out_proj(y.reshape(batch, length, self.d_inner))

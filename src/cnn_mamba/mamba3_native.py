"""Optional execution fusion for the pinned native SISO training backend."""

import torch
from torch.nn import functional as F


def _prepare(layer, u):
    if layer.is_mimo or layer.is_outproj_norm or layer.num_bc_heads != 1:
        raise ValueError('bidirectional fusion supports only SpliceMamba SISO layers')
    batch, length, _ = u.shape
    z, v, b, c, dd_dt, dd_a, trap, angles = layer.in_proj(u).split([
        layer.d_inner, layer.d_inner, layer.d_state, layer.d_state,
        layer.nheads, layer.nheads, layer.nheads, layer.num_rope_angles], -1)
    v = v.reshape(batch, length, layer.nheads, layer.headdim)
    z = z.reshape_as(v)
    b = layer.B_norm(b).unsqueeze(2)
    c = layer.C_norm(c).unsqueeze(2)
    dt = F.softplus(dd_dt + layer.dt_bias)
    adt = (-F.softplus(dd_a.float())).clamp(max=-layer.A_floor) * dt
    return dict(Q=c, K=b, V=v, Z=z, ADT=adt.transpose(1, 2),
                DT=dt.transpose(1, 2), Trap=trap.transpose(1, 2),
                Angles=angles.unsqueeze(2).expand(-1, -1, layer.nheads, -1).float(),
                Q_bias=layer.C_bias.squeeze(1), K_bias=layer.B_bias.squeeze(1), D=layer.D)


def fused_bidirectional(block, x):
    """Run both directions as independent head groups in one native scan.

    Projections and normalization retain their original per-direction weights.
    The native kernel's grouped Q/K layout keeps the two recurrences independent;
    only the backward direction is reversed again before the outputs are summed.
    Parameter layout and gradients remain unchanged. Extra concatenation buffers
    increase peak VRAM; this is opt-in and does not apply to portable inference.
    """
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    forward, backward = _prepare(block.fw, x), _prepare(block.bw, x.flip(1))
    head_axes = dict(Q=2, K=2, V=2, Z=2, ADT=1, DT=1, Trap=1,
                     Angles=2, Q_bias=0, K_bias=0, D=0)
    combined = {name: torch.cat((forward[name], backward[name]), dim=axis)
                for name, axis in head_axes.items()}
    y = mamba3_siso_combined(**combined, chunk_size=block.fw.chunk_size,
                            Input_States=None, return_final_states=False)
    a, b = y.split(block.fw.nheads, dim=2)
    a = block.fw.out_proj(a.reshape(*x.shape[:2], -1).to(forward['V'].dtype))
    b = block.bw.out_proj(b.reshape(*x.shape[:2], -1).to(backward['V'].dtype))
    return a + b.flip(1)

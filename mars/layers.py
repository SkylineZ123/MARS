"""Building blocks shared by MARS and the comparison routers.

QueryBlock   candidate-skill queries: self-attention among queries, cross-attention to history, FFN
SelfBlock    pre-norm Transformer block with an explicit boolean attention mask (windowed router)
MambaBlock   plain-PyTorch selective SSM block (S6, Gu & Dao 2023) with O(1) streaming step
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def window_indices(T: int, W: int, stride: int, device=None) -> Tensor:
    """(T, W) indices of x_t, x_{t-stride}, ..., clamped at the episode start."""
    t = torch.arange(T, device=device)[:, None]
    offsets = torch.arange(W, device=device)[None, :] * stride
    return (t - offsets).clamp_min(0)


def window_pad(T: int, W: int, stride: int, device=None) -> Tensor:
    """(T, W) bool: True where t - k*stride falls before the episode start."""
    t = torch.arange(T, device=device)[:, None]
    offsets = torch.arange(W, device=device)[None, :] * stride
    return (t - offsets) < 0


class SelfBlock(nn.Module):
    """Pre-norm Transformer block with an explicit (per-sample) boolean attention mask."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, blocked: Tensor) -> Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=blocked, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))


class QueryBlock(nn.Module):
    """Candidate-skill queries: self-attention among queries, cross-attention to the history tokens, FFN."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor, key_pad: Tensor, need_weights: bool = False):
        h = self.ln_self(q)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + self.drop(a)
        h = self.ln_cross(q)
        a, w = self.cross_attn(h, kv, kv, key_padding_mask=key_pad, need_weights=need_weights,
                               average_attn_weights=True)
        q = q + self.drop(a)
        return q + self.drop(self.ff(self.ln_ff(q))), w


class MambaBlock(nn.Module):
    """in_proj -> causal depthwise conv -> SiLU -> input-dependent (dt, B, C) -> selective scan
    -> gated by SiLU(z) -> out_proj, pre-norm residual. The sequential scan is fine for episodes
    of a few hundred steps; `step` advances the conv buffer and SSM state by one step."""

    def __init__(self, d: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dt_min: float = 1e-3,
                 dt_max: float = 1e-1):
        super().__init__()
        di = expand * d
        self.di, self.d_state, self.d_conv = di, d_state, d_conv
        self.dt_rank = math.ceil(d / 16)
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        # Mamba initialization: softplus(dt_bias) log-uniform in [dt_min, dt_max]; A = -(1..N).
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)

    def _params(self, xc: Tensor):
        dt, Bm, Cm = self.x_proj(xc).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        return F.softplus(self.dt_proj(dt)), Bm, Cm

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, d) -> (B, T, d), causal."""
        T = x.shape[1]
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        xc = F.silu(self.conv(xi.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bm, Cm = self._params(xc)
        A = -torch.exp(self.A_log)  # (di, N)
        dA = torch.exp(dt[..., None] * A)  # (B, T, di, N)
        dBx = dt[..., None] * Bm[:, :, None, :] * xc[..., None]
        state = torch.zeros(x.shape[0], self.di, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = dA[:, t] * state + dBx[:, t]
            ys.append((state * Cm[:, t, None, :]).sum(-1))
        y = (torch.stack(ys, dim=1) + xc * self.D) * F.silu(z)
        return x + self.out_proj(y)

    def init_cache(self, device=None) -> dict:
        return {"conv": torch.zeros(self.di, self.d_conv - 1, device=device),
                "state": torch.zeros(self.di, self.d_state, device=device)}

    def step(self, x_t: Tensor, cache: dict) -> Tensor:
        """x_t (d,) -> (d,); advances the conv buffer and SSM state by one step."""
        xi, z = self.in_proj(self.norm(x_t)).chunk(2, dim=-1)
        window = torch.cat([cache["conv"], xi[:, None]], dim=1)  # (di, d_conv)
        cache["conv"] = window[:, 1:]
        xc = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        dt, Bm, Cm = self._params(xc)
        A = -torch.exp(self.A_log)
        cache["state"] = torch.exp(dt[:, None] * A) * cache["state"] + dt[:, None] * Bm[None, :] * xc[:, None]
        y = ((cache["state"] * Cm[None, :]).sum(-1) + xc * self.D) * F.silu(z)
        return x_t + self.out_proj(y)

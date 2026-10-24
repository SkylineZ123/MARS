"""Reference-anchored chunk actor and twin critic (paper Sec. III-B).

Adapted from RLTChunkActor / RLTTwinCritic in
lerobot/src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py @cccf949, with an explicit
hidden-layer list (the paper's actor is an MLP with 2 hidden layers of 256 units;
the RLT `_build_mlp(num_layers=2)` only has one hidden layer).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ChunkActor(nn.Module):
    """mu^k(s_t, a_ref): input concat(z_t, q_t, flat(a_ref)), output a mean chunk (B, C, d_a)."""

    def __init__(self, z_dim: int, q_dim: int, C: int, d_a: int, hidden: list[int]):
        super().__init__()
        self.C, self.d_a = C, d_a
        self.net = mlp(z_dim + q_dim + C * d_a, hidden, C * d_a)

    def forward(self, z: Tensor, q: Tensor, ref: Tensor) -> Tensor:
        x = torch.cat([z, q, ref.reshape(ref.shape[0], -1)], dim=-1)
        return self.net(x).reshape(-1, self.C, self.d_a)


class TwinCritic(nn.Module):
    """Q_1, Q_2(s_t, a_chunk); no reference input (the reference acts through the actor)."""

    def __init__(self, z_dim: int, q_dim: int, C: int, d_a: int, hidden: list[int]):
        super().__init__()
        in_dim = z_dim + q_dim + C * d_a
        self.q1 = mlp(in_dim, hidden, 1)
        self.q2 = mlp(in_dim, hidden, 1)

    def forward(self, z: Tensor, q: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([z, q, action.reshape(action.shape[0], -1)], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, z: Tensor, q: Tensor, action: Tensor) -> Tensor:
        return self.q1(torch.cat([z, q, action.reshape(action.shape[0], -1)], dim=-1))

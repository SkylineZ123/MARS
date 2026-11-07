# Adapted from RL-Token-Pi05-open/routerlt (Apache-2.0). See NOTICE.
"""Backbone interface shared by the frozen VLA wrapper and the CPU test double.

The paper's adapter exposes, per backbone (Eq. 1):
  * prefix hidden states of the selected image / language tokens -> z_t = E(I_t, l)
  * the VLA's native proposal chunk  a~_t = A(I_t, q_t, l)  in R^{H x d_a}
Everything downstream (adapter, selector, specialists) only sees this interface and
works in the backbone's own normalized action / state space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor


@dataclass
class PrefixFeatures:
    """One frozen forward pass over the observation prefix."""

    hidden: Tensor  # (B, Lp, d_b), backbone dtype
    slices: dict[str, slice]  # token group -> positions in `hidden`
    group_masks: dict[str, Tensor]  # token group -> (B, n_group) bool validity
    state_norm: Tensor  # (B, d_q) normalized proprioception, float32
    cache: Any = None  # backbone-specific state needed to sample reference chunks
    extras: dict[str, Any] = field(default_factory=dict)


class Backbone(Protocol):
    d_b: int  # token width
    H: int  # proposal horizon
    d_a: int  # action dim
    d_q: int  # proprio dim
    c_vla: int  # actions executed per generalist query

    def preprocess(self, raw_obs: dict, task: str | list[str]) -> dict: ...

    def prefill(self, batch: dict) -> PrefixFeatures: ...

    def reference_chunk(self, feats: PrefixFeatures, noise: Tensor) -> Tensor: ...

    def make_noise(self, key: int, batch_size: int = 1) -> Tensor: ...

    def normalize_action(self, a: np.ndarray | Tensor) -> np.ndarray | Tensor: ...

    def unnormalize_action(self, a: np.ndarray | Tensor) -> np.ndarray | Tensor: ...

    def normalize_state(self, q: np.ndarray | Tensor) -> np.ndarray | Tensor: ...

    def state_from_raw(self, raw_obs: dict) -> np.ndarray: ...


def token_groups(feats: PrefixFeatures, groups: list[str]) -> tuple[Tensor, Tensor]:
    """Concatenate the requested token groups as float32 tokens plus a validity mask."""
    tokens, masks = [], []
    for name in groups:
        if name not in feats.slices:
            raise KeyError(f"token group {name!r} not in {sorted(feats.slices)}")
        tokens.append(feats.hidden[:, feats.slices[name]])
        masks.append(feats.group_masks[name])
    return torch.cat(tokens, dim=1).float(), torch.cat(masks, dim=1).bool()


def group_sizes(feats: PrefixFeatures, groups: list[str]) -> list[int]:
    return [feats.slices[g].stop - feats.slices[g].start for g in groups]


class _AffineNormalizer:
    """MEAN_STD normalization identical to lerobot's NormalizerProcessorStep."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.eps = eps

    def normalize(self, x):
        if isinstance(x, Tensor):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return (x - mean) / (std + self.eps)
        return (np.asarray(x, dtype=np.float32) - self.mean) / (self.std + self.eps)

    def unnormalize(self, x):
        if isinstance(x, Tensor):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return x * std + mean
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

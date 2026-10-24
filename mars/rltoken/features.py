"""Frozen VLA feature contract shared by the RL Token adapter and MARS."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PrefixFeatures:
    hidden: Tensor
    slices: dict[str, slice]
    group_masks: dict[str, Tensor]


def token_groups(features, groups: list[str], sizes: list[int] | None = None) -> tuple[Tensor, Tensor]:
    """Select named groups, padding each to its checkpoint width to keep slot identities fixed.

    No backbone is imported. Existing routerlt PrefixFeatures also satisfy this contract.
    Longer groups are rejected; truncate language explicitly when extracting VLA features.
    """
    if features.hidden.ndim != 3:
        raise ValueError("features.hidden must have shape (B, N, d)")
    if not groups or (sizes is not None and len(groups) != len(sizes)):
        raise ValueError("provide nonempty groups and one size per group")
    tokens, masks = [], []
    for i, group in enumerate(groups):
        x = features.hidden[:, features.slices[group]].detach().float()
        mask = features.group_masks[group].to(device=x.device, dtype=torch.bool)
        if mask.shape != x.shape[:2]:
            raise ValueError(f"{group} mask does not match its tokens")
        width = x.shape[1] if sizes is None else sizes[i]
        if x.shape[1] > width:
            raise ValueError(f"{group} has {x.shape[1]} tokens, checkpoint allows {width}")
        if x.shape[1] < width:
            padding = width - x.shape[1]
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        tokens.append(x.masked_fill(~mask[..., None], 0))
        masks.append(mask)
    return torch.cat(tokens, 1), torch.cat(masks, 1)

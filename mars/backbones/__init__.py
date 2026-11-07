"""Frozen generalist backbones; model dependencies load lazily."""
from .base import Backbone, PrefixFeatures, group_sizes, token_groups


def make_backbone(cfg):
    from .smolvla import SmolVLABackbone
    options = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
    kind = options.pop("type", "smolvla")
    if kind != "smolvla":
        raise ValueError(f"unsupported backbone {kind!r}")
    return SmolVLABackbone(**options)

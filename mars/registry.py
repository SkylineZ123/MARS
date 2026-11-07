"""Build, save and load routers by kind."""

from __future__ import annotations

import torch
from torch import nn

MARS_KINDS = ("mars", "mars_lstm")
QUERY_KINDS = (*MARS_KINDS, "windowed_query", "single_frame")
LEARNED_KINDS = (*MARS_KINDS, "mlp", "windowed_query", "single_frame")


def build_router(kind: str, z_dim: int, num_classes: int, cfg, text_dim: int = 0) -> nn.Module:
    """Build a retained router on the shared RL Token and optional proprioception."""
    from .baselines import MLPRouter, WindowedQueryRouter
    from .router import MARSRouter, SingleFrameRouter

    q_dim = int(cfg.q_dim) if cfg.use_q else 0
    common = dict(q_dim=q_dim, d_model=cfg.d_model, dropout=cfg.dropout)
    if kind == "mlp":
        return MLPRouter(z_dim, num_classes, **common)
    query = dict(heads=cfg.heads, query_layers=cfg.query_layers, ff_mult=cfg.ff_mult,
                 text_dim=text_dim if cfg.skill_query else 0, skill_query=cfg.skill_query)
    if kind in MARS_KINDS:
        return MARSRouter(z_dim, num_classes, **common, **query, layers=cfg.ssm_layers,
                          d_state=cfg.d_state, window=cfg.window,
                          trunk="mamba" if kind == "mars" else "lstm")
    if kind == "single_frame":
        # Older single-frame checkpoints always use text queries.
        return SingleFrameRouter(z_dim, num_classes, **common, **query)
    if kind == "windowed_query":
        return WindowedQueryRouter(z_dim, num_classes, **common, **query, window=cfg.window,
                                   stride=cfg.window_stride, temporal_layers=cfg.temporal_layers,
                                   state_fusion="sum")
    raise ValueError(f"no learned router of type {kind!r}")


def needs_text(kind: str, cfg) -> bool:
    return kind in QUERY_KINDS and cfg.skill_query


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_router(path, model: nn.Module, classes: list[str], extra: dict | None = None) -> None:
    """`classes` is the controller order of the outputs; deployment maps outputs to policies with it."""
    payload = {
        "kind": model.kind,
        "z_dim": model.z_dim,
        "classes": list(classes),
        "state_dict": model.state_dict(),
        "ctor": dict(model.ctor),
        "input_format": "multimodal" if getattr(model, "needs_modalities", False) else "rltoken_q",
        "hparams": {k: getattr(model, k) for k in ("window", "stride") if hasattr(model, k)},
        **(extra or {}),
    }
    torch.save(payload, path)


def _from_ctor(kind: str, ctor: dict) -> nn.Module:
    from .baselines import MLPRouter, WindowedQueryRouter
    from .router import MARSRouter, SingleFrameRouter

    classes = {"mars": MARSRouter, "mars_lstm": MARSRouter, "mlp": MLPRouter,
               "windowed_query": WindowedQueryRouter, "single_frame": SingleFrameRouter}
    if kind not in classes:
        raise ValueError(f"no learned router of type {kind!r}")
    model = classes[kind](**ctor)
    if model.kind != kind:
        raise ValueError("checkpoint kind does not match its constructor")
    return model


def load_router(path, device="cpu") -> tuple[nn.Module, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind, ctor = payload["kind"], dict(payload["ctor"])
    mode = ctor.pop("query_mode", "text")
    if kind in QUERY_KINDS:
        if ctor.get("skill_query", True) and mode != "text":
            raise ValueError(f"{path} holds learned skill-ID queries (query_mode={mode!r})")
        ctor.setdefault("text_dim", 0)
    model = _from_ctor(kind, ctor)
    model.load_state_dict(payload["state_dict"])
    if payload.get("input_format", "rltoken_q") != "rltoken_q":
        raise ValueError(f"{path} input format does not match its architecture")
    return model.to(device).eval(), payload["classes"]

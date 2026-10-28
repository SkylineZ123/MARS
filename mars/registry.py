"""Build, save and load routers by kind."""

from __future__ import annotations

import torch
from torch import nn

MARS_KINDS = ("mars", "mars_gru")
FUSION_KINDS = ("multimodal_mars", "multimodal_mars_gru")
RECURRENT_KINDS = ("mamba_head", "mamba_query", "gru_query")  # RecurrentRouter without modality fusion
WINDOWED_KINDS = ("windowed_tf", "windowed_xattn", "windowed_query")
QUERY_KINDS = ("single_frame", *FUSION_KINDS, *MARS_KINDS, *RECURRENT_KINDS, *WINDOWED_KINDS)  # may carry candidate text queries
LEARNED_KINDS = ("single_frame", *FUSION_KINDS, *MARS_KINDS, *RECURRENT_KINDS, "gru", *WINDOWED_KINDS)
# Checkpoints written by the original research code under earlier kind names.
LEGACY_KINDS = {"fusion_mamba": "mars", "fusion_gru": "mars_gru"}


def build_router(kind: str, z_dim: int, num_classes: int, cfg, text_dim: int = 0) -> nn.Module:
    """cfg: MARSConfig. z_dim is the RL Token width; multimodal_* kinds use a packed vector."""
    q_dim = int(cfg.q_dim) if cfg.use_q else 0
    if kind in MARS_KINDS:
        from .router import MARSRouter

        return MARSRouter(z_dim, num_classes, q_dim=q_dim, d_model=cfg.d_model,
                          layers=cfg.ssm_layers, d_state=cfg.d_state, window=cfg.window, heads=cfg.heads,
                          dropout=cfg.dropout, trunk="mamba" if kind == "mars" else "gru",
                          query_layers=cfg.query_layers, ff_mult=cfg.ff_mult,
                          skill_query=cfg.skill_query, text_dim=text_dim if cfg.skill_query else 0)
    if kind == "single_frame":
        from .router import SingleFrameRouter
        return SingleFrameRouter(z_dim, num_classes, q_dim=q_dim, d_model=cfg.d_model,
                                 heads=cfg.heads, dropout=cfg.dropout, text_dim=text_dim,
                                 query_layers=cfg.query_layers, ff_mult=cfg.ff_mult)
    if kind in FUSION_KINDS:
        from .router import MultimodalFusionRouter
        model = MultimodalFusionRouter(z_dim, num_classes, cfg.modality_dims, q_dim=q_dim,
                                      d_model=cfg.d_model, layers=cfg.ssm_layers, d_state=cfg.d_state,
                                      window=cfg.window, heads=cfg.heads, dropout=cfg.dropout,
                                      trunk="mamba" if kind == "multimodal_mars" else "gru",
                                      fusion=cfg.fusion, skill_query=cfg.skill_query,
                                      text_dim=text_dim if cfg.skill_query else 0)
        model.kind = kind
        return model
    if kind in RECURRENT_KINDS:
        from .router import RecurrentRouter

        sq = kind != "mamba_head"
        return RecurrentRouter(z_dim, num_classes, trunk="gru" if kind == "gru_query" else "mamba", skill_query=sq,
                               q_dim=q_dim, d_model=cfg.d_model, layers=cfg.ssm_layers, d_state=cfg.d_state,
                               window=cfg.window, stride=cfg.window_stride, heads=cfg.heads,
                               query_layers=cfg.query_layers, ff_mult=cfg.ff_mult, dropout=cfg.dropout,
                               text_dim=text_dim if sq else 0)
    if kind == "gru":
        from .baselines import GRURouter

        return GRURouter(z_dim, num_classes, cfg.proj_dim, cfg.hidden, cfg.dropout, q_dim=q_dim)
    if kind in WINDOWED_KINDS:
        from .baselines import WindowedQueryRouter

        sq = kind != "windowed_tf"
        return WindowedQueryRouter(z_dim, num_classes, window=cfg.window, stride=cfg.window_stride, q_dim=q_dim,
                                   d_model=cfg.d_model, heads=cfg.heads, temporal_layers=cfg.temporal_layers,
                                   query_layers=cfg.query_layers, ff_mult=cfg.ff_mult, dropout=cfg.dropout,
                                   temporal=kind != "windowed_xattn", skill_query=sq,
                                   text_dim=text_dim if sq else 0, state_fusion="sum")
    raise ValueError(f"no learned router of type {kind!r}")


def needs_text(kind: str, cfg) -> bool:
    """True when the router reads frozen candidate text embeddings (set them before training)."""
    if kind in (*MARS_KINDS, *FUSION_KINDS):
        return cfg.skill_query
    return kind in ("single_frame", "mamba_query", "gru_query", "windowed_xattn", "windowed_query")


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
    if kind in MARS_KINDS:
        if "modality_dims" in ctor:
            from .router import MultimodalFusionRouter
            model = MultimodalFusionRouter(**ctor)
            model.kind = "multimodal_mars" if kind == "mars" else "multimodal_mars_gru"
            return model
        from .router import MARSRouter
        return MARSRouter(**ctor)
    if kind in FUSION_KINDS:
        from .router import MultimodalFusionRouter
        model = MultimodalFusionRouter(**ctor)
        model.kind = kind
        return model
    if kind == "single_frame":
        from .router import SingleFrameRouter
        return SingleFrameRouter(**ctor)
    if kind in RECURRENT_KINDS:
        from .router import RecurrentRouter

        return RecurrentRouter(**ctor)
    if kind == "gru":
        from .baselines import GRURouter

        return GRURouter(**ctor)
    if kind in WINDOWED_KINDS:
        from .baselines import WindowedQueryRouter

        return WindowedQueryRouter(**ctor)
    raise ValueError(f"no learned router of type {kind!r}")


def load_router(path, device="cpu") -> tuple[nn.Module, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind, ctor = LEGACY_KINDS.get(payload["kind"], payload["kind"]), dict(payload["ctor"])
    # Earlier checkpoints record where the candidate queries came from; the earliest fusion
    # checkpoints predate text queries and used learned skill-ID queries.
    mode = ctor.pop("query_mode", "id" if payload["kind"] in LEGACY_KINDS else "text")
    if kind in QUERY_KINDS:
        if ctor.get("skill_query", True) and mode != "text":
            raise ValueError(f"{path} holds learned skill-ID queries (query_mode={mode!r}); "
                             "MARS builds candidate queries from the candidate texts only")
        ctor.setdefault("text_dim", 0)
    model = _from_ctor(kind, ctor)
    model.load_state_dict(payload["state_dict"])
    expected = "multimodal" if getattr(model, "needs_modalities", False) else "rltoken_q"
    if payload.get("input_format", expected) != expected:
        raise ValueError(f"{path} input format does not match its architecture")
    return model.to(device).eval(), payload["classes"]

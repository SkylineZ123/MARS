"""Flat configuration of a MARS router: architecture, candidate texts, training and switching.

`MARSConfig.from_yaml` accepts the flat field names, optionally grouped under the sections
`router`, `text`, `train` and `switch` (see configs/mars.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

# Capability descriptions of the LIBERO-Spatial candidates (generalist, Pickup, Place).
DEFAULT_SKILL_TEXT = {
    "vla": "move the robot arm to the black bowl and carry it",
    "pickup": "grasp the black bowl by its rim and lift it up",
    "place": "lower the held black bowl onto the plate and release it",
}

SECTIONS = ("router", "text", "train", "switch")


@dataclass
class MARSConfig:
    # ---- router (mars | mars_gru; ablations: mamba_head | mamba_query | gru_query | gru |
    #      windowed_query | windowed_tf | windowed_xattn)
    type: str = "mars"
    modality_dims: list[int] = field(default_factory=list)  # [z, image, image2, language] sizes
    fusion: str = "attention"  # attention | concat
    skill_query: bool = True  # False: direct linear head instead of candidate queries
    query_mode: str = "text"  # text | id | text_id
    use_q: bool = True  # read normalized proprioception q_t next to the packed observation
    q_dim: int = 8
    d_model: int = 128
    heads: int = 4
    dropout: float = 0.1
    ssm_layers: int = 2
    d_state: int = 16
    window: int = 8  # MARS: last L trunk states the queries read; windowed router: L samples
    window_stride: int = 1  # env steps between window samples (windowed router)
    temporal_layers: int = 2  # windowed router
    query_layers: int = 1
    ff_mult: int = 4
    proj_dim: int = 256  # GRU router
    hidden: int = 256  # GRU router
    # ---- candidate texts (frozen SigLIP2, encoded once before training)
    text_model: str = "ViT-B-16-SigLIP2"
    text_pretrained: str = "webli"
    text_embeddings_path: str = ""  # optional validated cache; no open_clip needed once it exists
    skill_text: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SKILL_TEXT))
    # ---- training
    epochs: int = 30
    lr: float = 1e-3
    batch_episodes: int = 16
    optimizer: str = "adamw"  # adam | adamw
    weight_decay: float = 1e-4
    select_best: bool = True  # keep the epoch with the best validation `selection_metric`
    selection_metric: str = "macro_f1"  # macro_f1 | expert_f1 (false-entry tie break) | balanced_accuracy
    seed: int = 0
    # ---- skill switching (switch.py)
    switch_tau: float = 0.6
    switch_k: int = 2
    switch_cooldown: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "MARSConfig":
        flat = {}
        for key, value in (data or {}).items():
            if key in SECTIONS and isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        unknown = sorted(set(flat) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown MARS config keys: {unknown}")
        return cls(**flat)

    @classmethod
    def from_yaml(cls, path) -> "MARSConfig":
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

"""Portable RL Token hyperparameters, adapted from routerlt/configs/schema.py."""

from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class AdapterTrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 200
    stats_frames: int = 2048
    grad_clip: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for name in ("epochs", "batch_size", "stats_frames"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.grad_clip <= 0 or self.warmup_steps < 0:
            raise ValueError("lr/grad_clip must be positive and warmup_steps nonnegative")


@dataclass
class SpecialistConfig:
    C: int = 10
    actor_hidden: list[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 256])
    sigma: float = 0.1
    gripper_sigma_scale: float = 0.5
    ref_dropout: float = 0.5
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_sigma: float = 0.1
    target_clip: float = 0.3
    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    critic_grad_clip: float = 10.0
    td_clamp_low: float = -0.5
    td_clamp_high: float = 1.5
    lambda_delta: float = 0.0
    alpha_start: int = 20000
    alpha_span: int = 50000
    ref_alignment: str = "executed"
    pad_mode: str = "ref"
    buffer_capacity: int = 300_000
    reward: str = "skill"

    def __post_init__(self):
        for name in ("C", "policy_delay", "batch_size", "buffer_capacity", "alpha_span"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("gamma", "tau", "ref_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("sigma", "target_sigma", "target_clip", "lambda_delta", "gripper_sigma_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.actor_hidden or not self.critic_hidden or min(*self.actor_hidden, *self.critic_hidden) <= 0:
            raise ValueError("actor_hidden and critic_hidden must contain positive widths")
        if min(self.lr_actor, self.lr_critic, self.critic_grad_clip) <= 0:
            raise ValueError("learning rates and critic_grad_clip must be positive")
        if self.td_clamp_low >= self.td_clamp_high:
            raise ValueError("td_clamp_low must be below td_clamp_high")
        if self.ref_alignment not in ("executed", "shifted_slice") or self.pad_mode not in ("ref", "zero"):
            raise ValueError("invalid ref_alignment or pad_mode")
        if self.reward not in ("skill", "task", "handoff_task"):
            raise ValueError("reward must be skill, task or handoff_task")


def load_config(path, section, cls):
    """Load a named section and reject typos instead of silently ignoring them."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    values = data.get(section, {})
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {section} config keys: {sorted(unknown)}")
    return cls(**values)

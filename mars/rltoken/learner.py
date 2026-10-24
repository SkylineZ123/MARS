"""TD3-style learner for one specialist (twin critics, Polyak targets, delayed actor).

Update structure follows perform_training_update in
scripts/train_rlt_stage2_pi05.py @cccf949; the losses are RouteRLT's (losses.py).
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .losses import actor_loss, critic_loss, critic_target
from .networks import ChunkActor, TwinCritic
from .config import SpecialistConfig
from .replay import TRAIN_KEYS


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters(), strict=True):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)


@contextmanager
def frozen(module: nn.Module):
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


class SpecialistLearner:
    def __init__(self, z_dim: int, q_dim: int, d_a: int, cfg, device: str = "cpu"):
        self.cfg = cfg
        if z_dim <= 0 or q_dim < 0 or d_a <= 0:
            raise ValueError("z_dim/d_a must be positive and q_dim nonnegative")
        self.z_dim, self.q_dim, self.d_a, self.C = z_dim, q_dim, d_a, cfg.C
        self.device = torch.device(device)
        self.actor = ChunkActor(z_dim, q_dim, cfg.C, d_a, cfg.actor_hidden).to(self.device)
        self.critic = TwinCritic(z_dim, q_dim, cfg.C, d_a, cfg.critic_hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.n_updates = 0

    def update(self, batch: dict[str, torch.Tensor]) -> dict:
        """One learner update; n (schedule clock) is the number of updates done so far."""
        cfg = self.cfg
        batch = {k: batch[k].to(device=self.device, dtype=torch.float32) for k in TRAIN_KEYS}
        B = batch["z"].shape[0]
        expected = {"z": (B, self.z_dim), "next_z": (B, self.z_dim),
                    "q": (B, self.q_dim), "next_q": (B, self.q_dim),
                    "ref": (B, self.C, self.d_a), "next_ref": (B, self.C, self.d_a),
                    "act": (B, self.C, self.d_a), "rew": (B, self.C), "valid": (B, self.C)}
        for key, shape in expected.items():
            if batch[key].shape != shape or not torch.isfinite(batch[key]).all():
                raise ValueError(f"{key} must be finite and have shape {shape}")
        if B == 0 or batch["done"].shape not in ((B,), (B, 1)):
            raise ValueError("nonempty batch with done (B,) or (B, 1) required")
        valid = batch["valid"]
        if not ((valid == 0) | (valid == 1)).all() or (valid.sum(-1) == 0).any() or (valid[:, 1:] > valid[:, :-1]).any():
            raise ValueError("valid must mark a nonempty contiguous prefix of executed actions")
        if not ((batch["done"] == 0) | (batch["done"] == 1)).all():
            raise ValueError("done must contain zero or one")
        n = self.n_updates

        y = critic_target(batch, self.actor_target, self.critic_target, cfg)
        loss_c, info = critic_loss(batch, self.critic, y)
        self.critic_opt.zero_grad(set_to_none=True)
        loss_c.backward()
        info["critic_grad_norm"] = float(nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip))
        self.critic_opt.step()

        if n % cfg.policy_delay == 0:
            with frozen(self.critic):
                loss_a, info_a = actor_loss(batch, self.actor, self.critic, n, cfg)
            self.actor_opt.zero_grad(set_to_none=True)
            loss_a.backward()
            self.actor_opt.step()
            polyak_update(self.actor, self.actor_target, cfg.tau)
            polyak_update(self.critic, self.critic_target, cfg.tau)
            info.update(info_a)
        self.n_updates += 1
        info["n"] = self.n_updates
        return info

    @torch.no_grad()
    def act(self, z: np.ndarray, q: np.ndarray, ref: np.ndarray, explore: bool = False,
            rng: np.random.Generator | None = None, gripper_sigma_scale: float | None = None) -> np.ndarray:
        """Normalized (C, d_a) chunk: the mean, plus fixed-std Gaussian noise when exploring (Eq. 2)."""
        was_training = self.actor.training
        self.actor.eval()
        mu = self.actor(
            torch.as_tensor(z, dtype=torch.float32, device=self.device)[None],
            torch.as_tensor(q, dtype=torch.float32, device=self.device)[None],
            torch.as_tensor(ref, dtype=torch.float32, device=self.device)[None],
        )[0].cpu().numpy()
        self.actor.train(was_training)
        if explore:
            rng = rng or np.random.default_rng()
            scale = np.full(self.d_a, self.cfg.sigma, dtype=np.float32)
            scale[-1] *= self.cfg.gripper_sigma_scale if gripper_sigma_scale is None else gripper_sigma_scale
            mu = mu + rng.standard_normal(mu.shape).astype(np.float32) * scale
        return mu.astype(np.float32)

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "n_updates": self.n_updates,
            "dims": {"z_dim": self.z_dim, "q_dim": self.q_dim, "d_a": self.d_a, "C": self.C},
        }

    def load_state_dict(self, state: dict, actor_only: bool = False) -> None:
        expected = {"z_dim": self.z_dim, "q_dim": self.q_dim, "d_a": self.d_a, "C": self.C}
        if state.get("dims", expected) != expected:
            raise ValueError(f"checkpoint dimensions differ from {expected}")
        self.actor.load_state_dict(state["actor"])
        if actor_only:
            return
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.n_updates = int(state["n_updates"])


def save_specialist(path, learner: SpecialistLearner) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "mars-rltoken-v1", "config": asdict(learner.cfg),
                "learner": learner.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}, path)


def load_specialist(path, cfg=None, device: str = "cpu", resume: bool = False) -> SpecialistLearner:
    """Load MARS or routerlt chunk specialists; pass config for older checkpoints.

    resume=True restores optimizers, targets, update clock and saved Torch RNG.
    Replay data and its sampling RNG are restored separately by ReplayBuffer.load.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    learner_state = state["learner"] if "learner" in state else state
    if learner_state.get("kind") == "dsrl":
        raise ValueError("expected an RL Token chunk specialist, got a DSRL noise-space checkpoint")
    saved_config = state.get("config")
    if cfg is None:
        if saved_config is None:
            raise ValueError("pass cfg for a routerlt checkpoint without embedded config")
        cfg = SpecialistConfig(**saved_config)
    elif resume and saved_config is not None and asdict(cfg) != saved_config:
        raise ValueError("resume config differs from saved specialist config")
    dims = learner_state["dims"]
    if dims["C"] != cfg.C:
        raise ValueError(f"checkpoint chunk length {dims['C']} != config C={cfg.C}")
    learner = SpecialistLearner(dims["z_dim"], dims["q_dim"], dims["d_a"], cfg, device=device)
    learner.load_state_dict(learner_state, actor_only=not resume)
    if resume:
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and learner.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        learner.actor.eval()
    return learner


class IdentitySpecialist:
    """Returns the generalist reference unchanged: a routing / switching dry run that
    isolates selector and skill-switch effects from RL (stage s3)."""

    def __init__(self, C: int):
        self.C = C

    def act(self, z, q, ref, explore=False, rng=None, gripper_sigma_scale=1.0) -> np.ndarray:
        return np.asarray(ref, dtype=np.float32)[: self.C].copy()

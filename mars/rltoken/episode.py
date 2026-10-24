"""Per-episode record shared by the runner, rollout storage, window builder and metrics.

Index conventions (T executed actions):
  step-indexed  (T,)   : action t, its owner/source, the reward signal after it
  obs-indexed   (T+1,) : observation t (before action t), its phase label, z, q
  query-indexed (Q,)   : every generalist query (time + full normalized chunk)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOURCE_VLA, SOURCE_RL, SOURCE_EXPERT = 0, 1, 2
SOURCE_NAMES = {SOURCE_VLA: "vla", SOURCE_RL: "rl", SOURCE_EXPERT: "expert"}
SKILLS = ("pickup", "place")


@dataclass
class EpisodeLog:
    meta: dict = field(default_factory=dict)
    # step-indexed
    exec_norm: list = field(default_factory=list)
    exec_raw: list = field(default_factory=list)
    owner: list = field(default_factory=list)  # controller that executed action t
    skill_owner: list = field(default_factory=list)  # specialist whose interval holds step t (0 = none)
    source: list = field(default_factory=list)
    c_hat: list = field(default_factory=list)
    queried: list = field(default_factory=list)
    success_after: list = field(default_factory=list)
    rewards: list = field(default_factory=list)  # optional environment rewards, same for all actors
    skill_success_after: dict = field(default_factory=lambda: {k: [] for k in SKILLS})
    goal_after: list = field(default_factory=list)  # per goal predicate (meta goal_state order), after action t
    target_id: list = field(default_factory=list)  # bound target on observation t; -1 if absent
    subtask_id: list = field(default_factory=list)
    interval_id: list = field(default_factory=list)  # changes on controller OR target OR instruction change
    # obs-indexed
    state: list = field(default_factory=list)
    q_norm: list = field(default_factory=list)
    z: list = field(default_factory=list)
    z_valid: list = field(default_factory=list)
    modal: list = field(default_factory=list)  # optional packed frozen inputs for router adaptation
    phase: list = field(default_factory=list)
    priv: list = field(default_factory=list)
    sim_state: list = field(default_factory=list)  # privileged replay audit only
    probs: list = field(default_factory=list)
    # query-indexed
    vla_t: list = field(default_factory=list)
    vla_chunks: list = field(default_factory=list)
    spec_t: list = field(default_factory=list)
    spec_id: list = field(default_factory=list)
    spec_chunks: list = field(default_factory=list)
    latent_t: list = field(default_factory=list)  # noise-space specialist queries (time + latent noise)
    latent: list = field(default_factory=list)
    # optional raw observation arrays (base collection)
    frames: dict = field(default_factory=dict)
    finalized: bool = False

    @property
    def T(self) -> int:
        return len(self.exec_norm)

    def finalize(self, d_z: int | None = None) -> EpisodeLog:
        """Convert lists to arrays (idempotent)."""
        if self.finalized:
            return self

        def arr(x, dtype, shape_tail=()):
            if len(x) == 0:
                return np.zeros((0, *shape_tail), dtype=dtype)
            return np.asarray(x, dtype=dtype)

        d_a = len(self.exec_norm[0]) if len(self.exec_norm) else 7
        self.exec_norm = arr(self.exec_norm, np.float32, (d_a,))
        self.exec_raw = arr(self.exec_raw, np.float32, (d_a,))
        self.owner = arr(self.owner, np.int8)
        self.skill_owner = arr(self.skill_owner, np.int8)
        self.source = arr(self.source, np.uint8)
        self.c_hat = arr(self.c_hat, np.int8)
        self.queried = arr(self.queried, bool)
        self.success_after = arr(self.success_after, bool)
        self.rewards = arr(self.rewards, np.float32)
        self.skill_success_after = {k: arr(v, bool) for k, v in self.skill_success_after.items()}
        self.goal_after = arr(self.goal_after, bool, (len(self.meta.get("goal_state") or []),))
        self.target_id = arr(self.target_id, np.int16)
        self.subtask_id = arr(self.subtask_id, np.int16)
        self.interval_id = arr(self.interval_id, np.int32)
        self.state = arr(self.state, np.float32, (8,))
        self.q_norm = arr(self.q_norm, np.float32, (8,))
        self.z_valid = arr(self.z_valid, bool)
        if len(self.z) and any(v is not None for v in self.z):
            dim = next(v for v in self.z if v is not None).shape[-1]
            self.z = np.stack([v if v is not None else np.zeros(dim, np.float16) for v in self.z]).astype(np.float16)
        else:
            self.z = np.zeros((len(self.z_valid), d_z or 0), dtype=np.float16)
        self.phase = arr(self.phase, np.uint8)
        self.priv = arr(self.priv, np.float32) if len(self.priv) else np.zeros((0, 0), np.float32)
        self.sim_state = arr(self.sim_state, np.float64) if len(self.sim_state) else np.zeros((0, 0), np.float64)
        if len(self.probs) and any(p is not None for p in self.probs):
            # Steps without a selector query (and the final observation) are NaN rows.
            k = len(next(p for p in self.probs if p is not None))
            self.probs = np.stack([np.full(k, np.nan, np.float32) if p is None else np.asarray(p, np.float32)
                                   for p in self.probs])
        else:
            self.probs = np.zeros((0, 0), np.float32)
        self.vla_t = arr(self.vla_t, np.int32)
        H = self.vla_chunks[0].shape[0] if len(self.vla_chunks) else 0
        self.vla_chunks = arr(self.vla_chunks, np.float32, (H, d_a))
        self.spec_t = arr(self.spec_t, np.int32)
        self.spec_id = arr(self.spec_id, np.int8)
        C = self.spec_chunks[0].shape[0] if len(self.spec_chunks) else 0
        self.spec_chunks = arr(self.spec_chunks, np.float32, (C, d_a))
        self.latent_t = arr(self.latent_t, np.int32)
        W = len(self.latent[0]) if len(self.latent) else 0
        self.latent = arr(self.latent, np.float32, (W,))
        self.frames = {k: np.asarray(v) for k, v in self.frames.items()}
        self.finalized = True
        return self

    # ------------------------------------------------------------------ derived views
    def governing_query(self, steps: np.ndarray) -> np.ndarray:
        """Index of the latest generalist query at or before each step."""
        idx = np.searchsorted(self.vla_t, steps, side="right") - 1
        if np.any(idx < 0):
            raise ValueError("a step precedes the first generalist query")
        return idx

    def ref_rows(self, steps: np.ndarray) -> np.ndarray:
        """'Executed' alignment: for each step, the row of the generalist chunk in effect.

        Steps beyond the last query continue the last chunk (clamped to its horizon).
        """
        steps = np.asarray(steps)
        q = self.governing_query(steps)
        offset = np.clip(steps - self.vla_t[q], 0, self.vla_chunks.shape[1] - 1)
        return self.vla_chunks[q, offset]

    def shifted_slice(self, t: int, C: int) -> np.ndarray:
        """'Shifted slice' alignment: C consecutive rows of the chunk in effect at step t."""
        q = int(self.governing_query(np.asarray([t]))[0])
        rows = np.clip(np.arange(C) + t - self.vla_t[q], 0, self.vla_chunks.shape[1] - 1)
        return self.vla_chunks[q, rows]


def save_episode(path, log: EpisodeLog) -> None:
    from pathlib import Path
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.finalize()
    torch.save(vars(log), path)


def load_episode(path) -> EpisodeLog:
    import torch

    return EpisodeLog(**torch.load(path, map_location="cpu", weights_only=False)).finalize()

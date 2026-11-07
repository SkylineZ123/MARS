# Adapted from RL-Token-Pi05-open/routerlt (Apache-2.0). See NOTICE.
"""Deterministic keys for matched (Base vs RouteRLT) episodes.

VLA flow-matching noise is keyed by (episode, env step t), not by query count, so
Base and RouteRLT draw identical generalist chunks at t = 0, C_VLA, 2*C_VLA, ...
until the first handoff, and an off-grid re-query after a handoff is still
reproducible. Exploration noise uses a separate generator so it never perturbs
the VLA noise stream.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _hash(*parts) -> int:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 62) - 1)


def episode_key(suite: str, task_id: int, init_id: int, seed: int) -> int:
    return _hash("episode", suite, task_id, init_id, seed)


def vla_noise_key(ep_key: int, t: int) -> int:
    return _hash("vla", ep_key, t)


def env_seed(ep_key: int) -> int:
    return _hash("env", ep_key) % (2**31 - 1)


def explore_rng(ep_key: int, salt: str = "explore") -> np.random.Generator:
    return np.random.default_rng(_hash(salt, ep_key))

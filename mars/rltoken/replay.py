"""Numpy ring buffer of chunk-level windows.

Adapted from ReplayBuffer in scripts/train_rlt_stage2_pi05.py @cccf949 (lazy
allocation from the first record, fp16 -> fp32 on sample); batch insertion and
npz persistence added, threading removed (the sim loop is synchronous).
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

PHASE_WARMUP, PHASE_ONLINE = 0, 1
TRAIN_KEYS = ("z", "q", "ref", "act", "rew", "valid", "done", "next_z", "next_q", "next_ref")
LATENT_KEYS = ("z", "q", "act", "rew", "valid", "done", "next_z", "next_q")  # noise-space specialist


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0, keys: tuple[str, ...] = TRAIN_KEYS):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.keys = keys
        self._storage: dict[str, np.ndarray] | None = None
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def _allocate(self, record: dict[str, np.ndarray]) -> None:
        self._storage = {k: np.empty((self.capacity, *v.shape[1:]), dtype=v.dtype) for k, v in record.items()}

    def add_batch(self, windows: dict[str, np.ndarray], collection_phase: int, episode_id: int) -> int:
        if not windows:
            return 0
        n = len(windows["z"])
        if n == 0:
            return 0
        if set(self.keys) - set(windows):
            raise KeyError(f"missing training keys: {sorted(set(self.keys) - set(windows))}")
        if any(len(value) != n for value in windows.values()):
            raise ValueError("all window arrays must have the same length")
        record = dict(windows)
        record["collection_phase"] = np.full(n, collection_phase, dtype=np.uint8)
        record["episode_id"] = np.full(n, episode_id, dtype=np.int32)
        if self._storage is None:
            self._allocate(record)
        missing = set(self._storage) ^ set(record)
        if missing:
            raise KeyError(f"window keys differ from buffer schema: {sorted(missing)}")
        # Avoid repeated fancy-assignment indices when a batch exceeds capacity.
        skip = max(0, n - self.capacity)
        idx = (self._position + np.arange(skip, n)) % self.capacity
        for key, value in record.items():
            if value.shape[1:] != self._storage[key].shape[1:]:
                raise ValueError(f"{key} shape {value.shape[1:]} != {self._storage[key].shape[1:]}")
            self._storage[key][idx] = value[skip:]
        self._position = int((self._position + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)
        return n

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {k: torch.from_numpy(self._storage[k][idx].astype(np.float32)) for k in self.keys}

    def stats(self) -> dict:
        if self._storage is None or self._size == 0:
            return {"size": 0}
        phase = self._storage["collection_phase"][: self._size]
        rew = self._storage["rew"][: self._size]
        out = {
            "size": self._size,
            "warmup": int((phase == PHASE_WARMUP).sum()),
            "online": int((phase == PHASE_ONLINE).sum()),
            "success_frac": float((rew.sum(axis=1) > 0).mean()),
        }
        if "source" in self._storage:
            out["expert_frac"] = float((self._storage["source"][: self._size] == 2).any(axis=1).mean())
        return out

    def save(self, path: Path) -> None:
        if self._storage is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v[: self._size] for k, v in self._storage.items()},
                 _meta=np.asarray([self._position, self._size, self.capacity]),
                 _keys=np.asarray(self.keys),
                 _rng=np.asarray(json.dumps(self._rng.bit_generator.state)))

    @classmethod
    def load(cls, path: Path, capacity: int | None = None, seed: int = 0) -> ReplayBuffer:
        data = np.load(path)
        position, size, saved_capacity = (int(x) for x in data["_meta"])
        keys = tuple(str(k) for k in data["_keys"]) if "_keys" in data else TRAIN_KEYS
        buf = cls(saved_capacity if capacity is None else capacity, seed=seed, keys=keys)
        record = {k: data[k] for k in data.files if not k.startswith("_")}
        buf._allocate(record)
        n = min(size, buf.capacity)
        order = (position + np.arange(size)) % size if size == saved_capacity else np.arange(size)
        selected = order[-n:]
        for k, v in record.items():
            # Preserve physical layout when capacity is unchanged (sampling RNG parity).
            buf._storage[k][:n] = v[:n] if buf.capacity == saved_capacity else v[selected]
        buf._size = n
        buf._position = position if buf.capacity == saved_capacity else n % buf.capacity
        if "_rng" in data:
            buf._rng.bit_generator.state = json.loads(str(data["_rng"]))
        data.close()
        return buf

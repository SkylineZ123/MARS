"""Streaming deployment: one router query -> candidate probabilities -> k-of-k skill switch.

    ctrl = MARSController.from_checkpoint("mars.pt", cfg)
    ctrl.reset()                                  # at every episode start
    for t in range(max_steps):
        if t % query_interval == 0:
            active, switched, probs = ctrl.query(z_t, q_t)
            if switched:
                action_queue.clear()              # the old controller's chunk is invalid
        if not action_queue:
            action_queue.extend(policies[ctrl.active_name](obs))
        env.step(action_queue.popleft())

`z_t` is the shared RL Token, `q_t` the normalized
proprioception. Output index i of the router is controller `classes[i]`; index 0 is the
generalist that holds control at the episode start.
"""

from __future__ import annotations

import numpy as np
import torch

from .registry import load_router
from .switch import SkillSwitcher


class MARSController:
    def __init__(self, router, classes: list[str], tau: float = 0.6, k: int = 2, cooldown: int = 0,
                 device="cpu"):
        self.router = router.eval()
        self.classes = list(classes)
        self.device = torch.device(device)
        self.uses_q = bool(getattr(router, "q_dim", 0))
        self.switcher = SkillSwitcher(tau, k, cooldown, vla_id=0)
        self.reset()

    @classmethod
    def from_checkpoint(cls, path, cfg=None, device="cpu", **switch) -> "MARSController":
        """Switch values come from `switch` (tau, k, cooldown), else from cfg, else the defaults."""
        router, classes = load_router(path, device=device)
        if cfg is not None:
            switch = {"tau": cfg.switch_tau, "k": cfg.switch_k, "cooldown": cfg.switch_cooldown, **switch}
        return cls(router, classes, device=device, **switch)

    def reset(self) -> None:
        """Start a new episode: empty router memory, generalist in control."""
        self.state = self.router.init_state(self.device)
        self.switcher.reset()

    @property
    def active(self) -> int:
        return self.switcher.active

    @property
    def active_name(self) -> str:
        return self.classes[self.switcher.active]

    def query(self, z, q=None) -> tuple[int, bool, np.ndarray]:
        """One router query; returns (active controller, whether it changed, probabilities (K,))."""
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)
        if self.uses_q:
            if q is None:
                raise ValueError(f"{self.router.kind} router was trained with proprioception; pass q")
            q_t = torch.as_tensor(np.asarray(q, dtype=np.float32), device=self.device)
            probs, self.state = self.router.step(z_t, self.state, q_t)
        else:
            probs, self.state = self.router.step(z_t, self.state)
        probs = probs.float().cpu().numpy()
        switched = self.switcher.update(int(probs.argmax()), probs)
        return self.switcher.active, switched, probs

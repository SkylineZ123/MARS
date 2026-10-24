"""Skill switching: a score threshold with a k-of-k debounce (after Robo-MLT's System-2 monitor).

At every router query the top-scoring controller j is a trigger when it is not the active
controller and its score reaches tau. The active controller becomes j once j has triggered on k
consecutive queries; a query without a trigger, or one naming another candidate, restarts the
count. Entering a specialist, returning to the generalist and a direct specialist-to-specialist
handoff all follow this one rule. After a switch, `cooldown` queries count no trigger (Robo-MLT's
switch cooldown). Below the threshold the current controller keeps control.

Routers that return a hard decision without scores (rules, scripted schedules) switch at once:
the debounce filters noisy scores, not rules.

Every switch invalidates the queued action chunk of the previous controller; the caller discards
it and queries the new controller at the same step (Robo-MLT, Algorithm 1).
"""

from __future__ import annotations

import numpy as np


class SkillSwitcher:
    def __init__(self, tau: float = 0.6, k: int = 2, cooldown: int = 0, vla_id: int = 0):
        if k < 1:
            raise ValueError("switch k must be >= 1")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("switch tau must lie in [0, 1]")
        if cooldown < 0:
            raise ValueError("switch cooldown must be >= 0")
        self.tau, self.k, self.cooldown, self.vla_id = float(tau), int(k), int(cooldown), vla_id
        self.reset()

    def reset(self) -> None:
        self.active = self.vla_id
        self.cand = -1  # controller currently accumulating triggers
        self.count = 0  # consecutive triggers of `cand`
        self.cooling = 0  # queries left in the post-switch cooldown
        self.switches = 0

    def _switch(self, controller: int) -> bool:
        self.active = controller
        self.cand, self.count = -1, 0
        self.cooling = self.cooldown
        self.switches += 1
        return True

    def update(self, c_hat: int, probs=None) -> bool:
        """Process one router query; True when the active controller changed."""
        if probs is None:
            return self._switch(int(c_hat)) if c_hat != self.active else False
        if self.cooling > 0:
            self.cooling -= 1
            self.cand, self.count = -1, 0
            return False
        p = np.asarray(probs, dtype=np.float64)
        j = int(p.argmax())
        if j == self.active or p[j] < self.tau:
            self.cand, self.count = -1, 0
            return False
        self.count = self.count + 1 if j == self.cand else 1
        self.cand = j
        return self._switch(j) if self.count >= self.k else False


def replay_switch(probs: np.ndarray, tau: float, k: int, cooldown: int = 0) -> np.ndarray:
    """Active controller per step when a (T, K) probability sequence is queried once per step."""
    sw = SkillSwitcher(tau, k, cooldown)
    owners = np.zeros(len(probs), dtype=np.int64)
    for t, p in enumerate(probs):
        sw.update(int(np.argmax(p)), p)
        owners[t] = sw.active
    return owners

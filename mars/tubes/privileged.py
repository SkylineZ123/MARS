"""Privileged simulator state and phase labels; used only for labels, oracle routing,
rewards and scripted corrections, never as a policy input."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def tilt_deg_from_wxyz(quat_wxyz: np.ndarray) -> float:
    """Angle between the body z-axis and world z (0 = upright)."""
    _, x, y, _ = np.asarray(quat_wxyz, dtype=np.float64)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))


@dataclass
class ObjectInfo:
    name: str
    pos: np.ndarray  # (3,) world position of the body / site
    quat: np.ndarray  # (4,) wxyz
    z0: float  # z after the reset settle steps
    grasped: bool = False
    is_site: bool = False

    @property
    def tilt_deg(self) -> float:
        return tilt_deg_from_wxyz(self.quat)

    @property
    def lift(self) -> float:
        return float(self.pos[2] - self.z0)


@dataclass
class PrivilegedState:
    eef_pos: np.ndarray
    gripper_qpos: np.ndarray
    objects: dict[str, ObjectInfo]
    predicates: dict[tuple, bool] = field(default_factory=dict)
    success: bool = False

    def obj(self, name: str) -> ObjectInfo:
        return self.objects[name]

    def grasped_names(self) -> list[str]:
        return [n for n, o in self.objects.items() if o.grasped and not o.is_site]


@dataclass
class PhaseLabel:
    kind: str
    obj: str | None
    target: str | None
    raw_kind: str
    # Skill success of the object that was active *before* this observation
    # (evaluated on the post-action state, i.e. the reward signal).
    skill_success: dict[str, bool] = field(default_factory=dict)
    # Goal predicates (goal_state order) on this observation, for predicate-gated routing.
    goals: tuple = ()

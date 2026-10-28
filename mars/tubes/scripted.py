"""Feedback demonstration controller; privileged, training / qualification only."""
from __future__ import annotations
import numpy as np

from .environment import GRIP_ROT, RACK_FLOOR, TABLE_Z, TIP

SAFE_Z = TABLE_Z + .26


class ScriptedTubes:
    def __init__(self, env):
        self.env = env
        self.stage = -1
        self.moves = []
        self.move = self.dwell = self.age = 0

    def _begin(self):
        e = self.env
        self.stage = e.stage_index
        self.move = self.dwell = self.age = 0
        self.moves = []
        i = e.active_tube

        def add(target, grip, dwell=2, tol=.008, speed=.9):
            # A callable target is re-evaluated every step (closed loop on the tube itself).
            self.moves.append((target if callable(target) else np.asarray(target).copy(), grip, dwell, tol, speed))

        if e.phase != "insert":
            # Release, then leave the previous contact straight upwards before lateral travel.
            add(e.eef, -1, 6, .02)
            add([*e.eef[:2], max(e.eef[2], SAFE_Z)], -1)
        if e.phase == "grasp":
            def grasp():
                return e.grasp_point(i)
            add(lambda: grasp() + [0, 0, .10], -1)
            add(grasp, -1, 4, .004, .4)
            add(grasp, 1, 12, .01, .4)
            # Straight up out of the source slot before any lateral travel.
            add([*e.slot(i)[:2], SAFE_Z], 1, 2, .01, .5)
        elif e.phase == "insert":
            h = e.hole(e.target_hole(i))

            def tube_goal(z):
                goal = np.r_[h[:2], z]
                return lambda: e.eef + goal - e.tube_pos(i)
            add([*e.eef[:2], SAFE_Z], 1)
            add(tube_goal(h[2] + TIP + .03), 1)
            add(tube_goal(h[2] + TIP + .03), 1, 8, .0015, .3)
            add(tube_goal(TABLE_Z + RACK_FLOOR + TIP - .004), 1, 8, .006, .25)
            add(tube_goal(TABLE_Z + RACK_FLOOR + TIP - .004), -1, 14, .01, .25)
        elif e.phase == "home":
            add(e.home, -1, 8, .02)

    def perturbable(self, grip, last_grip):
        """Recovery data only disturbs free-space travel high above both racks.

        Grasp and insertion contact stay unperturbed: their precision is what the
        specialists are trained for.
        """
        speed = self.moves[self.move][4]
        return grip == last_grip and speed >= .9 and self.env.eef[2] >= TABLE_Z + .15

    def act(self):
        from robosuite.utils.control_utils import orientation_error
        if self.stage != self.env.stage_index:
            self._begin()
        if not self.moves:
            return np.r_[np.zeros(6), -1].astype(np.float32)
        target, grip, wait, tol, speed = self.moves[self.move]
        target = target() if callable(target) else target
        error = target - self.env.eef
        rotation_error = orientation_error(GRIP_ROT, self.env.rotation)
        self.dwell = self.dwell + 1 if np.linalg.norm(error) < tol and np.linalg.norm(rotation_error) < .1 else 0
        self.age += 1
        if self.dwell >= wait and self.move < len(self.moves)-1:
            self.move += 1
            self.dwell = self.age = 0
        # OSC_POSE maps action 1 to 5 cm and 0.5 rad of goal offset.
        return np.r_[np.clip(error/.05, -speed, speed),
                     np.clip(rotation_error/.5, -.5, .5), grip].astype(np.float32)

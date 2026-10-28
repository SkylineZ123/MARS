"""Panda tube-transfer workcell on robosuite 1.4 / MuJoCo (task tubes_v2).

Three tubes start in a fixed source rack. The robot moves them one by one, from
left to right, into the matching holes of a target rack whose pose is randomised.
Table, robot placement, start pose, controller and cameras follow LIBERO's tabletop
scenes and observations use LiberoEnv's raw layout, so a LIBERO-trained SmolVLA
transfers. Only reset writes object coordinates; everything else is physical contact.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np

from .protocol import SKILL_KINDS, SUITE, TubesSpec, physical_seed

TABLE_Z = .90
TABLE_SIZE = (1.0, 1.2, .05)
# LIBERO tabletop agentview and MountedPanda start pose; joint 7 is turned a quarter
# so the fingers open along x, across the racks' rows of holes.
AGENTVIEW = ((0.6586131746834771, 0.0, 1.6103500240372423),
             (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349))
INIT_QPOS = np.array([0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, -np.pi / 4])
GRIP_ROT = np.diag([-1., 1., -1.])

TUBE_R, TUBE_HALF, CAP_R, CAP_HALF = .008, .045, .0095, .006
TIP = TUBE_HALF + TUBE_R            # body origin to the rounded bottom
GRASP_Z = .030                      # grasp point on the tube body, below the cap
SOURCE_XY = np.array([-.04, -.17])  # fixed source rack (orange), robot's right
RACK_XY = np.array([-.04, .13])     # nominal target rack (green), robot's left
RACK_PITCH, RACK_TOP, RACK_PLATE, RACK_FLOOR = .045, .060, .008, .008
RACK_HALF = (.03, .08)
INSERT_DEPTH = .045
HOME_TOL = .04
SOURCE_COLOR, TARGET_COLOR = (.85, .55, .2, 1), (.2, .6, .55, 1)


def _geom(parent, name, gtype, pos, size, rgba, **kwargs):
    collision = ET.SubElement(parent, "geom", name=name, type=gtype, pos=" ".join(map(str, pos)),
                              size=" ".join(map(str, size)), rgba=" ".join(map(str, rgba)),
                              friction="1 .005 .0001", group="0", **kwargs)
    # robosuite renders only group 1; the visual copy carries no mass or contacts.
    visual = dict(collision.attrib)
    visual.update(name=name + "_visual", group="1", contype="0", conaffinity="0", density="0")
    ET.SubElement(parent, "geom", **visual)
    return collision


def hole_offsets(clearance):
    """Rack-frame xy of the three holes, left (+y) first, and the square hole width."""
    return [np.array([0., RACK_PITCH * (1 - k)]) for k in range(3)], .016 + clearance


def _rack(parent, name, xy, clearance, color, site):
    """Floor plate, top plate cut into strips around three square holes, end walls."""
    rack = ET.SubElement(parent, "body", name=name, pos=f"{xy[0]} {xy[1]} {TABLE_Z}")
    hx, hy = RACK_HALF
    holes, width = hole_offsets(clearance)
    z = RACK_TOP - RACK_PLATE / 2
    _geom(rack, f"{name}_floor", "box", (0, 0, RACK_FLOOR / 2), (hx, hy, RACK_FLOOR / 2), color)
    for sign in (-1, 1):
        _geom(rack, f"{name}_end{sign}", "box", (0, sign * (hy - .003), (RACK_FLOOR + z) / 2),
              (hx, .003, (z - RACK_FLOOR) / 2), color)
        band = (hx - width / 2) / 2
        _geom(rack, f"{name}_side{sign}", "box", (sign * (width / 2 + band), 0, z),
              (band, hy, RACK_PLATE / 2), color)
    edges = [-hy] + [e for h in sorted(o[1] for o in holes) for e in (h - width / 2, h + width / 2)] + [hy]
    for j in range(0, len(edges), 2):
        lo, hi = edges[j], edges[j + 1]
        _geom(rack, f"{name}_bar{j // 2}", "box", (0, (lo + hi) / 2, z),
              (width / 2, (hi - lo) / 2, RACK_PLATE / 2), color)
    for k, o in enumerate(holes):
        ET.SubElement(rack, "site", name=f"{site}{k}", pos=f"{o[0]} {o[1]} {RACK_TOP}",
                      size=".003", rgba="0 0 0 0")
    return rack


def require_python_transforms():
    """tubes physics is versioned with robosuite's transforms running as plain Python.

    numba-compiled transforms (robosuite's default) round differently, so replays and
    paired evaluations would drift by ~1e-3 over an episode. Export NUMBA_DISABLE_JIT=1
    before robosuite is imported.
    """
    from robosuite.utils import transform_utils
    if hasattr(transform_utils.mat2quat, "py_func"):
        raise RuntimeError(f"{SUITE} requires NUMBA_DISABLE_JIT=1 before robosuite is imported "
                           "(export it in the shell or set it in conftest)")


def make_workcell(render=True, image_size=256, control_freq=20, clearance=.002):
    # Import lazily: protocol / dataset tooling and CPU tests need no simulator.
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.tasks import ManipulationTask
    from robosuite.controllers import load_controller_config

    class TubeWorkcell(SingleArmEnv):
        def _load_model(self):
            super()._load_model()
            robot = self.robots[0].robot_model
            robot.set_base_xpos(robot.base_xpos_offset["table"](TABLE_SIZE[0]))
            arena = TableArena(table_full_size=TABLE_SIZE, table_offset=(0, 0, TABLE_Z))
            arena.set_origin((0, 0, 0))
            arena.set_camera("agentview", pos=list(AGENTVIEW[0]), quat=list(AGENTVIEW[1]))
            world = arena.worldbody
            _rack(world, "source", SOURCE_XY, clearance, SOURCE_COLOR, "slot")
            _rack(world, "rack", RACK_XY, clearance, TARGET_COLOR, "hole")
            self.tube_geoms = []
            for i in range(3):
                tube = ET.SubElement(world, "body", name=f"tube{i}", pos=f"0 0 {.5 + .2 * i}")
                ET.SubElement(tube, "freejoint", name=f"tube{i}_joint")
                common = dict(condim="4", density="700", solref=".01 1")
                _geom(tube, f"tube{i}_glass", "capsule", (0, 0, 0), (TUBE_R, TUBE_HALF),
                      (.78, .86, .95, 1), **common)
                _geom(tube, f"tube{i}_cap", "cylinder", (0, 0, TUBE_HALF + CAP_HALF), (CAP_R, CAP_HALF),
                      (.15, .35, .85, 1), **common)
                ET.SubElement(tube, "site", name=f"tube{i}_grasp", pos=f"0 0 {GRASP_Z}",
                              size=".003", rgba="0 0 0 0")
                self.tube_geoms.append([f"tube{i}_glass", f"tube{i}_cap"])
            self.model = ManipulationTask(mujoco_arena=arena,
                                         mujoco_robots=[r.robot_model for r in self.robots],
                                         mujoco_objects=[])

        def reward(self, action=None):
            return 0.0  # Protocol reward and termination live in TubesEnv.

        def _check_success(self):
            return False

    # robosuite's default OSC_POSE, as LIBERO uses it.
    controller = load_controller_config(default_controller="OSC_POSE")
    return TubeWorkcell(robots="Panda", controller_configs=controller,
                        initialization_noise=None, use_camera_obs=False,
                        has_renderer=False, has_offscreen_renderer=render,
                        camera_names=["agentview", "robot0_eye_in_hand"],
                        camera_heights=image_size, camera_widths=image_size,
                        control_freq=control_freq, horizon=100000, ignore_done=True,
                        hard_reset=False)


class TubesEnv:
    suite_name = SUITE
    skills = tuple(SKILL_KINDS.values())

    def __init__(self, spec=None, render=True, image_size=256):
        self.spec = spec or TubesSpec()
        self.max_steps = self.spec.max_steps
        self.task_id = self.spec.order_id
        self.task_description = self.spec.instruction
        self.image_size, self.render_enabled = image_size, render
        self.domain = make_workcell(render, image_size, self.spec.control_freq, self.spec.clearance)
        require_python_transforms()
        self.sim = self.domain.sim
        self.robot = self.domain.robots[0]
        self.tube_ids = [self.sim.model.body_name2id(f"tube{i}") for i in range(3)]
        self.tube_addr = [self.sim.model.get_joint_qpos_addr(f"tube{i}_joint")[0] for i in range(3)]
        self.tube_vel = [self.sim.model.get_joint_qvel_addr(f"tube{i}_joint")[0] for i in range(3)]
        self.grasp_sites = [self.sim.model.site_name2id(f"tube{i}_grasp") for i in range(3)]
        self.hole_sites = [self.sim.model.site_name2id(f"hole{k}") for k in range(3)]
        self.slot_sites = [self.sim.model.site_name2id(f"slot{k}") for k in range(3)]
        self.rack_id = self.sim.model.body_name2id("rack")
        self.rack_home = self.sim.model.body_pos[self.rack_id].copy()
        # robosuite integrates gripper commands in this attribute and sim.reset() keeps it.
        self._gripper_action0 = np.array(self.robot.gripper.current_action, copy=True)
        self.hole_half = hole_offsets(self.spec.clearance)[1] / 2
        self.num_init_states = 30100
        self.goal_state = [["inserted", f"tube{i}", f"hole{self.spec.order[i]}"] for i in range(self.spec.parts)]
        self.stages = []
        self.total_stages = len(self.spec.stages)

    def stage_name(self, i):
        return self.spec.stages[i]

    # ------------------------------------------------------------------ reset
    def _place(self, i, xy):
        """Stand tube i upright on a rack floor (both racks share the hole geometry)."""
        a, v = self.tube_addr[i], self.tube_vel[i]
        self.sim.data.qpos[a:a + 7] = np.r_[xy, TABLE_Z + RACK_FLOOR + TIP + .0005, 1, 0, 0, 0]
        self.sim.data.qvel[v:v + 6] = 0

    def reset(self, init_id=0, seed=0):
        from scipy.spatial.transform import Rotation
        self.scene_id, self.seed = init_id, seed
        rng = np.random.default_rng(physical_seed(init_id, seed))
        self.domain.reset()
        self.robot.gripper.current_action = self._gripper_action0.copy()
        s = self.spec
        yaw = np.deg2rad(rng.uniform(-s.rack_yaw_deg, s.rack_yaw_deg))
        self.sim.model.body_pos[self.rack_id] = self.rack_home + np.r_[rng.uniform(-s.rack_xy, s.rack_xy, 2), 0]
        self.sim.model.body_quat[self.rack_id] = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)
        qpos = INIT_QPOS + rng.normal(0, s.joint_noise, 7)
        self.sim.data.qpos[self.robot._ref_joint_pos_indexes] = qpos
        self.sim.data.qvel[self.robot._ref_joint_vel_indexes] = 0
        self.sim.forward()
        # Tube i starts in source slot i (left to right); the source rack never moves.
        for i in range(3):
            self._place(i, self.slot(i)[:2])
        self.sim.forward()
        self.robot.controller.update(force=True)
        self.robot.controller.reset_goal()
        for _ in range(15):
            self.domain.step(np.r_[np.zeros(6), -1])
        self.home = self.eef.copy()
        self.t = self.stage_index = self._stable = 0
        self._toppled = 0
        self.events, self.completed = [], []
        self.last_reward = 0.0
        self.last_skill_success = dict.fromkeys(self.skills, False)
        self._finished = False
        return self._obs()

    # ------------------------------------------------------------------ state queries
    @property
    def eef(self):
        return self.sim.data.site_xpos[self.robot.eef_site_id].copy()

    @property
    def rotation(self):
        return self.sim.data.site_xmat[self.robot.eef_site_id].reshape(3, 3).copy()

    @property
    def gripper_opening(self):
        q = self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes]
        return float(q[0] - q[1])

    @property
    def active_tube(self):
        return min(self.stage_index // 2, self.spec.parts - 1)

    @property
    def phase(self):
        return "done" if self.stage_index >= self.total_stages else self.spec.stages[self.stage_index]

    def tube_pos(self, i):
        return self.sim.data.body_xpos[self.tube_ids[i]].copy()

    def tube_axis(self, i):
        return self.sim.data.body_xmat[self.tube_ids[i]].reshape(3, 3)[:, 2].copy()

    def tube_tip(self, i):
        return self.tube_pos(i) - TIP * self.tube_axis(i)

    def tube_tilt(self, i):
        return float(np.degrees(np.arccos(np.clip(self.tube_axis(i)[2], -1, 1))))

    def grasp_point(self, i):
        return self.sim.data.site_xpos[self.grasp_sites[i]].copy()

    def hole(self, k):
        return self.sim.data.site_xpos[self.hole_sites[k]].copy()

    def slot(self, k):
        return self.sim.data.site_xpos[self.slot_sites[k]].copy()

    def target_hole(self, i):
        return self.spec.order[i]

    def grasped(self, i):
        return bool(self.domain._check_grasp(self.robot.gripper, self.domain.tube_geoms[i]))

    def lifted(self, i):
        return self.tube_tip(i)[2] >= TABLE_Z + RACK_TOP + .015

    def seated(self, i):
        k = self.target_hole(i); h = self.hole(k); tip = self.tube_tip(i)
        return (tip[2] <= h[2] - INSERT_DEPTH and np.linalg.norm(self.tube_pos(i)[:2] - h[:2]) <= self.hole_half + .002
                and self.tube_tilt(i) <= 15 and not self.grasped(i) and self.gripper_opening >= .064)

    def _condition(self):
        i = self.active_tube
        return {
            "grasp": lambda: self.grasped(i) and self.lifted(i),
            "insert": lambda: self.seated(i),
            "home": lambda: (np.linalg.norm(self.eef - self.home) < HOME_TOL
                             and all(self.seated(j) for j in range(self.spec.parts))),
            # Waiting after completion must never create additional milestones or reward.
            "done": lambda: False,
        }[self.phase]()

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self._finished:
            raise RuntimeError("episode ended; reset before stepping again")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("action must be a finite (7,) vector")
        self.domain.step(np.clip(a, -1, 1))
        self.t += 1
        self.last_reward = 0.0
        self.last_skill_success = dict.fromkeys(self.skills, False)
        self._stable = self._stable + 1 if self._condition() else 0
        if self._stable >= self.spec.stable_steps:
            phase = self.phase
            self.events.append({"step": self.t, "stage": self.stage_index, "phase": phase,
                                "tube": self.active_tube})
            if phase in SKILL_KINDS:
                self.last_skill_success[SKILL_KINDS[phase]] = True
            if phase == "insert":
                self.completed.append(self.active_tube)
            self.stage_index += 1
            self._stable = 0
            self.last_reward = 1.0 / self.total_stages
        live = range(3)
        down = [i for i in live if not self.grasped(i) and self.tube_tilt(i) > 60]
        self._toppled = self._toppled + 1 if down else 0
        success = self.stage_index == self.total_stages and all(self.seated(i) for i in range(self.spec.parts))
        failure = ("dropped_off_table" if any(self.tube_pos(i)[2] < TABLE_Z - .1 for i in live) else
                   "tube_toppled" if self._toppled >= 40 else None)
        failed = failure is not None and not success
        truncated = self.t >= self.max_steps and not success and not failed
        self._finished = success or failed or truncated
        if success:
            self.last_reward += 1
        return self._obs(), success, {"is_success": success, "terminated": success or failed,
                  "truncated": truncated, "reward": self.last_reward, "phase": self.phase,
                  "stage_index": self.stage_index, "failure": failure if failed else None}

    def _obs(self):
        # LiberoEnv(obs_type="pixels_agent_pos") layout; LiberoProcessorStep does the rest.
        o = self.domain._get_observations(force_update=True)
        rs = {"eef": {"pos": o["robot0_eef_pos"], "quat": o["robot0_eef_quat"],
                      "mat": self.robot.controller.ee_ori_mat.copy()},
              "gripper": {"qpos": o["robot0_gripper_qpos"], "qvel": o["robot0_gripper_qvel"]},
              # LIBERO activates robosuite's (inactive by default) joint_pos observable.
              "joints": {"pos": self.sim.data.qpos[self.robot._ref_joint_pos_indexes].copy(),
                         "vel": o["robot0_joint_vel"]}}
        pixels = {}
        if self.render_enabled:
            for key, camera in (("image", "agentview"), ("image2", "robot0_eye_in_hand")):
                # Raw robosuite frames (OpenGL convention), exactly as LIBERO returns them.
                pixels[key] = self.sim.render(camera_name=camera, height=self.image_size,
                                              width=self.image_size).copy()
        return {"pixels": pixels, "robot_state": rs}

    def physics_state(self):
        return self.sim.get_state().flatten().copy()

    def privileged(self):
        from .privileged import ObjectInfo, PrivilegedState
        objects = {}
        for i in range(3):
            objects[f"tube{i}"] = ObjectInfo(f"tube{i}", self.tube_pos(i),
                self.sim.data.body_xquat[self.tube_ids[i]].copy(), TABLE_Z + RACK_FLOOR + TIP,
                self.grasped(i))
        for k in range(3):
            objects[f"hole{k}"] = ObjectInfo(f"hole{k}", self.hole(k), np.array([1., 0, 0, 0]),
                                             TABLE_Z + RACK_TOP, is_site=True)
            objects[f"slot{k}"] = ObjectInfo(f"slot{k}", self.slot(k), np.array([1., 0, 0, 0]),
                                             TABLE_Z + RACK_TOP, is_site=True)
        success = self.stage_index == self.total_stages and all(self.seated(i) for i in range(self.spec.parts))
        return PrivilegedState(self.eef, self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes].copy(),
                objects, {("inserted", f"tube{i}"): i in self.completed for i in range(self.spec.parts)}, success)

    def make_labeler(self, cfg):
        from .privileged import PhaseLabel
        env = self

        class TubesLabeler:
            """Ground-truth skill windows with entry/exit hysteresis against boundary flicker."""

            def reset(self, priv):
                self.kind, self.stage = None, -1
                return self.update(priv)

            def update(self, priv):
                phase, i = env.phase, env.active_tube
                previous = self.kind if self.stage == env.stage_index else None
                obj, target, kind = f"tube{i}", None, "approach"
                if phase == "grasp":
                    d = np.linalg.norm(env.eef - env.grasp_point(i))
                    if d < .06 or (previous == "pickup" and d < .08):
                        kind = "pickup"
                elif phase == "insert":
                    k = env.target_hole(i); target = f"hole{k}"
                    d = np.linalg.norm(env.tube_pos(i)[:2] - env.hole(k)[:2])
                    if d < .05 or (previous == "insert" and d < .07):
                        kind = "insert"
                    elif env.grasped(i):
                        kind = "transport"
                elif phase == "done":
                    kind = "done"
                self.kind, self.stage = kind, env.stage_index
                return PhaseLabel(kind, obj, target, kind, dict(env.last_skill_success),
                                  tuple(priv.predicates.values()))

        return TubesLabeler()

    # ------------------------------------------------------------------ training snapshots
    def snapshot(self):
        """Full physics + controller + protocol state; only for training entry curricula."""
        import copy
        return {"physics": self.physics_state(), "rack_pos": self.sim.model.body_pos[self.rack_id].copy(),
                "rack_quat": self.sim.model.body_quat[self.rack_id].copy(),
                "ctrl": self.sim.data.ctrl.copy(), "gripper_action": np.array(self.robot.gripper.current_action),
                "stage_index": self.stage_index,
                "stable": self._stable, "t": self.t, "events": copy.deepcopy(self.events),
                "completed": self.completed.copy(), "home": self.home.copy(),
                "toppled": self._toppled, "spec": self.spec.fingerprint}

    def restore_training_snapshot(self, state):
        if state["spec"] != self.spec.fingerprint:
            raise ValueError("snapshot protocol mismatch")
        import copy
        self.sim.model.body_pos[self.rack_id] = state["rack_pos"]
        self.sim.model.body_quat[self.rack_id] = state["rack_quat"]
        self.sim.set_state_from_flattened(state["physics"])
        self.sim.data.ctrl[:] = state["ctrl"]
        self.robot.gripper.current_action = state["gripper_action"].copy()
        self.sim.forward()
        self.robot.controller.update(force=True)
        self.robot.controller.reset_goal()
        self.stage_index, self._stable, self.t = state["stage_index"], state["stable"], state["t"]
        self.events, self.completed = copy.deepcopy(state["events"]), state["completed"].copy()
        self.home, self._toppled = state["home"].copy(), state["toppled"]
        self.last_skill_success = dict.fromkeys(self.skills, False)
        self._finished = False
        return self._obs()

    def close(self):
        self.domain.close()


class TubesEnvPool:
    def __init__(self, cfg):
        self.cfg, self.env, self.task_id = cfg, None, None

    def get(self, task_id):
        if self.task_id != task_id:
            self.close()
            self.env = TubesEnv(TubesSpec(parts=self.cfg.assembly_parts, order_id=task_id,
                                          max_steps=self.cfg.max_steps or 1200), image_size=self.cfg.obs_size)
            self.task_id = task_id
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = self.task_id = None

"""Versioned tube-transfer task and scene splits. Simulator truth is never a policy input."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

SUITE = "tubes_v2"
# Source slots and target holes are numbered from the robot's left (+y). Tube k leaves
# source slot k for target hole ORDERS[order][k]; the only order is left to right.
ORDERS = ((0, 1, 2),)
TRAIN_ORDERS = (0,)
HELDOUT_ORDERS = ()
SPLITS = {"train": range(0, 10000), "val": range(10000, 10050),
          "calibration": range(20000, 20050), "test": range(30000, 30100)}
PHASES = ("grasp", "insert")
CONTROLLERS = ("vla", "grasp", "insert")
# Phase-label kind served by each specialist; env.last_skill_success uses these keys.
SKILL_KINDS = {"grasp": "pickup", "insert": "insert"}


@dataclass(frozen=True)
class TubesSpec:
    version: str = SUITE
    parts: int = 3  # tubes moved per episode
    order_id: int = 0
    max_steps: int = 1200
    control_freq: int = 20
    stable_steps: int = 5
    clearance: float = 0.002  # both racks; smallest of {2, 3} mm with a 50/50 scripted pilot
    rack_xy: float = 0.04
    rack_yaw_deg: float = 15.0
    joint_noise: float = 0.02
    cameras: int = 2

    def __post_init__(self):
        if self.cameras != 2:
            raise ValueError("tubes uses the two LIBERO cameras")
        if not 1 <= self.parts <= 3 or not 0 <= self.order_id < len(ORDERS):
            raise ValueError("parts must be 1..3 and order_id must index ORDERS")
        if self.max_steps < 1 or self.stable_steps < 1 or self.control_freq < 1:
            raise ValueError("budgets and frequencies must be positive")
        if not .0015 <= self.clearance <= .006:
            raise ValueError("clearance must be in [1.5, 6] mm")
        if not 0 <= self.rack_xy <= .06 or not 0 <= self.rack_yaw_deg <= 30 or not 0 <= self.joint_noise <= .05:
            raise ValueError("randomization out of range")

    @property
    def order(self):
        return ORDERS[self.order_id][:self.parts]

    @property
    def stages(self):
        return ("grasp", "insert") * self.parts + ("home",)

    @property
    def instruction(self):
        tubes = {1: "the leftmost test tube", 2: "the two leftmost test tubes", 3: "the three test tubes"}[self.parts]
        return (f"Move {tubes} from the orange rack to the green rack one by one from left to right, "
                "inserting each into the matching hole, then return to the start position.")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def check_split(split, scene_id, order_id, heldout=False):
    if split not in SPLITS or scene_id not in SPLITS[split]:
        raise ValueError(f"scene {scene_id} is not in {split} split")
    if heldout:
        raise ValueError("tubes has no held-out orders")
    if order_id not in TRAIN_ORDERS:
        raise ValueError(f"order {order_id} not allowed in this protocol")


def physical_seed(scene_id, seed=0):
    return int.from_bytes(hashlib.blake2s(f"{SUITE}:{scene_id}:{seed}".encode(),
                                        digest_size=4).digest(), "little")


def source_fingerprint():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in ("environment.py", "scripted.py", "protocol.py"):
        h.update((root / name).read_bytes())
    return h.hexdigest()

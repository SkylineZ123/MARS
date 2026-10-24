"""MARS: Memory-Augmented Router for atomic Skills."""

from .baselines import GRURouter, WindowedQueryRouter
from .config import DEFAULT_SKILL_TEXT, MARSConfig
from .controller import MARSController
from .registry import LEARNED_KINDS, build_router, count_params, load_router, needs_text, save_router
from .router import MARSRouter, RecurrentRouter, pack_modalities
from .switch import SkillSwitcher, replay_switch
from .train import predict, train_router

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SKILL_TEXT", "GRURouter", "LEARNED_KINDS", "MARSConfig", "MARSController", "MARSRouter",
    "RecurrentRouter", "SkillSwitcher", "WindowedQueryRouter", "build_router", "count_params", "load_router",
    "needs_text", "pack_modalities", "predict", "replay_switch", "save_router", "train_router",
]

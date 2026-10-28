"""Tube-transfer workcell: three tubes from a fixed source rack into a target rack.

Only the protocol is imported eagerly; ``mars.tubes.environment`` needs robosuite 1.4
(``pip install mars-router[tubes]``) and ``NUMBA_DISABLE_JIT=1``.
"""
from .protocol import SPLITS, SUITE, TubesSpec, check_split, physical_seed

__all__ = ["SPLITS", "SUITE", "TubesSpec", "check_split", "physical_seed"]

import numpy as np
import pytest

from mars.switch import SkillSwitcher, replay_switch

VLA, PICK, PLACE = 0, 1, 2


def p(*xs):
    return np.array(xs, dtype=np.float64)


def test_switch_needs_k_consecutive_triggers_above_tau():
    s = SkillSwitcher(tau=0.6, k=2)
    assert not s.update(PLACE, p(0.3, 0.0, 0.7))  # first trigger
    assert not s.update(VLA, p(0.5, 0.0, 0.5))  # below tau: streak broken
    assert not s.update(PLACE, p(0.2, 0.0, 0.8))
    assert s.update(PLACE, p(0.1, 0.0, 0.9)) and s.active == PLACE
    assert s.switches == 1


def test_exit_and_reentry_follow_the_same_rule():
    s = SkillSwitcher(tau=0.6, k=2)
    for _ in range(2):
        s.update(PLACE, p(0.1, 0.0, 0.9))
    assert not s.update(VLA, p(0.7, 0.0, 0.3))
    assert s.update(VLA, p(0.8, 0.0, 0.2)) and s.active == VLA
    s.update(PLACE, p(0.1, 0.0, 0.9))
    assert s.update(PLACE, p(0.1, 0.0, 0.9)) and s.active == PLACE  # re-entry


def test_a_different_candidate_restarts_the_count():
    s = SkillSwitcher(tau=0.5, k=2)
    assert not s.update(PICK, p(0.2, 0.7, 0.1))
    assert not s.update(PLACE, p(0.2, 0.1, 0.7))  # new candidate: count 1
    assert s.update(PLACE, p(0.2, 0.1, 0.7)) and s.active == PLACE


def test_direct_specialist_to_specialist_switch():
    s = SkillSwitcher(tau=0.6, k=1)
    assert s.update(PICK, p(0.1, 0.8, 0.1)) and s.active == PICK
    assert s.update(PLACE, p(0.1, 0.2, 0.7)) and s.active == PLACE


def test_cooldown_counts_no_trigger():
    s = SkillSwitcher(tau=0.6, k=1, cooldown=2)
    assert s.update(PLACE, p(0.1, 0.0, 0.9))
    assert not s.update(VLA, p(0.9, 0.0, 0.1))  # cooling
    assert not s.update(VLA, p(0.9, 0.0, 0.1))  # cooling
    assert s.update(VLA, p(0.9, 0.0, 0.1)) and s.active == VLA


def test_hard_decisions_switch_at_once():
    s = SkillSwitcher(tau=0.9, k=5, cooldown=10)
    assert s.update(PLACE) and s.active == PLACE  # no scores: a rule, applied immediately
    assert not s.update(PLACE)
    assert s.update(VLA) and s.active == VLA


def test_tau_zero_k_one_is_argmax_switching():
    probs = np.array([[0.6, 0.4], [0.4, 0.6], [0.55, 0.45], [0.3, 0.7]])
    assert replay_switch(probs, 0.0, 1).tolist() == probs.argmax(-1).tolist()


def test_replay_matches_online():
    probs = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
    owners = replay_switch(probs, 0.6, 2)
    assert owners.tolist() == [0, 0, 1, 1, 0, 0]


def test_confirmation_filters_boundary_flicker():
    probs = np.array([[.49, .51], [.51, .49]] * 4 + [[.1, .9]] * 3)
    owners = replay_switch(probs, .6, 2)
    assert (owners[:9] == 0).all()
    assert (owners[9:] == 1).all()
    assert (np.diff(owners) != 0).sum() < (np.diff(probs.argmax(-1)) != 0).sum()


def test_bad_parameters():
    with pytest.raises(ValueError):
        SkillSwitcher(k=0)
    with pytest.raises(ValueError):
        SkillSwitcher(tau=1.5)
    with pytest.raises(ValueError):
        SkillSwitcher(cooldown=-1)

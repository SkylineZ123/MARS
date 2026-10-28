import os

# Physics is versioned with robosuite's transforms as plain Python; set before numba loads.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest

pytest.importorskip("robosuite")


@pytest.fixture(scope="module")
def scripted_episode():
    """One full physical three-tube episode of the privileged teacher."""
    from mars.tubes.environment import TubesEnv
    from mars.tubes.scripted import ScriptedTubes
    env = TubesEnv(render=False)
    obs = env.reset(10000)
    first = env.physics_state()
    start = [(env.tube_pos(i), env.slot(i)) for i in range(3)]
    teacher, labeler = ScriptedTubes(env), env.make_labeler(None)
    label = labeler.reset(env.privileged())
    trace = {"kinds": [label.kind], "rewards": [], "pickup": [], "insert": [], "grasped": [],
             "obs": obs, "start": start}
    while True:
        obs, success, info = env.step(teacher.act())
        label = labeler.update(env.privileged())
        trace["kinds"].append(label.kind)
        trace["rewards"].append(info["reward"])
        trace["pickup"].append(env.last_skill_success["pickup"])
        trace["insert"].append(env.last_skill_success["insert"])
        trace["grasped"].append(tuple(i for i in range(3) if env.grasped(i)))
        if success or info["terminated"] or info["truncated"]:
            break
    trace.update(success=success, info=info, events=list(env.events), steps=env.t, first=first)
    yield env, trace
    env.close()


def test_scripted_teacher_completes_every_milestone_once(scripted_episode):
    env, trace = scripted_episode
    assert trace["success"] and env.stage_index == env.total_stages == 7
    phases = [e["phase"] for e in trace["events"]]
    assert phases == list(env.spec.stages)
    assert sum(trace["pickup"]) == 3 and sum(trace["insert"]) == 3
    assert sum(trace["rewards"]) == pytest.approx(2.0)  # 7 x 1/7 milestones + 1 for success


def test_tubes_move_left_to_right_from_source_slots_into_matching_holes(scripted_episode):
    env, trace = scripted_episode
    for pos, slot in trace["start"]:
        assert np.linalg.norm(pos[:2] - slot[:2]) < 1e-3  # every tube starts in its source slot
    # Tubes are carried one at a time, leftmost first.
    carried = [held[0] for held in trace["grasped"] if held]
    assert all(len(held) <= 1 for held in trace["grasped"])
    assert [t for k, t in enumerate(carried) if k == 0 or carried[k-1] != t] == [0, 1, 2]
    assert [e["tube"] for e in trace["events"] if e["phase"] == "insert"] == [0, 1, 2]
    for i in range(3):
        assert env.seated(i) and np.linalg.norm(env.tube_pos(i)[:2] - env.hole(i)[:2]) < .003


def test_skill_labels_enter_each_window_once(scripted_episode):
    _, trace = scripted_episode
    kinds = trace["kinds"]
    entries = [b for a, b in zip(kinds, kinds[1:]) if a != b]
    assert entries.count("pickup") == 3 and entries.count("insert") == 3
    assert kinds[-1] == "done"


def test_completed_protocol_cannot_issue_more_milestone_rewards():
    from mars.tubes.environment import TubesEnv
    env = TubesEnv(render=False)
    try:
        env.reset(10001)
        env.stage_index = env.total_stages  # e.g. a seated tube was knocked out afterwards
        for _ in range(10):
            _, success, info = env.step(np.r_[np.zeros(6), -1])
            assert not success and info["reward"] == 0
        assert env.stage_index == env.total_stages and not env.events
    finally:
        env.close()


def test_reset_is_deterministic_and_snapshots_restore_physics():
    from mars.tubes.environment import TubesEnv
    from mars.tubes.scripted import ScriptedTubes
    env = TubesEnv(render=False)
    try:
        env.reset(10002)
        a = env.physics_state()
        env.reset(10003)
        assert not np.allclose(env.physics_state(), a)
        env.reset(10002)
        np.testing.assert_array_equal(env.physics_state(), a)
        teacher = ScriptedTubes(env)
        for _ in range(60):
            env.step(teacher.act())
        state, actions, after = env.snapshot(), [], []
        for _ in range(20):
            actions.append(teacher.act()); env.step(actions[-1]); after.append(env.physics_state())
        env.restore_training_snapshot(state)
        for action, expected in zip(actions, after):
            env.step(action)
            np.testing.assert_allclose(env.physics_state(), expected, atol=1e-8)
    finally:
        env.close()

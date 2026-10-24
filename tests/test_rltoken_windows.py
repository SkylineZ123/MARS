import numpy as np

from mars.rltoken.episode import SOURCE_EXPERT, SOURCE_RL, SOURCE_VLA, EpisodeLog
from mars.rltoken.windows import build_windows, skill_intervals

D_A, H, C = 2, 6, 3


def make_log(owner, source=None, success_at=None, end_reason="budget", vla_t=(0, 3, 6), z_valid=None):
    T = len(owner)
    log = EpisodeLog(meta={"end_reason": end_reason})
    log.exec_norm = [np.array([t, -t], dtype=np.float32) for t in range(T)]
    log.exec_raw = list(log.exec_norm)
    log.owner = list(owner)
    log.skill_owner = list(owner)
    log.source = list(source if source is not None else [SOURCE_RL if o else SOURCE_VLA for o in owner])
    log.c_hat = [0] * T
    log.queried = [False] * T
    log.success_after = [False] * T
    place = [False] * T
    if success_at is not None:
        place[success_at] = True
    log.skill_success_after = {"pickup": [False] * T, "place": place}
    log.state = [np.zeros(8, np.float32)] * (T + 1)
    log.q_norm = [np.full(8, t, np.float32) for t in range(T + 1)]
    log.z = [np.full(4, t, np.float16) for t in range(T + 1)]
    log.z_valid = list(z_valid if z_valid is not None else [True] * (T + 1))
    log.phase = [0] * (T + 1)
    log.vla_t = list(vla_t)
    # chunk q has rows 100*q + row index
    log.vla_chunks = [np.stack([np.full(D_A, 100 * q + r, np.float32) for r in range(H)]) for q in range(len(vla_t))]
    return log.finalize()


def test_intervals_truncate_at_success_and_classify_end():
    owner = np.array([0, 1, 1, 1, 0, 1, 1])
    succ = np.array([0, 0, 1, 0, 0, 0, 0], dtype=bool)
    ivs = skill_intervals(owner, succ, 1, budget_end=True)
    assert [(i.start, i.end, i.reason) for i in ivs] == [(1, 3, "success"), (5, 7, "budget")]
    ivs = skill_intervals(owner, np.zeros(7, bool), 1, budget_end=False)
    assert [i.reason for i in ivs] == ["handoff", "handoff"]


def test_overlapping_windows_lengths_and_done_flags():
    owner = [0, 1, 1, 1, 1, 0, 0, 0]
    log = make_log(owner)  # interval [1, 5) ends by handoff
    w, stats = build_windows(log, 1, "place", C)
    assert stats["windows"] == 4
    assert w["step"].tolist() == [1, 2, 3, 4]
    assert w["L"].tolist() == [3, 3, 2, 1]
    assert w["done"].tolist() == [0, 1, 1, 1]  # windows reaching the handoff end are terminal
    assert w["valid"][2].tolist() == [1, 1, 0]
    assert w["rew"].sum() == 0
    np.testing.assert_array_equal(w["act"][0][:, 0], [1, 2, 3])
    np.testing.assert_array_equal(w["next_z"][0], np.full(4, 4))  # s_{t+L} = obs 4


def test_budget_end_keeps_bootstrap():
    owner = [0, 0, 1, 1]
    log = make_log(owner, end_reason="budget")
    w, _ = build_windows(log, 1, "place", C)
    assert w["done"].tolist() == [0, 0]
    np.testing.assert_array_equal(w["next_z"][0], np.full(4, 4))


def test_environment_reward_is_shared_by_task_and_skill_training():
    log = make_log([1] * 6, vla_t=(0, 3))
    log.rewards = np.array([0., .1, 0., .2, 0., 1.], np.float32)
    task, _ = build_windows(log, 1, "task", 3)
    local, _ = build_windows(log, 1, "place", 3)
    np.testing.assert_array_equal(task["rew"], local["rew"])
    np.testing.assert_allclose(task["rew"][3], [.2, 0., 1.])


def test_whole_task_windows_stop_at_instruction_and_bootstrap_next_instruction():
    log = make_log([1] * 6, vla_t=(0, 3))
    log.subtask_id = np.array([0, 0, 0, 1, 1, 1])
    log.success_after[-1] = True
    windows, stats = build_windows(log, 1, "task", 4)
    assert windows["L"].tolist() == [3, 2, 1, 3, 2, 1]
    assert windows["done"].tolist() == [0, 0, 0, 1, 1, 1]
    assert not windows["rew"][:3].any()
    assert stats["reasons"]["continuation"] == 1
    np.testing.assert_array_equal(windows["next_z"][2], np.full(4, 3))


def test_success_reward_on_last_valid_row():
    owner = [0, 1, 1, 1, 1, 1]
    log = make_log(owner, success_at=3, end_reason="success")
    w, stats = build_windows(log, 1, "place", C)
    assert stats["reasons"]["success"] == 1
    assert w["step"].tolist() == [1, 2, 3]  # truncated at the success step
    assert w["rew"][0].tolist() == [0, 0, 1]
    assert w["rew"][2].tolist() == [1, 0, 0]
    assert w["done"].tolist() == [1, 1, 1]


def test_executed_reference_alignment_across_requery():
    owner = [0, 0, 1, 1, 1, 1, 1, 1, 1]
    log = make_log(owner, vla_t=(0, 3, 6))
    w, _ = build_windows(log, 1, "place", C, alignment="executed")
    # window at t=2: rows for steps 2 (chunk0 row2), 3 (chunk1 row0), 4 (chunk1 row1)
    np.testing.assert_array_equal(w["ref"][0][:, 0], [2, 100, 101])
    # shifted slice keeps one chunk: chunk0 rows 2..4
    w2, _ = build_windows(log, 1, "place", C, alignment="shifted_slice")
    np.testing.assert_array_equal(w2["ref"][0][:, 0], [2, 3, 4])


def test_expert_rows_replace_reference_and_padding():
    owner = [0, 1, 1, 1, 0]
    source = [SOURCE_VLA, SOURCE_RL, SOURCE_EXPERT, SOURCE_EXPERT, SOURCE_VLA]
    log = make_log(owner, source=source, vla_t=(0,))
    w, _ = build_windows(log, 1, "place", C, alignment="executed", pad_mode="ref")
    np.testing.assert_array_equal(w["ref"][0][:, 0], [1, 2, 3])  # rows 2,3 replaced by expert actions
    assert w["source"][0].tolist() == [SOURCE_RL, SOURCE_EXPERT, SOURCE_EXPERT]
    # window at t=3 has L=1; padded act rows copy the reference rows beyond the interval
    np.testing.assert_array_equal(w["act"][2][:, 0], [3, 4, 5])
    w0, _ = build_windows(log, 1, "place", C, pad_mode="zero")
    np.testing.assert_array_equal(w0["act"][2][1:], 0)


def test_missing_next_state_drops_nonterminal_window():
    owner = [0, 1, 1, 1, 1, 1]
    z_valid = [True, True, True, True, False, True, True]
    log = make_log(owner, z_valid=z_valid, end_reason="budget")
    w, stats = build_windows(log, 1, "place", C)
    assert 1 not in w["step"].tolist()  # next state obs 4 has no z and d=0
    assert stats["dropped"] >= 1


def test_ref_rows_clamp_to_horizon():
    log = make_log([1] * 9, vla_t=(0,))
    rows = log.ref_rows(np.arange(4, 12))
    assert rows[:, 0].tolist() == [4, 5, 5, 5, 5, 5, 5, 5]


def test_task_reward_uses_task_success():
    owner = [1] * 6  # a whole-episode specialist owns every step
    log = make_log(owner, success_at=2, end_reason="success")  # place succeeds early, the task at the end
    log.success_after[-1] = True
    w, stats = build_windows(log, 1, "task", C)
    assert stats["intervals"] == 1 and stats["reasons"]["success"] == 1
    assert w["step"].tolist() == list(range(6))  # not truncated at the skill success
    assert w["rew"].sum() == 3 and w["rew"][5].tolist() == [1, 0, 0]  # every window covering the last step
    assert w["done"][-1] == 1


def test_handoff_earns_discounted_later_task_outcome():
    owner = [0, 1, 1, 0, 0, 0]  # specialist hands back at step 3; the task succeeds after action 5
    log = make_log(owner, end_reason="success")
    log.success_after[5] = True
    gamma = 0.5
    w, stats = build_windows(log, 1, "task", C, handoff_gamma=gamma)
    assert stats["reasons"]["handoff"] == 1 and stats["handoff_credit"] == [gamma ** 2]
    assert w["step"].tolist() == [1, 2] and w["done"].tolist() == [1, 1]
    # window at t: sum_j gamma^j r = gamma^L * G, the post-handoff value seen from t
    for row, L in zip(w["rew"], w["L"]):
        assert np.isclose((row * gamma ** np.arange(C)).sum(), gamma ** L * gamma ** 2)

    log = make_log(owner, end_reason="budget")  # the task never succeeds: zero credit
    w, stats = build_windows(log, 1, "task", C, handoff_gamma=gamma)
    assert stats["handoff_credit"] == [0.0] and not w["rew"].any()

    plain, _ = build_windows(make_log(owner, end_reason="success"), 1, "task", C)
    assert not plain["rew"].any()  # without credit a handoff stays a zero-return end


def test_segment_ownership_follows_goal_predicates():
    from mars.rltoken.windows import virtual_segment_owner

    log = make_log([0] * 5)
    log.goal_after = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool)  # after actions 0..4
    # action t is owned by the goals on obs t, i.e. after action t-1
    assert virtual_segment_owner(log, 1, 1).tolist() == [0, 0, 1, 1, 0]


def test_same_controller_cannot_borrow_next_objects_success():
    log = make_log([1] * 6, success_at=5, end_reason="success")
    log.target_id = np.array([0, 0, 0, 1, 1, 1])
    log.subtask_id = np.array([0, 0, 0, 1, 1, 1])
    w, stats = build_windows(log, 1, "place", C)
    assert stats["intervals"] == 2
    assert w["L"].tolist() == [3, 2, 1, 3, 2, 1]
    assert not w["rew"][:3].any()
    assert w["rew"][3:].sum() == 3

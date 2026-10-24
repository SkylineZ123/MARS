"""Chunk-level replay windows from an episode log (paper Sec. III-B b).

A skill interval is a maximal run of steps held by specialist k (its own actions or
expert corrections on its behalf), truncated at the first skill success. Every step
t of an interval starts a window of L = min(C, e - t) executed actions, so windows
overlap and never cross an ownership change.
  reward  r_{t+j} = 1 iff the skill succeeds right after action t+j (skill="task": the task)
  done    d = 1 iff the window reaches the interval end and the interval ended by
          success or by a handoff to another controller; a step-budget end, or a
          window ending while the interval continues, keeps d = 0
  handoff credit (handoff_gamma set, skill="task"): a handoff is not a zero-return end; the
          window reaching it also earns G = gamma^(t_s - e) 1{task succeeds at t_s >= e}, the
          Monte Carlo value of the post-handoff state under the downstream controllers
  ref     the generalist rows in effect ('executed' alignment, refreshed at the
          backbone's chunk cadence) or a shifted slice of one chunk; rows executed by
          the expert are replaced by the demonstrated action

Window construction follows build_transition / build_stride2_transitions in
scripts/train_rlt_stage2_pi05.py @cccf949.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .episode import SOURCE_EXPERT, EpisodeLog

PAD_SOURCE = 255


@dataclass
class SkillInterval:
    start: int
    end: int  # exclusive
    reason: str  # success | handoff | budget


def success_after(log: EpisodeLog, skill: str) -> np.ndarray:
    """Per-step reward source: the skill predicate, or task success for skill="task" (a
    specialist that owns the whole episode)."""
    return np.asarray(log.success_after if skill == "task" else log.skill_success_after[skill])


def virtual_segment_owner(log: EpisodeLog, k: int, goal_index: int) -> np.ndarray:
    """SegmentRouter ownership for generalist-only logs (warm-up): the goals on obs t decide action t;
    goals on the reset observation are taken as not yet holding."""
    goals = np.asarray(log.goal_after, dtype=bool).reshape(log.T, -1)
    before = np.vstack([np.zeros((1, goals.shape[1]), bool), goals[:-1]]) if log.T else goals
    return np.asarray([k if all(g[:goal_index]) and not g[goal_index] else 0 for g in before], dtype=np.int32)


def virtual_skill_owner(log: EpisodeLog, phase_map) -> np.ndarray:
    """Oracle ownership for generalist-only logs (warm-up): g(phase of obs t) owns action t."""
    return phase_map.phase_ids_to_classes(log.phase[: log.T]).astype(np.int8)


def skill_intervals(skill_owner: np.ndarray, skill_success: np.ndarray, k: int, budget_end: bool,
                    boundaries: np.ndarray | None = None, bootstrap_boundary: bool = False) -> list[SkillInterval]:
    T = len(skill_owner)
    out: list[SkillInterval] = []
    t = 0
    while t < T:
        if skill_owner[t] != k:
            t += 1
            continue
        s = t
        while t < T and skill_owner[t] == k and (boundaries is None or boundaries[t] == boundaries[s]):
            t += 1
        e_raw = t
        hits = np.flatnonzero(skill_success[s:e_raw])
        if hits.size:
            out.append(SkillInterval(s, s + int(hits[0]) + 1, "success"))
        elif e_raw == T and budget_end:
            out.append(SkillInterval(s, e_raw, "budget"))
        elif bootstrap_boundary and e_raw < T and skill_owner[e_raw] == k:
            out.append(SkillInterval(s, e_raw, "continuation"))
        else:
            out.append(SkillInterval(s, e_raw, "handoff"))
    return out


def _ref_block(log: EpisodeLog, t: int, C: int, alignment: str, interval: SkillInterval) -> np.ndarray:
    if alignment == "executed":
        ref = log.ref_rows(np.arange(t, t + C)).copy()
    elif alignment == "shifted_slice":
        ref = log.shifted_slice(t, C).copy()
    else:
        raise ValueError(f"unknown ref alignment {alignment!r}")
    for j in range(C):
        s = t + j
        if interval.start <= s < interval.end and s < log.T and log.source[s] == SOURCE_EXPERT:
            ref[j] = log.exec_norm[s]
    return ref


def build_windows(
    log: EpisodeLog,
    k: int,
    skill: str,
    C: int,
    alignment: str = "executed",
    pad_mode: str = "ref",
    skill_owner: np.ndarray | None = None,
    handoff_gamma: float | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    if C <= 0 or k <= 0:
        raise ValueError("C and specialist controller index k must be positive")
    if alignment not in ("executed", "shifted_slice") or pad_mode not in ("ref", "zero"):
        raise ValueError("invalid alignment or pad_mode")
    log.finalize()
    owner = log.skill_owner if skill_owner is None else skill_owner
    budget_end = log.meta.get("end_reason", "budget") == "budget"
    success = success_after(log, skill)
    boundaries = interval_boundaries(log, skill)
    intervals = skill_intervals(owner, success, k, budget_end, boundaries, bootstrap_boundary=skill == "task")
    if handoff_gamma is not None and skill != "task":
        raise ValueError("handoff credit needs the task-success reward (skill='task')")
    stats = {"intervals": len(intervals), "dropped": 0,
             "reasons": {r: sum(i.reason == r for i in intervals) for r in ("success", "handoff", "budget", "continuation")}}
    credit = {}
    if handoff_gamma is not None:
        hits = np.flatnonzero(success)
        for itv in intervals:
            if itv.reason == "handoff":
                later = hits[hits >= itv.end]
                credit[itv.start] = float(handoff_gamma ** (later[0] - itv.end)) if later.size else 0.0
        stats["handoff_credit"] = list(credit.values())
    rows: dict[str, list] = {key: [] for key in
                             ("z", "q", "ref", "act", "rew", "valid", "done", "next_z", "next_q",
                              "next_ref", "source", "step", "L")}
    for itv in intervals:
        for t in range(itv.start, itv.end):
            L = min(C, itv.end - t)
            nxt = t + L
            done = float(nxt == itv.end and itv.reason in ("success", "handoff"))
            if not log.z_valid[t]:
                stats["dropped"] += 1
                continue
            if nxt >= len(log.z_valid) or not log.z_valid[nxt]:
                if not done:
                    stats["dropped"] += 1
                    continue
                next_z = np.zeros_like(log.z[t])
                next_q = log.q_norm[min(nxt, len(log.q_norm) - 1)]
            else:
                next_z, next_q = log.z[nxt], log.q_norm[nxt]

            ref = _ref_block(log, t, C, alignment, itv)
            act = np.zeros((C, log.exec_norm.shape[1]), dtype=np.float32)
            act[:L] = log.exec_norm[t: t + L]
            if L < C and pad_mode == "ref":
                act[L:] = ref[L:]
            rew = np.zeros(C, dtype=np.float32)
            rew[:L] = (log.rewards[t: t + L] if len(log.rewards) == log.T else success[t: t + L])
            if done and itv.start in credit:
                rew[L - 1] += handoff_gamma * credit[itv.start]  # weight gamma^(L-1): adds gamma^L G
            valid = np.zeros(C, dtype=np.float32)
            valid[:L] = 1.0
            source = np.full(C, PAD_SOURCE, dtype=np.uint8)
            source[:L] = log.source[t: t + L]

            rows["z"].append(log.z[t])
            rows["q"].append(log.q_norm[t])
            rows["ref"].append(ref)
            rows["act"].append(act)
            rows["rew"].append(rew)
            rows["valid"].append(valid)
            rows["done"].append(done)
            rows["next_z"].append(next_z)
            rows["next_q"].append(next_q)
            rows["next_ref"].append(_ref_block(log, nxt, C, alignment, itv))
            rows["source"].append(source)
            rows["step"].append(t)
            rows["L"].append(L)

    if not rows["z"]:
        return {}, stats
    out = {key: np.asarray(v) for key, v in rows.items()}
    out["z"] = out["z"].astype(np.float16)
    out["next_z"] = out["next_z"].astype(np.float16)
    for key in ("q", "ref", "act", "rew", "valid", "done", "next_q", "next_ref"):
        out[key] = out[key].astype(np.float32)
    out["step"] = out["step"].astype(np.int32)
    out["L"] = out["L"].astype(np.int16)
    stats["windows"] = len(out["z"])
    stats["success_windows"] = int((out["rew"].sum(axis=1) > 0).sum())
    return out, stats


def interval_boundaries(log, skill):
    # Whole-task windows stop at instructions but bootstrap across them. Local
    # critics additionally bind targets and terminate credit at their boundary.
    if skill == "task":
        return np.asarray(log.subtask_id) if len(log.subtask_id) == log.T else None
    if log.meta.get("suite") == "assembly_v1" and len(log.interval_id) == log.T:
        return np.asarray(log.interval_id)
    if len(log.target_id) != log.T:
        return None
    subtasks = log.subtask_id if len(log.subtask_id) == log.T else np.zeros(log.T, dtype=int)
    keys = np.stack([log.target_id, subtasks], axis=1)
    return np.cumsum(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]) if log.T else np.zeros(0, int)

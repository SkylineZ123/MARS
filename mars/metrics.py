"""Offline routing metrics: per-step accuracy / F1, controller intervals (IoU, false entries,
missed intervals, entry/exit lag) and, optionally, the same metrics of the deployed owner after
skill switching."""

from __future__ import annotations

import numpy as np

from .switch import replay_switch


def intervals_from_flags(flags: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool)
    out, t, T = [], 0, len(flags)
    while t < T:
        if flags[t]:
            s = t
            while t < T and flags[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def episode_metrics(pred: np.ndarray, label: np.ndarray, num_classes: int) -> dict:
    out = {}
    for k in range(1, num_classes):
        p, y = pred == k, label == k
        union = (p | y).sum()
        out[f"iou_{k}"] = float((p & y).sum() / union) if union else 1.0
        p_iv, y_iv = intervals_from_flags(p), intervals_from_flags(y)
        out[f"false_entries_{k}"] = sum(1 for s, e in p_iv if not y[s:e].any())
        out[f"missed_{k}"] = sum(1 for s, e in y_iv if not p[s:e].any())
        if p_iv and y_iv:
            out[f"entry_lag_{k}"] = p_iv[0][0] - y_iv[0][0]
            out[f"exit_lag_{k}"] = p_iv[-1][1] - y_iv[-1][1]
    out["switches"] = int((np.diff(pred) != 0).sum()) if len(pred) > 1 else 0
    return out


def selector_report(preds: list[np.ndarray], labels: list[np.ndarray], num_classes: int,
                    switch_grid: list[tuple[float, int]] | None = None,
                    probs: list[np.ndarray] | None = None) -> dict:
    """Per-step and interval metrics of argmax predictions; with `switch_grid`, also of the skill
    switcher replayed on the per-step probabilities `probs` for every (tau, k)."""
    allp = np.concatenate(preds)
    ally = np.concatenate(labels)
    report: dict = {"accuracy": float((allp == ally).mean()), "n_steps": int(len(ally))}
    recalls = []
    for k in range(num_classes):
        tp = int(((allp == k) & (ally == k)).sum())
        fp = int(((allp == k) & (ally != k)).sum())
        fn = int(((allp != k) & (ally == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        report[f"precision_{k}"] = prec
        report[f"recall_{k}"] = rec
        report[f"f1_{k}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if tp + fn:
            recalls.append(rec)
    report["balanced_accuracy"] = float(np.mean(recalls)) if recalls else 0.0
    present = [k for k in range(num_classes) if (ally == k).any()]
    report["macro_f1"] = float(np.mean([report[f"f1_{k}"] for k in present])) if present else 0.0
    expert = [k for k in present if k > 0]
    report["expert_f1"] = float(np.mean([report[f"f1_{k}"] for k in expert])) if expert else 0.0

    per_ep = [episode_metrics(p, y, num_classes) for p, y in zip(preds, labels, strict=True)]
    keys = sorted({k for m in per_ep for k in m})
    for key in keys:
        vals = np.asarray([m[key] for m in per_ep if key in m], dtype=np.float64)
        if vals.size:
            report[f"{key}_mean"] = float(vals.mean())
            if "lag" in key:
                report[f"{key}_median"] = float(np.median(vals))
    if switch_grid and probs is None:
        raise ValueError("switch_grid needs the per-step probabilities")
    for tau, k_of_k in switch_grid or []:
        owners = [replay_switch(p, tau, k_of_k) for p in probs]
        tag = switch_tag(tau, k_of_k)
        # Report the actual deployed owner, including F1 and entry/exit delay;
        # a smaller switch count alone can hide a controller that never switches.
        deployed = selector_report(owners, labels, num_classes)
        report.update({f"{tag}_{key}": value for key, value in deployed.items()})
    return report


def switch_tag(tau: float, k: int) -> str:
    return f"switch_t{tau:g}_k{k}"


def labeled_segments(sequence):
    """Half-open intervals; VLA (class 0) is a real controller, not background."""
    sequence = np.asarray(sequence, dtype=np.int64)
    if not len(sequence):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1]
    ends = np.r_[starts[1:], len(sequence)]
    return [(int(sequence[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def normalized_edit(pred, label):
    a = [c for c, _, _ in labeled_segments(pred)]
    b = [c for c, _, _ in labeled_segments(label)]
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return 100.0 * (1 - previous[-1] / max(len(a), len(b))) if a or b else 100.0


def segment_counts(pred, label, overlap):
    truth = labeled_segments(label)
    hits = set()
    tp = fp = 0
    for c, s, e in labeled_segments(pred):
        scores = [max(0, min(e, ye) - max(s, ys)) / (max(e, ye) - min(s, ys))
                  if c == yc else 0.0 for yc, ys, ye in truth]
        j = int(np.argmax(scores)) if scores else -1
        if j >= 0 and scores[j] >= overlap and j not in hits:
            tp += 1
            hits.add(j)
        else:
            fp += 1
    return np.array([tp, fp, len(truth) - len(hits)], dtype=np.int64)


def segmentation_report(preds, labels, num_classes=3):
    """Table 5: frame metrics pooled; Edit/switches per episode; segment F1 pooled."""
    if not preds or len(preds) != len(labels):
        raise ValueError("provide matching nonempty episode lists")
    for p, y in zip(preds, labels, strict=True):
        if len(p) != len(y) or not len(y):
            raise ValueError("episode predictions and labels must have equal nonzero lengths")
        if min(np.min(p), np.min(y)) < 0 or max(np.max(p), np.max(y)) >= num_classes:
            raise ValueError("controller label out of range")
    p, y = np.concatenate(preds), np.concatenate(labels)
    f1 = []
    for c in range(num_classes):
        tp = np.sum((p == c) & (y == c))
        denom = np.sum(p == c) + np.sum(y == c)
        f1.append(2 * tp / denom if denom else 0.)
    out = {"Acc": 100. * float(np.mean(p == y)), "Macro-F1": 100. * float(np.mean(f1)),
           "Edit": float(np.mean([normalized_edit(p, y) for p, y in zip(preds, labels)])),
           "switches": float(np.mean([np.count_nonzero(np.diff(p)) for p in preds])),
           "n_episodes": len(preds), "n_steps": len(y)}
    for pct in (10, 25, 50):
        tp, fp, fn = np.sum([segment_counts(p, y, pct / 100) for p, y in zip(preds, labels)], axis=0)
        out[f"F1@{pct}"] = float(200 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.
    return out

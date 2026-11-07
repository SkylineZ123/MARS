"""Pooled candidate F1 and confusion counts, excluding IGNORE labels."""

import numpy as np

IGNORE = -100


def selector_report(preds: list[np.ndarray], labels: list[np.ndarray], num_classes: int) -> dict:
    """Macro-F1 averages every candidate; absent candidates have F1 zero."""
    if num_classes < 1 or len(preds) != len(labels):
        raise ValueError("provide matching episode lists and a positive num_classes")
    counts = np.zeros((num_classes, 3), dtype=np.int64)
    n_steps = 0
    for pred, label in zip(preds, labels, strict=True):
        pred, label = np.asarray(pred), np.asarray(label)
        if pred.ndim != 1 or pred.shape != label.shape:
            raise ValueError("episode predictions and labels must be matching vectors")
        mask = label != IGNORE
        p, y = pred[mask], label[mask]
        if not (np.isin(p, np.arange(num_classes)).all() and np.isin(y, np.arange(num_classes)).all()):
            raise ValueError("controller label out of range")
        n_steps += len(y)
        for k in range(num_classes):
            counts[k] += [np.sum((p == k) & (y == k)), np.sum((p == k) & (y != k)),
                          np.sum((p != k) & (y == k))]
    report = {"n_steps": n_steps}
    for k, (tp, fp, fn) in enumerate(counts):
        report.update({f"tp_{k}": int(tp), f"fp_{k}": int(fp), f"fn_{k}": int(fn),
                       f"f1_{k}": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0})
    report["macro_f1"] = float(np.mean([report[f"f1_{k}"] for k in range(num_classes)]))
    return report

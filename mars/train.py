"""Train a router on per-episode sequences.

Sequences are (z, y) or (z, y, q) tuples: packed observation z (T, d_z), controller labels y (T,)
with 0 = generalist, and proprioception q (T, d_q). Cross-entropy is weighted inversely to the
training-label frequency so the short specialist phases are not swamped by the generalist class.
Labels equal to IGNORE are skipped.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import selector_report

IGNORE = -100


def class_weights(labels: list[np.ndarray], num_classes: int) -> torch.Tensor:
    labels = [y[y != IGNORE] for y in labels]
    counts = np.bincount(np.concatenate(labels), minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def _pad_batch(seqs, device):
    T = max(len(s[1]) for s in seqs)
    d = seqs[0][0].shape[1]
    z = torch.zeros(len(seqs), T, d, device=device)
    y = torch.full((len(seqs), T), IGNORE, dtype=torch.long, device=device)
    has_q = len(seqs[0]) > 2 and seqs[0][2] is not None
    q = torch.zeros(len(seqs), T, seqs[0][2].shape[1], device=device) if has_q else None
    for i, s in enumerate(seqs):
        n = len(s[1])
        z[i, :n] = torch.as_tensor(s[0][:n], dtype=torch.float32, device=device)
        y[i, :n] = torch.as_tensor(s[1], dtype=torch.long, device=device)
        if has_q:
            q[i, :n] = torch.as_tensor(s[2][:n], dtype=torch.float32, device=device)
    return z, y, q


def _forward(model, z, q):
    return model.forward_seq(z, q) if getattr(model, "q_dim", 0) else model.forward_seq(z)


@torch.no_grad()
def predict(model, seqs, device, batch_episodes: int = 32, probs: bool = False) -> list[np.ndarray]:
    """Per-episode argmax predictions, or (T, K) probabilities with probs=True."""
    model.eval()
    preds = []
    for i in range(0, len(seqs), batch_episodes):
        chunk = seqs[i: i + batch_episodes]
        z, _, q = _pad_batch(chunk, device)
        logits = _forward(model, z, q)
        for j, s in enumerate(chunk):
            lj = logits[j, : len(s[1])]
            preds.append(torch.softmax(lj, -1).cpu().numpy() if probs else lj.argmax(-1).cpu().numpy())
    return preds


def _optimizer(model, cfg):
    if getattr(cfg, "optimizer", "adam") == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=getattr(cfg, "weight_decay", 0.0))


def train_router(model, train_seqs, val_seqs, cfg, device="cpu", seed: int = 0, log=print) -> dict:
    """cfg: MARSConfig (epochs, lr, batch_episodes, optimizer, weight_decay, select_best,
    selection_metric). With select_best the epoch with the best validation metric is restored."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    num_classes = model.num_classes
    weights = class_weights([s[1] for s in train_seqs], num_classes).to(device)
    opt = _optimizer(model, cfg)
    select_best = bool(getattr(cfg, "select_best", False)) and bool(val_seqs)
    best = {"score": (-1.0, -float("inf")), "epoch": 0, "state": None}
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_seqs))
        losses = []
        for i in range(0, len(order), cfg.batch_episodes):
            batch = [train_seqs[j] for j in order[i: i + cfg.batch_episodes]]
            z, y, q = _pad_batch(batch, device)
            logits = _forward(model, z, q)
            loss = F.cross_entropy(logits.reshape(-1, num_classes), y.reshape(-1), weight=weights,
                                   ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        entry = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if val_seqs:
            preds = predict(model, val_seqs, device, batch_episodes=cfg.batch_episodes)
            rep = selector_report(preds, [s[1] for s in val_seqs], num_classes)
            entry.update({"val_accuracy": rep["accuracy"], "val_balanced_accuracy": rep["balanced_accuracy"],
                          "val_macro_f1": rep["macro_f1"]})
            for k in range(1, num_classes):
                entry[f"val_iou_{k}"] = rep.get(f"iou_{k}_mean", float("nan"))
            metric = getattr(cfg, "selection_metric", "macro_f1")
            false_entries = float(np.mean([rep.get(f"false_entries_{k}_mean", 0.)
                                           for k in range(1, num_classes)]))
            score = (rep[metric], -false_entries if metric == "expert_f1" else 0.)
            entry.update(val_expert_f1=rep["expert_f1"], val_false_entries=false_entries)
            if select_best and score > best["score"]:
                best = {"score": score, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict())}
        history.append(entry)
        log(entry)
    if select_best and best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "class_weights": weights.cpu().tolist(),
            "best_epoch": best["epoch"] if select_best else cfg.epochs}

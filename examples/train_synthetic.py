"""End-to-end MARS example on synthetic episodes (CPU, about a minute).

Each episode has three controllers: 0 = generalist, 1 = Pickup, 2 = Place. A short Pickup phase
is followed later by a Place phase. The "phase started" cue in the image features is the same for
both skills, so telling them apart needs memory of whether a pickup already happened.

Real deployments use the shared RL Token and proprioception, and
encode the candidate descriptions with SigLIP2 (`python -m mars.text`); here random unit vectors
stand in for the text embeddings so the example runs without open_clip.

    python examples/train_synthetic.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from mars import MARSConfig, MARSController, build_router, count_params, predict, save_router, train_router
from mars.metrics import selector_report

CLASSES = ["vla", "pickup", "place"]
Z_DIM = 32  # shared RL Token
Q_DIM = 4


def episodes(n: int, seed: int, T: int = 60):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(0, 0.1, size=(T, Z_DIM)).astype(np.float32)
        q = rng.normal(0, 0.1, size=(T, Q_DIM)).astype(np.float32)
        y = np.zeros(T, dtype=np.int64)
        a = int(rng.integers(5, 20))
        b = int(rng.integers(a + 15, T - 10))
        for start, skill in ((a, 1), (b, 2)):
            z[start: start + 6, 0] = 1.0  # RL Token cue: "a critical phase is on"
            y[start: start + 6] = skill
        out.append((z, y, q))
    return out


def main():
    cfg = MARSConfig(z_dim=Z_DIM, q_dim=Q_DIM, d_model=32, heads=4, ssm_layers=1, d_state=8, window=4,
                     epochs=25, lr=3e-3, dropout=0.0)
    torch.manual_seed(0)
    model = build_router(cfg.type, Z_DIM, len(CLASSES), cfg, text_dim=16)
    text = torch.nn.functional.normalize(torch.randn(len(CLASSES), 16), dim=-1)
    model.set_text_embeddings(text)
    print(f"{model.kind}: {count_params(model):,} trainable parameters")

    train, val = episodes(256, seed=0), episodes(64, seed=1)
    result = train_router(model, train, val, cfg, log=lambda e: None)
    print(f"best epoch {result['best_epoch']}")

    probs = predict(model, val, "cpu", probs=True)
    rep = selector_report([p.argmax(-1) for p in probs], [s[1] for s in val], len(CLASSES),
                          switch_grid=[(cfg.switch_tau, cfg.switch_k)], probs=probs)
    tag = f"switch_t{cfg.switch_tau:g}_k{cfg.switch_k}"
    print(f"per-step macro-F1 {rep['macro_f1']:.3f}, deployed macro-F1 {rep[tag + '_macro_f1']:.3f}, "
          f"switches/ep {rep['switches_mean']:.2f} -> {rep[tag + '_switches_mean']:.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mars.pt"
        save_router(path, model, CLASSES)
        ctrl = MARSController.from_checkpoint(path, cfg)
    z, y, q = val[0]
    ctrl.reset()
    for t in range(len(y)):
        active, switched, _ = ctrl.query(z[t], q[t])
        if switched:
            print(f"t={t:2d}: switch to {ctrl.active_name:6s} (label {CLASSES[y[t]]})")


if __name__ == "__main__":
    main()

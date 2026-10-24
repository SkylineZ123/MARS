import numpy as np
import pytest
import torch

from mars import MARSConfig, MARSController, build_router, predict, save_router, train_router
from mars.baselines import GRURouter
from mars.metrics import intervals_from_flags, selector_report
from mars.router import RecurrentRouter
from mars.switch import replay_switch
from mars.train import class_weights

TRAIN_CFG = MARSConfig(epochs=40, lr=3e-3, batch_episodes=16, optimizer="adamw", weight_decay=1e-4)


def _cue_sequences(n, T=30, d=4, start=10, seed=0):
    """The class for t >= start is decided by a cue shown only in the first 3 steps (needs memory)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        cue = int(rng.integers(0, 2))
        z = rng.normal(0, 0.1, size=(T, d)).astype(np.float32)
        z[:3, 0] = 3.0 if cue else -3.0
        z[start:, 1] = 1.0  # "a critical phase started", but which specialist depends on the cue
        y = np.zeros(T, dtype=np.int64)
        y[start:] = 1 + cue
        seqs.append((z, y))
    return seqs


def _late_accuracy(model, val):
    z = torch.tensor(np.stack([s[0] for s in val]))
    y = np.stack([s[1] for s in val])
    pred = model.eval().forward_seq(z).argmax(-1).numpy()
    return (pred[:, 10:] == y[:, 10:]).mean()


def test_mamba_memory_reaches_an_early_cue():
    torch.manual_seed(0)
    model = RecurrentRouter(4, 3, trunk="mamba", q_dim=0, d_model=16, layers=1, d_state=8, dropout=0.0)
    train_router(model, _cue_sequences(128), _cue_sequences(32, seed=1), TRAIN_CFG, log=lambda e: None)
    assert _late_accuracy(model, _cue_sequences(32, seed=1)) > 0.95


def test_gru_router_reaches_an_early_cue():
    torch.manual_seed(0)
    model = GRURouter(4, 3, proj_dim=16, hidden=16, dropout=0.0)
    cfg = MARSConfig(epochs=40, lr=1e-2, batch_episodes=16, optimizer="adam", weight_decay=0.0)
    train_router(model, _cue_sequences(128), _cue_sequences(32, seed=1), cfg, log=lambda e: None)
    assert _late_accuracy(model, _cue_sequences(32, seed=1)) > 0.95


def test_class_weights_inverse_frequency():
    w = class_weights([np.array([0, 0, 0, 1])], 3)
    assert w.tolist() == pytest.approx([4 / 9, 4 / 3, 0.0])


def test_selector_report_intervals():
    assert intervals_from_flags(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    pred = [np.array([0, 0, 1, 1, 1, 0])]
    label = [np.array([0, 1, 1, 1, 0, 0])]
    probs = [np.eye(2)[pred[0]]]
    rep = selector_report(pred, label, 2, switch_grid=[(0.5, 1), (0.5, 2)], probs=probs)
    assert rep["iou_1_mean"] == pytest.approx(0.5)
    assert rep["entry_lag_1_mean"] == 1 and rep["exit_lag_1_mean"] == 1
    assert rep["switch_t0.5_k1_iou_1_mean"] == pytest.approx(0.5)  # k = 1 on one-hot scores: argmax
    assert rep["switch_t0.5_k2_iou_1_mean"] == pytest.approx(0.2)  # owners [0,0,0,1,1,1]: one step late both ways
    with pytest.raises(ValueError):
        selector_report(pred, label, 2, switch_grid=[(0.5, 1)])


def test_controller_streams_like_offline_replay(tmp_path):
    torch.manual_seed(1)
    cfg = MARSConfig(modality_dims=[4, 4, 4, 4], q_dim=3, d_model=16, ssm_layers=1, window=4, dropout=0.0,
                     switch_tau=0.4, switch_k=2)
    model = build_router("mars", 16, 3, cfg, text_dim=8)
    model.set_text_embeddings(torch.randn(3, 8))
    path = tmp_path / "mars.pt"
    save_router(path, model, ["vla", "pickup", "place"])
    rng = np.random.default_rng(0)
    z = rng.normal(size=(25, 16)).astype(np.float32)
    q = rng.normal(size=(25, 3)).astype(np.float32)
    probs = predict(model, [(z, np.zeros(25, dtype=np.int64), q)], "cpu", probs=True)[0]
    owners = replay_switch(probs, cfg.switch_tau, cfg.switch_k)

    ctrl = MARSController.from_checkpoint(path, cfg)
    assert (ctrl.switcher.tau, ctrl.switcher.k) == (0.4, 2)
    for _ in range(2):  # reset gives a fresh episode
        ctrl.reset()
        assert ctrl.active_name == "vla"
        for t in range(25):
            active, switched, p = ctrl.query(z[t], q[t])
            np.testing.assert_allclose(p, probs[t], atol=1e-5)
            assert active == owners[t]
            assert switched == (t > 0 and owners[t] != owners[t - 1] or t == 0 and owners[0] != 0)
    with pytest.raises(ValueError, match="proprioception"):
        ctrl.query(z[0])

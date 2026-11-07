import numpy as np
import pytest
import torch

from mars import MARSConfig, MARSController, build_router, predict, save_router, train_router
from mars.metrics import selector_report
from mars.router import MARSRouter
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
    model = MARSRouter(4, 3, skill_query=False, trunk="mamba", q_dim=0, d_model=16, layers=1, d_state=8, dropout=0.0)
    train_router(model, _cue_sequences(128), _cue_sequences(32, seed=1), TRAIN_CFG, log=lambda e: None)
    assert _late_accuracy(model, _cue_sequences(32, seed=1)) > 0.95


def test_class_weights_inverse_frequency():
    w = class_weights([np.array([0, 0, 0, 1])], 3)
    assert w.tolist() == pytest.approx([4 / 9, 4 / 3, 0.0])


def test_selector_report_masks_ignore_and_includes_absent_candidates():
    rep = selector_report([np.array([0, 1, 2, 999])], [np.array([0, 1, 0, -100])], 4)
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 1) / 4)
    assert rep["n_steps"] == 3
    assert rep["tp_0"] == 1 and rep["fn_0"] == 1 and rep["fp_2"] == 1
    assert set(rep) == {"macro_f1", "n_steps"} | {f"{key}_{k}" for key in ("f1", "tp", "fp", "fn") for k in range(4)}
    assert selector_report([np.array([5])], [np.array([-100])], 3)["macro_f1"] == 0
    with pytest.raises(ValueError):
        selector_report([np.array([3])], [np.array([0])], 3)


def test_controller_streams_like_offline_replay(tmp_path):
    torch.manual_seed(1)
    cfg = MARSConfig(q_dim=3, d_model=16, ssm_layers=1, window=4, dropout=0.0,
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


def test_training_selects_masked_all_candidate_macro_f1(monkeypatch):
    import mars.train as training
    cfg = MARSConfig(type="mlp", use_q=False, d_model=8, dropout=0, epochs=2, batch_episodes=1)
    model = build_router("mlp", 4, 3, cfg)
    seqs = [(np.ones((3, 4), dtype=np.float32), np.array([0, 1, -100]))]
    predictions = iter([[np.array([0, 1, 99])], [np.array([0, 0, 1])]])
    monkeypatch.setattr(training, "predict", lambda *args, **kwargs: next(predictions))
    snapshots = []
    def record(entry):
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
    result = train_router(model, seqs, seqs, cfg, log=record)
    assert result["best_epoch"] == 1
    assert result["history"][0]["val_macro_f1"] == pytest.approx(2 / 3)
    assert result["history"][0]["val_n_steps"] == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshots[0][key])


def test_training_all_ignored_batch_is_finite():
    cfg = MARSConfig(type="mlp", use_q=False, d_model=8, epochs=1)
    model = build_router("mlp", 4, 3, cfg)
    seqs = [(np.ones((3, 4), dtype=np.float32), np.full(3, -100))]
    result = train_router(model, seqs, seqs, cfg, log=lambda _: None)
    assert result["history"][0]["train_loss"] == 0
    assert result["history"][0]["val_macro_f1"] == 0

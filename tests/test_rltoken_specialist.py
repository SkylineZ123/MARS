from dataclasses import replace

import numpy as np
import pytest
import torch

from mars.rltoken.config import SpecialistConfig
from mars.rltoken import losses
from mars.rltoken.learner import SpecialistLearner, frozen
from mars.rltoken.networks import ChunkActor, TwinCritic
from mars.rltoken.replay import PHASE_ONLINE, PHASE_WARMUP, ReplayBuffer
from mars.rltoken.schedules import alpha, lambda_q, lambda_ref

Z, Q, C, A = 6, 8, 3, 2


def test_schedule_endpoints():
    assert alpha(0) == alpha(20000) == 0.0
    assert alpha(45000) == pytest.approx(0.5)
    assert alpha(70000) == alpha(10**6) == 1.0
    assert (lambda_ref(0), lambda_q(0)) == (7.0, pytest.approx(0.05))
    assert (lambda_ref(70000), lambda_q(70000)) == (2.5, pytest.approx(0.45))


def test_actor_has_two_hidden_layers():
    actor = ChunkActor(Z, Q, C, A, [256, 256])
    linears = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in linears] == [256, 256, C * A]


def test_discounted_chunk_return():
    rew = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    ret, boot = losses.discounted_chunk_return(rew, valid, 0.9)
    assert ret.squeeze().tolist() == pytest.approx([0.81, 1.0])
    assert boot.squeeze().tolist() == pytest.approx([0.729, 0.9])


def _batch(B=4):
    g = torch.Generator().manual_seed(0)
    return {
        "z": torch.randn(B, Z, generator=g), "q": torch.randn(B, Q, generator=g),
        "ref": torch.randn(B, C, A, generator=g), "act": torch.randn(B, C, A, generator=g),
        "rew": torch.zeros(B, C), "valid": torch.ones(B, C), "done": torch.zeros(B),
        "next_z": torch.randn(B, Z, generator=g), "next_q": torch.randn(B, Q, generator=g),
        "next_ref": torch.randn(B, C, A, generator=g),
    }


class ConstQ(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, z, q, a):
        v = torch.full((z.shape[0], 1), self.value) + 0 * self.w
        return v, v + 100.0  # Q2 differs: the actor must use Q1

    def q1_only(self, z, q, a):
        return self.forward(z, q, a)[0] + 0 * a.sum()


def test_critic_target_masks_done_and_bootstraps_gamma_L():
    cfg = replace(SpecialistConfig(), C=C, target_sigma=0.0, gamma=0.5, td_clamp_low=-100, td_clamp_high=100)
    batch = _batch(2)
    batch["rew"] = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    batch["valid"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    batch["done"] = torch.tensor([1.0, 0.0])
    actor = ChunkActor(Z, Q, C, A, [8])
    y = losses.critic_target(batch, actor, ConstQ(2.0), cfg)
    assert y.squeeze().tolist() == pytest.approx([0.5, 0.5**3 * 2.0])


def test_actor_loss_uses_mean_and_q1_with_normalized_proximity():
    cfg = replace(SpecialistConfig(), C=C, ref_dropout=0.0)
    batch = _batch()
    actor = ChunkActor(Z, Q, C, A, [8])
    with torch.no_grad():
        mu = actor(batch["z"], batch["q"], batch["ref"])
    loss, info = losses.actor_loss(batch, actor, ConstQ(3.0), n=0, cfg=cfg)
    expected = -0.05 * 3.0 + 7.0 * ((mu - batch["ref"]) ** 2).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert info["actor_q1"] == pytest.approx(3.0)


def test_reference_dropout_only_touches_actor_input():
    cfg = replace(SpecialistConfig(), C=C, ref_dropout=1.0)
    batch = _batch()
    seen = {}

    class Spy(ChunkActor):
        def forward(self, z, q, ref):
            seen["ref"] = ref.clone()
            return super().forward(z, q, ref)

    actor = Spy(Z, Q, C, A, [8])
    loss, info = losses.actor_loss(batch, actor, ConstQ(0.0), n=0, cfg=cfg)
    assert torch.all(seen["ref"] == 0)
    with torch.no_grad():
        mu = actor(batch["z"], batch["q"], torch.zeros_like(batch["ref"]))
    assert info["ref_mse"] == pytest.approx(((mu - batch["ref"]) ** 2).mean().item(), rel=1e-5)


def test_actor_step_leaves_critic_untouched():
    cfg = replace(SpecialistConfig(), C=C, actor_hidden=[16], critic_hidden=[16], policy_delay=1)
    learner = SpecialistLearner(Z, Q, A, cfg)
    with frozen(learner.critic):
        loss, _ = losses.actor_loss(_batch(), learner.actor, learner.critic, 0, cfg)
        loss.backward()
    assert all(p.grad is None for p in learner.critic.parameters())
    assert all(p.requires_grad for p in learner.critic.parameters())


def test_learner_update_and_act():
    cfg = replace(SpecialistConfig(), C=C, actor_hidden=[16], critic_hidden=[16], sigma=0.1)
    learner = SpecialistLearner(Z, Q, A, cfg)
    for _ in range(3):
        info = learner.update(_batch(8))
    assert learner.n_updates == 3 and np.isfinite(info["critic_loss"])
    mu = learner.act(np.zeros(Z), np.zeros(Q), np.zeros((C, A)))
    noisy = learner.act(np.zeros(Z), np.zeros(Q), np.zeros((C, A)), explore=True, rng=np.random.default_rng(0))
    assert mu.shape == (C, A) and not np.allclose(mu, noisy)
    state = learner.state_dict()
    other = SpecialistLearner(Z, Q, A, cfg)
    other.load_state_dict(state)
    assert other.n_updates == 3


def test_replay_ring_and_persistence(tmp_path):
    buf = ReplayBuffer(capacity=5, seed=0)
    def windows(n, base):
        return {"z": np.full((n, Z), base, np.float16), "q": np.zeros((n, Q), np.float32),
                "ref": np.zeros((n, C, A), np.float32), "act": np.zeros((n, C, A), np.float32),
                "rew": np.zeros((n, C), np.float32), "valid": np.ones((n, C), np.float32),
                "done": np.zeros(n, np.float32), "next_z": np.zeros((n, Z), np.float16),
                "next_q": np.zeros((n, Q), np.float32), "next_ref": np.zeros((n, C, A), np.float32),
                "source": np.zeros((n, C), np.uint8), "step": np.arange(n, dtype=np.int32),
                "L": np.full(n, C, np.int16)}
    buf.add_batch(windows(3, 1), PHASE_WARMUP, -1)
    buf.add_batch(windows(4, 2), PHASE_ONLINE, 0)
    assert len(buf) == 5
    assert buf.stats()["online"] == 4
    batch = buf.sample(7)
    assert batch["z"].dtype == torch.float32 and batch["z"].shape == (7, Z)
    buf.save(tmp_path / "r.npz")
    again = ReplayBuffer.load(tmp_path / "r.npz")
    assert len(again) == 5

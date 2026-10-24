from dataclasses import replace

import numpy as np
import pytest
import torch

from mars import MARSConfig, MARSController, build_router, pack_modalities
from mars.rltoken import (
    AdapterSpec, AdapterTrainConfig, IdentitySpecialist, PHASE_ONLINE, PrefixFeatures, RLTAdapter,
    ReplayBuffer, RoutedRLTPolicy, SpecialistConfig, SpecialistLearner, load_adapter,
    load_specialist, load_episode, save_episode, EpisodeLog, train_adapter, train_specialist,
)
from mars.rltoken.adapter import TokenStandardizer
from mars.rltoken.cli import main
from mars.rltoken.features import token_groups


def make_adapter():
    return RLTAdapter(AdapterSpec(4, 8, 6, enc_layers=1, dec_layers=1, heads=2, dropout=0),
                      ["image", "image2", "lang"], [2, 2, 2])


def features():
    return PrefixFeatures(torch.randn(1, 6, 4),
                          {"image": slice(0, 2), "image2": slice(2, 4), "lang": slice(4, 6)},
                          {g: torch.ones(1, 2, dtype=torch.bool) for g in ("image", "image2", "lang")})


def windows(n=6):
    rng = np.random.default_rng(9)
    def normal(*shape):
        return rng.normal(size=shape).astype(np.float32)
    return {"z": normal(n, 8), "next_z": normal(n, 8), "q": normal(n, 3), "next_q": normal(n, 3),
            "act": normal(n, 2, 2), "ref": normal(n, 2, 2), "next_ref": normal(n, 2, 2),
            "rew": np.zeros((n, 2), np.float32), "valid": np.ones((n, 2), np.float32),
            "done": np.zeros(n, np.float32)}


def config():
    return SpecialistConfig(C=2, actor_hidden=[8], critic_hidden=[8], batch_size=3, buffer_capacity=10)


def test_adapter_training_freezes_backbone_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    model = make_adapter()
    tokens = torch.randn(6, 6, 4, requires_grad=True)
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[:, -1] = False
    path = train_adapter(model, tokens, mask, AdapterTrainConfig(batch_size=3, epochs=2, warmup_steps=0),
                         tmp_path, val=(tokens[:2], mask[:2]), log=lambda _: None)
    assert tokens.grad is None
    restored = load_adapter(path)
    assert restored.decoder is None and all(not p.requires_grad for p in restored.parameters())
    full = load_adapter(path, drop_decoder=False)
    torch.testing.assert_close(restored.encode(tokens, mask), full.encode(tokens, mask))
    changed = tokens.detach().clone()
    changed[:, -1] = float("nan")
    torch.testing.assert_close(restored.encode(tokens, mask), restored.encode(changed, mask))
    with pytest.raises(ValueError, match="token count"):
        restored.encode(torch.zeros(1, 7, 4), torch.ones(1, 7, dtype=torch.bool))


def test_single_valid_token_stats_remain_finite():
    norm = TokenStandardizer(4)
    norm.fit(torch.ones(1, 4), torch.ones(1, dtype=torch.bool))
    assert torch.isfinite(norm.std).all() and (norm.std > 0).all()
    with pytest.raises(ValueError, match="finite valid"):
        norm.fit(torch.ones(1, 4), torch.zeros(1, dtype=torch.bool))


def test_group_padding_keeps_modality_slots():
    f = features()
    f.slices["lang"] = slice(4, 5)
    f.group_masks["lang"] = torch.ones(1, 1, dtype=torch.bool)
    tokens, mask = token_groups(f, ["image", "image2", "lang"], [2, 2, 2])
    assert tokens.shape == (1, 6, 4) and not mask[0, -1]
    model = make_adapter().eval()
    torch.testing.assert_close(model.encode_features(f), model.encode(tokens, mask))


def test_switch_immediately_discards_both_generalist_and_expert_queues():
    adapter = make_adapter().eval()
    cfg = MARSConfig(modality_dims=[8, 4, 4, 4], q_dim=3, d_model=8, heads=2, ssm_layers=1,
                     dropout=0, skill_query=False)
    router = build_router("mars", 20, 2, cfg)
    ctrl = MARSController(router, ["vla", "pickup"], tau=0.6, k=1)
    ref_calls = []
    class Expert(IdentitySpecialist):
        def act(self, z, q, ref, **kwargs):
            return np.full((self.C, 2), 9, np.float32)
    def reference(f, q):
        ref_calls.append(1)
        return np.full((5, 2), len(ref_calls), np.float32)
    policy = RoutedRLTPolicy(ctrl, adapter, reference, {"pickup": Expert(3)})
    # Drive actual MARS output logits while retaining the real streaming/switch path.
    final = router.head
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([10., -10.]))
    f, q = features(), np.zeros(3, np.float32)
    a = policy.step(f, q)
    assert a.active_name == "vla" and np.all(a.action == 1) and len(policy.queue) == 4
    with torch.no_grad():
        final.bias.copy_(torch.tensor([-10., 10.]))
    b = policy.step(f, q)
    assert b.switched and b.active_name == "pickup" and np.all(b.action == 9)
    assert len(ref_calls) == 2 and len(policy.queue) == 2
    with torch.no_grad():
        final.bias.copy_(torch.tensor([10., -10.]))
    c = policy.step(f, q)
    assert c.switched and c.active_name == "vla" and np.all(c.action == 3)
    policy.reset()
    assert ctrl.active == 0 and not policy.queue and policy.t == 0
    assert pack_modalities(f, adapter.encode_features(f)).shape == (1, 20)


def test_resume_matches_uninterrupted_learner_and_replay(tmp_path):
    torch.manual_seed(2)
    learner = SpecialistLearner(8, 3, 2, config())
    replay = ReplayBuffer(10, seed=5)
    replay.add_batch(windows(), PHASE_ONLINE, 1)
    checkpoint = tmp_path / "skill.pt"
    train_specialist(learner, replay, 2, checkpoint, log=lambda _: None)
    train_specialist(learner, replay, 3, log=lambda _: None)
    expected = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    again = load_specialist(checkpoint, resume=True)
    again_replay = ReplayBuffer.load(checkpoint.with_suffix(".replay.npz"))
    train_specialist(again, again_replay, 3, log=lambda _: None)
    assert again.n_updates == learner.n_updates == 5
    for k, v in expected.items():
        torch.testing.assert_close(v, again.actor.state_dict()[k], rtol=0, atol=0)
    inference = load_specialist(checkpoint)
    assert not inference.actor.training
    inference.act(np.zeros(8), np.zeros(3), np.zeros((2, 2)))
    assert not inference.actor.training
    with pytest.raises(ValueError, match="resume config"):
        load_specialist(checkpoint, replace(config(), gamma=0.5), resume=True)


def test_legacy_routerlt_checkpoint_compatibility(tmp_path):
    cfg = config()
    learner = SpecialistLearner(8, 3, 2, cfg)
    path = tmp_path / "old.pt"
    torch.save({"learner": learner.state_dict()}, path)
    loaded = load_specialist(path, cfg)
    for k, v in learner.actor.state_dict().items():
        torch.testing.assert_close(v, loaded.actor.state_dict()[k])
    with pytest.raises(ValueError, match="pass cfg"):
        load_specialist(path)


def test_replay_oversized_batch_shrink_and_sampling_rng(tmp_path):
    replay = ReplayBuffer(4, seed=1)
    w = windows(10)
    w["z"] = np.broadcast_to(np.arange(10)[:, None], (10, 8)).copy().astype(np.float32)
    replay.add_batch(w, PHASE_ONLINE, 0)
    assert set(replay._storage["z"][:, 0]) == {6, 7, 8, 9}
    replay.save(tmp_path / "replay.npz")
    again = ReplayBuffer.load(tmp_path / "replay.npz")
    torch.testing.assert_close(replay.sample(5)["z"], again.sample(5)["z"])
    small = ReplayBuffer.load(tmp_path / "replay.npz", capacity=2)
    assert set(small._storage["z"][:, 0]) == {8, 9}


def test_invalid_windows_rejected_before_optimizer_update():
    learner = SpecialistLearner(8, 3, 2, config())
    w = {k: torch.from_numpy(v) for k, v in windows().items()}
    w["valid"][0] = torch.tensor([0., 1.])
    with pytest.raises(ValueError, match="contiguous prefix"):
        learner.update(w)
    assert learner.n_updates == 0


def test_cli_adapter_specialist_resume(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("adapter:\n  d_model: 8\n  heads: 2\n  enc_layers: 1\n  dec_layers: 1\n"
                   "adapter_train:\n  batch_size: 3\nspecialist:\n  C: 2\n  actor_hidden: [8]\n"
                   "  critic_hidden: [8]\n  batch_size: 3\n  buffer_capacity: 10\n")
    data = tmp_path / "tokens.npz"
    np.savez(data, tokens=np.ones((6, 6, 4), np.float32), mask=np.ones((6, 6), bool),
             token_groups=np.array(["image", "image2", "lang"]), group_sizes=np.array([2, 2, 2]))
    main(["train-adapter", "--data", str(data), "--config", str(cfg), "--output", str(tmp_path / "adapter")])
    assert load_adapter(tmp_path / "adapter/adapter.pt").z_dim == 8
    data = tmp_path / "windows.npz"
    np.savez(data, **windows())
    checkpoint = tmp_path / "skill.pt"
    main(["train-specialist", "--data", str(data), "--config", str(cfg), "--output", str(checkpoint),
          "--updates", "2"])
    main(["train-specialist", "--resume", str(checkpoint), "--config", str(cfg), "--output", str(checkpoint),
          "--updates", "1"])
    assert load_specialist(checkpoint, resume=True).n_updates == 3


def test_episode_roundtrip_and_cli_window_builder(tmp_path):
    log = EpisodeLog(meta={"end_reason": "budget"})
    log.exec_norm = [np.zeros(2, np.float32)] * 4
    log.skill_owner, log.source = [1] * 4, [0] * 4
    log.success_after = [False] * 4
    log.skill_success_after = {"custom_skill": [False, False, False, True]}
    log.z, log.z_valid = [np.ones(8, np.float32)] * 5, [True] * 5
    log.q_norm = [np.zeros(3, np.float32)] * 5
    log.vla_t, log.vla_chunks = [0], [np.ones((4, 2), np.float32)]
    path = tmp_path / "episode.pt"
    save_episode(path, log)
    again = load_episode(path)
    np.testing.assert_equal(again.z, log.z)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("specialist:\n  C: 2\n")
    out = tmp_path / "windows.npz"
    main(["build-windows", "--episode", str(path), "--skill", "custom_skill", "--controller-id", "1",
          "--config", str(cfg), "--output", str(out)])
    with np.load(out) as data:
        assert len(data["z"]) == 4
        assert data["done"].tolist() == [0, 0, 1, 1]

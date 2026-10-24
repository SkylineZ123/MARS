from pathlib import Path

import pytest
import torch

from mars import MARSConfig, build_router, count_params, load_router, needs_text, save_router
from mars.baselines import GRURouter, WindowedQueryRouter
from mars.layers import window_indices, window_pad
from mars.router import RecurrentRouter
from mars.text import load_text_cache, text_spec

Z, Q, K = 6, 8, 3
CLASSES = ["vla", "pickup", "place"]


def make_model(kind="mars", mode="text", **overrides):
    cfg = MARSConfig(use_q=True, q_dim=3, d_model=16, heads=4, ssm_layers=1, temporal_layers=1, window=4,
                     window_stride=1, dropout=0., modality_dims=[4, 4, 4, 4], query_mode=mode, **overrides)
    model = build_router(kind, 16, 3, cfg, text_dim=12 if mode != "id" else 0)
    if needs_text(kind, cfg):
        model.set_text_embeddings(torch.randn(3, 12))
    return model, cfg


@pytest.mark.parametrize("kind", ["mars", "mars_gru", "mamba_query", "gru_query", "windowed_query"])
def test_text_queries_causal_step_and_roundtrip(kind, tmp_path):
    torch.manual_seed(3)
    model, cfg = make_model(kind)
    model.eval()
    assert model.kind == kind and needs_text(kind, cfg)
    assert not any("id_emb" in n for n, _ in model.named_parameters())
    assert "text_emb" in dict(model.named_buffers())
    assert not model.text_emb.requires_grad
    z, q = torch.randn(1, 9, 16), torch.randn(1, 9, 3)
    with torch.no_grad():
        logits = model.forward_seq(z, q)
        changed = z.clone()
        changed[:, 5:] += 10
        torch.testing.assert_close(logits[:, :5], model.forward_seq(changed, q)[:, :5])
        state = model.init_state()
        for t in range(9):
            p, state = model.step(z[0, t], state, q[0, t])
            torch.testing.assert_close(p, logits.softmax(-1)[0, t], atol=1e-6, rtol=1e-5)
        path = tmp_path / "router.pt"
        save_router(path, model, CLASSES)
        restored, names = load_router(path)
        assert names == CLASSES and restored.kind == kind
        torch.testing.assert_close(restored.forward_seq(z, q), logits)
    model.train().forward_seq(z, q).square().mean().backward()
    assert model.text_proj[1].weight.grad is not None
    assert model.text_emb.grad is None


@pytest.mark.parametrize("overrides", [dict(fusion="concat"), dict(skill_query=False), dict(query_mode="id"),
                                       dict(query_mode="text_id")])
def test_mars_variants_step_matches_sequence(overrides):
    torch.manual_seed(0)
    mode = overrides.pop("query_mode", "text")
    model, _ = make_model("mars", mode, **overrides)
    model.eval()
    z, q = torch.randn(1, 11, 16), torch.randn(1, 11, 3)
    probs = model.forward_seq(z, q).softmax(-1)[0]
    state = model.init_state()
    for t in range(11):
        p, state = model.step(z[0, t], state, q[0, t])
        torch.testing.assert_close(p, probs[t], atol=1e-6, rtol=1e-5)


def test_candidate_permutation_and_identical_descriptions():
    model, _ = make_model()
    model.eval()
    z, q = torch.randn(1, 7, 16), torch.randn(1, 7, 3)
    with torch.no_grad():
        original = model.forward_seq(z, q)
        perm = torch.tensor([2, 0, 1])
        model.set_text_embeddings(model.text_emb[perm].clone())
        torch.testing.assert_close(model.forward_seq(z, q), original[..., perm], atol=1e-6, rtol=1e-5)
        model.set_text_embeddings(model.text_emb[:1].expand(3, -1).clone())
        equal = model.forward_seq(z, q)
        torch.testing.assert_close(equal[..., 0], equal[..., 1])


def test_legacy_fusion_checkpoint_defaults(tmp_path):
    model, _ = make_model(mode="id")
    model.ctor.pop("query_mode")
    model.ctor.pop("text_dim")
    model.kind = "fusion_mamba"
    path = tmp_path / "old.pt"
    save_router(path, model, CLASSES)
    restored, _ = load_router(path)
    assert restored.kind == "mars" and restored.query_mode == "id"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])


def test_text_cache_rejects_misaligned_metadata(tmp_path):
    spec = text_spec(MARSConfig(), CLASSES[:2])
    path = tmp_path / "text.pt"
    torch.save({"spec": spec, "embeddings": torch.randn(2, 8)}, path)
    assert load_text_cache(path, spec).shape == (2, 8)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_text_cache(path, {**spec, "classes": ["pickup", "vla"]})
    with pytest.raises(ValueError, match="missing controller descriptions"):
        text_spec(MARSConfig(), ["vla", "insert"])


def test_invalid_configuration():
    with pytest.raises(ValueError, match="query_mode"):
        RecurrentRouter(16, 3, skill_query=True, query_mode="typo")
    with pytest.raises(ValueError, match="text_dim"):
        RecurrentRouter(16, 3, skill_query=True, query_mode="text")
    with pytest.raises(ValueError, match="modality_dims"):
        build_router("mars", 16, 3, MARSConfig(modality_dims=[4, 4, 4]), text_dim=8)
    with pytest.raises(ValueError, match="unknown MARS config keys"):
        MARSConfig.from_dict({"router": {"windows": 3}})


def test_shipped_config_builds_the_paper_router():
    cfg = MARSConfig.from_yaml(Path(__file__).parents[1] / "configs" / "mars.yaml")
    assert cfg.type == "mars" and cfg.query_mode == "text" and (cfg.switch_tau, cfg.switch_k) == (0.6, 2)
    model = build_router(cfg.type, sum(cfg.modality_dims), 3, cfg, text_dim=768)
    assert model.kind == "mars" and model.window == 8
    assert not any("id_emb" in n for n, _ in model.named_parameters())
    assert count_params(model) == 1_382_161  # SigLIP2 ViT-B-16 text width 768, three candidates


# ---------------------------------------------------------------- recurrent trunk without fusion
def _recurrent(trunk, skill_query, seed=0):
    torch.manual_seed(seed)
    model = RecurrentRouter(Z, K, trunk=trunk, skill_query=skill_query, q_dim=Q, d_model=16, layers=2, d_state=4,
                            window=4, stride=1, heads=2, dropout=0.0, query_mode="text_id", text_dim=10)
    model.set_text_embeddings(torch.randn(K, 10))
    return model.eval()


@pytest.mark.parametrize("trunk,sq", [("mamba", False), ("mamba", True), ("gru", True)])
def test_recurrent_step_matches_sequence_and_is_causal(trunk, sq):
    model = _recurrent(trunk, sq)
    assert model.kind == {("mamba", False): "mamba_head", ("mamba", True): "mamba_query",
                          ("gru", True): "gru_query"}[(trunk, sq)]
    z, q = torch.randn(1, 19, Z), torch.randn(1, 19, Q)
    seq = model.forward_seq(z, q)
    probs = torch.softmax(seq[0], -1)
    state = model.init_state()
    for t in range(19):
        p, state = model.step(z[0, t], state, q[0, t])
        assert torch.allclose(p, probs[t], atol=1e-5), t
    z2 = z.clone()
    z2[0, 11:] += 3.0
    out = model.forward_seq(z2, q)
    assert torch.allclose(out[0, :11], seq[0, :11], atol=1e-5)
    assert not torch.allclose(out[0, 11:], seq[0, 11:], atol=1e-3)


# ---------------------------------------------------------------- comparison routers
def _windowed(kind="windowed_query", q_dim=Q, window=4, stride=3, query_mode="text_id", seed=0):
    torch.manual_seed(seed)
    model = WindowedQueryRouter(Z, K, window=window, stride=stride, q_dim=q_dim, d_model=16, heads=2,
                                temporal_layers=2, query_layers=1, dropout=0.0, temporal=kind != "windowed_xattn",
                                skill_query=kind != "windowed_tf", query_mode=query_mode, text_dim=10)
    if model.skill_query and query_mode == "text_id":
        model.set_text_embeddings(torch.randn(K, 10))
    return model.eval()


@pytest.mark.parametrize("kind", ["windowed_tf", "windowed_xattn", "windowed_query"])
def test_windowed_router_is_causal_and_streams(kind):
    model = _windowed(kind)
    assert model.kind == kind
    z, q = torch.randn(1, 23, Z), torch.randn(1, 23, Q)
    base = model.forward_seq(z, q)
    z2 = z.clone()
    z2[0, 12:] += 3.0 * torch.randn(11, Z)
    out = model.forward_seq(z2, q)
    assert torch.allclose(out[0, :12], base[0, :12], atol=1e-5)
    assert not torch.allclose(out[0, 12:], base[0, 12:], atol=1e-3)
    seq = torch.softmax(base[0], -1)
    hist = model.init_state()
    for t in range(23):
        p, hist = model.step(z[0, t], hist, q[0, t])
        assert torch.allclose(p, seq[t], atol=1e-5), t


def test_windowed_padding_content_is_ignored():
    model = _windowed(window=4, stride=3)
    tok = torch.randn(1, 4, 2, 16)
    pad = torch.tensor([[False, False, True, True]])
    tok2 = tok.clone()
    tok2[0, 2:] = 10.0 * torch.randn(2, 2, 16)
    assert torch.allclose(model.forward_windows(tok, pad), model.forward_windows(tok2, pad), atol=1e-5)
    assert window_pad(5, 3, 2)[1].tolist() == [False, True, True]
    assert window_indices(5, 3, 2)[4].tolist() == [4, 2, 0]


def test_queries_attend_over_several_history_tokens():
    model = _windowed(window=4, stride=3)
    _, attn = model.forward_seq(torch.randn(1, 15, Z), torch.randn(1, 15, Q), return_attn=True)
    assert attn.shape == (1, 15, K, 8)  # K candidate queries x (L=4 steps x 2 modalities)
    late = attn[0, -1]
    assert torch.allclose(late.sum(-1), torch.ones(K), atol=1e-5)
    assert (late > 1e-4).sum(-1).min() > 1
    assert torch.all(attn[0, 0, :, 2:] < 1e-6)  # t=0: only the current step is real


def test_gru_router_step_matches_sequence():
    torch.manual_seed(0)
    model = GRURouter(Z, K, 8, 8, 0.0, q_dim=Q).eval()
    z, q = torch.randn(1, 10, Z), torch.randn(1, 10, Q)
    seq = torch.softmax(model.forward_seq(z, q)[0], -1)
    state = model.init_state()
    for t in range(10):
        p, state = model.step(z[0, t], state, q[0, t])
        assert torch.allclose(p, seq[t], atol=1e-5)

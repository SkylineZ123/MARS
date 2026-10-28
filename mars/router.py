"""Paper MARS: RL Token + proprioception, causal memory and text skill queries.

MARSRouter implements the paper's separately projected, normalized and summed
state embedding. MultimodalFusionRouter retains the earlier four-input variant
for explicitly identified legacy checkpoints only.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from .layers import MambaBlock, QueryBlock, window_indices, window_pad


def check_text_queries(skill_query: bool, text_dim: int) -> None:
    if skill_query and text_dim <= 0:
        raise ValueError("candidate text queries require text_dim > 0 (the text encoder width)")


def init_text_queries(module: nn.Module, num_classes: int, text_dim: int, d: int) -> None:
    """Frozen candidate text embeddings (buffer) and the shared trainable query projection."""
    module.register_buffer("text_emb", torch.zeros(num_classes, text_dim))
    module.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))


def store_text_embeddings(module: nn.Module, emb: Tensor) -> None:
    if not module.skill_query:
        return
    if tuple(emb.shape) != tuple(module.text_emb.shape) or not torch.isfinite(emb).all():
        raise ValueError(f"expected finite text embeddings of shape {tuple(module.text_emb.shape)}, "
                         f"got {tuple(emb.shape)}")
    with torch.no_grad():
        module.text_emb.copy_(emb.detach().to(module.text_emb))


def pack_modalities(features, z: Tensor) -> Tensor:
    """Pack one router input vector per sample: [z, mean(image), mean(image2), mean(lang)].

    `features` is the frozen VLA prefix of the current observation, duck-typed:
      features.hidden       (B, N, d) prefix hidden states
      features.slices       {"image": slice, "image2": slice, "lang": slice} token ranges in N
      features.group_masks  {name: (B, n) bool} validity of each token in that range
    `z` (B, d_z) is the shared RL token. Each token group is mean-pooled over its valid tokens.
    """
    groups = []
    for key in ("image", "image2", "lang"):
        tokens = features.hidden[:, features.slices[key]].float()
        mask = features.group_masks[key].float()[..., None]
        groups.append((tokens * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat([z.float(), *groups], dim=-1)


class RecurrentRouter(nn.Module):
    """Per-step recurrent trunk (Mamba or GRU) with a linear head or a candidate-skill query head."""

    def __init__(self, z_dim: int, num_classes: int, trunk: str = "mamba", skill_query: bool = False,
                 q_dim: int = 8, d_model: int = 128, layers: int = 2, d_state: int = 16, window: int = 8,
                 stride: int = 1, heads: int = 4, query_layers: int = 1, ff_mult: int = 4, dropout: float = 0.1,
                 text_dim: int = 0):
        super().__init__()
        if trunk not in ("mamba", "gru"):
            raise ValueError(f"unknown trunk {trunk!r}")
        check_text_queries(skill_query, text_dim)
        self.ctor = dict(z_dim=z_dim, num_classes=num_classes, trunk=trunk, skill_query=skill_query, q_dim=q_dim,
                         d_model=d_model, layers=layers, d_state=d_state, window=window, stride=stride, heads=heads,
                         query_layers=query_layers, ff_mult=ff_mult, dropout=dropout, text_dim=text_dim)
        self.z_dim, self.q_dim, self.num_classes = z_dim, q_dim, num_classes
        self.trunk, self.skill_query = trunk, skill_query
        self.window, self.stride = window, stride
        self.kind = {("mamba", False): "mamba_head", ("mamba", True): "mamba_query", ("gru", True): "gru_query",
                     ("gru", False): "gru_trunk"}[(trunk, skill_query)]
        d = d_model
        self.z_in = nn.Sequential(nn.Linear(z_dim, d), nn.LayerNorm(d))
        self.q_in = nn.Sequential(nn.Linear(q_dim, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)) if q_dim else None
        self.drop = nn.Dropout(dropout)
        if trunk == "mamba":
            self.blocks = nn.ModuleList([MambaBlock(d, d_state) for _ in range(layers)])
        else:
            self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        if skill_query:
            self.age_emb = nn.Parameter(torch.randn(window, d) * 0.02)
            init_text_queries(self, num_classes, text_dim, d)
            self.query_blocks = nn.ModuleList([QueryBlock(d, heads, ff_mult, dropout) for _ in range(query_layers)])
            self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.head = nn.Linear(d, num_classes)

    # ------------------------------------------------------------------ pieces
    def set_text_embeddings(self, emb: Tensor) -> None:
        """Store the frozen candidate text embeddings (num_classes, text_dim), in class order."""
        store_text_embeddings(self, emb)

    def queries(self) -> Tensor:
        """(K, d) candidate queries, a function of the candidate texts only."""
        return self.text_proj(self.text_emb)

    def _inputs(self, z: Tensor, q: Tensor | None) -> Tensor:
        x = self.z_in(z)
        if self.q_dim:
            if q is None:
                raise ValueError("this router was built with proprio input; pass q")
            x = x + self.q_in(q)
        return self.drop(x)

    def _head(self, states: Tensor, pad: Tensor | None = None, return_attn: bool = False):
        """states (M, L, d) newest first (or (M, d) without queries) -> logits (M, K)."""
        if not self.skill_query:
            return self.head(states), None
        kv = states + self.age_emb
        q = self.queries()[None].expand(states.shape[0], -1, -1)
        attn = None
        for block in self.query_blocks:
            q, attn = block(q, kv, pad, need_weights=return_attn)
        return self.score(q).squeeze(-1), attn

    # ------------------------------------------------------------------ router interface
    def forward_seq(self, z: Tensor, q: Tensor | None = None, return_attn: bool = False):
        """z (B, T, z_dim), q (B, T, q_dim) -> logits (B, T, K); step t never reads steps after t."""
        x = self._inputs(z, q)
        if self.trunk == "mamba":
            for block in self.blocks:
                x = block(x)
        else:
            x, _ = self.gru(x)
        h = self.out_norm(x)  # (B, T, d)
        B, T, d = h.shape
        if not self.skill_query:
            logits = self.head(h)
            return (logits, None) if return_attn else logits
        idx = window_indices(T, self.window, self.stride, h.device)
        pad = window_pad(T, self.window, self.stride, h.device)
        win = h[:, idx].reshape(B * T, self.window, d)
        logits, attn = self._head(win, pad[None].expand(B, -1, -1).reshape(B * T, -1), return_attn)
        logits = logits.reshape(B, T, -1)
        if return_attn:
            return logits, None if attn is None else attn.reshape(B, T, *attn.shape[1:])
        return logits

    def init_state(self, device=None) -> dict:
        """Fresh per-episode memory."""
        state = {"hist": deque(maxlen=(self.window - 1) * self.stride + 1)}
        if self.trunk == "mamba":
            state["caches"] = [b.init_cache(device) for b in self.blocks]
        else:
            state["h"] = torch.zeros(self.gru.num_layers, 1, self.gru.hidden_size, device=device)
        return state

    @torch.no_grad()
    def step(self, z_t: Tensor, state: dict, q_t: Tensor | None = None) -> tuple[Tensor, dict]:
        """One streaming query: (z_t, q_t) -> candidate probabilities (K,), updated memory."""
        x = self._inputs(z_t.reshape(-1), None if q_t is None else q_t.reshape(-1))
        if self.trunk == "mamba":
            for block, cache in zip(self.blocks, state["caches"], strict=True):
                x = block.step(x, cache)
        else:
            out, state["h"] = self.gru(x.reshape(1, 1, -1), state["h"])
            x = out[0, 0]
        h = self.out_norm(x)
        if not self.skill_query:
            return torch.softmax(self.head(h), dim=-1), state
        state["hist"].append(h)
        hist = list(state["hist"])
        n = len(hist)
        picks = [hist[max(n - 1 - k * self.stride, 0)] for k in range(self.window)]
        pad = torch.tensor([[n - 1 - k * self.stride < 0 for k in range(self.window)]], device=h.device)
        logits, _ = self._head(torch.stack(picks)[None], pad)
        return torch.softmax(logits[0], dim=-1), state


class MultimodalFusionRouter(RecurrentRouter):
    """Explicit modality fusion followed by the causal recurrent trunk and the skill-query head.

    The packed input z is [z_rl, camera_1, camera_2, language] (see `pack_modalities`, sizes in
    `modality_dims`); proprioception q is separate. No phase, progress, reward or simulator
    state enters this module.
    """

    def __init__(self, z_dim: int, num_classes: int, modality_dims, q_dim: int = 8, d_model: int = 128,
                 layers: int = 2, d_state: int = 16, window: int = 8, heads: int = 4, dropout: float = 0.1,
                 trunk: str = "mamba", fusion: str = "attention", skill_query: bool = True,
                 text_dim: int = 0):
        dims = tuple(modality_dims)
        if len(dims) != 4 or sum(dims) != z_dim or min(dims) <= 0:
            raise ValueError("modality_dims must describe z, image, image2, language")
        if fusion not in ("attention", "concat"):
            raise ValueError(f"unknown fusion {fusion}")
        super().__init__(z_dim, num_classes, trunk=trunk, skill_query=skill_query,
                         q_dim=q_dim, d_model=d_model, layers=layers, d_state=d_state,
                         window=window, stride=1, heads=heads, dropout=dropout, text_dim=text_dim)
        self.ctor = dict(z_dim=z_dim, num_classes=num_classes, modality_dims=list(dims), q_dim=q_dim,
                         d_model=d_model, layers=layers, d_state=d_state, window=window,
                         heads=heads, dropout=dropout, trunk=trunk, fusion=fusion, skill_query=skill_query,
                         text_dim=text_dim)
        self.kind = "mars" if trunk == "mamba" else "mars_gru"
        self.needs_modalities = True
        self.modality_dims, self.fusion = dims, fusion
        del self.z_in
        del self.q_in
        self.modal_projs = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_model))
                                          for d in (*dims, *((q_dim,) if q_dim else ()))])
        n = len(self.modal_projs)
        self.modal_ids = nn.Parameter(torch.randn(n, d_model) * .02)
        if fusion == "attention":
            self.fuse = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout=dropout,
                                                   batch_first=True, norm_first=True)
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * .02)
            self.pool = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        else:
            self.fuse = nn.Sequential(nn.Linear(n * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def _inputs(self, z, q):
        groups = list(z.split(self.modality_dims, dim=-1))
        if self.q_dim:
            if q is None:
                raise ValueError("proprioception is required")
            groups.append(q)
        tokens = torch.stack([p(x) for p, x in zip(self.modal_projs, groups, strict=True)], dim=-2)
        shape = tokens.shape[:-2]
        tokens = tokens.reshape(-1, *tokens.shape[-2:]) + self.modal_ids
        if self.fusion == "attention":
            tokens = self.fuse(tokens)
            result, _ = self.pool(self.pool_query.expand(len(tokens), -1, -1), tokens, tokens, need_weights=False)
            result = result[:, 0]
        else:
            result = self.fuse(tokens.flatten(1))
        return self.drop(result.reshape(*shape, result.shape[-1]))


class MARSRouter(RecurrentRouter):
    """Paper MARS: RL Token and proprioception, projected separately and summed."""

    needs_modalities = False

    def __init__(self, z_dim, num_classes, q_dim=8, d_model=128, layers=2,
                 d_state=16, window=8, heads=4, dropout=0.1, trunk="mamba",
                 skill_query=True, text_dim=0, query_layers=2, ff_mult=4):
        super().__init__(z_dim, num_classes, trunk=trunk, skill_query=skill_query,
                         q_dim=q_dim, d_model=d_model, layers=layers, d_state=d_state,
                         window=window, heads=heads, dropout=dropout, text_dim=text_dim,
                         query_layers=query_layers, ff_mult=ff_mult)
        self.kind = "mars" if trunk == "mamba" else "mars_gru"
        self.ctor.pop("stride")


class SingleFrameRouter(RecurrentRouter):
    """Same embedding and text head, with an identity trunk and one current token."""

    needs_modalities = False

    def __init__(self, z_dim, num_classes, q_dim=8, d_model=128, heads=4,
                 dropout=0.1, text_dim=0, query_layers=2, ff_mult=4):
        super().__init__(z_dim, num_classes, q_dim=q_dim, d_model=d_model,
                         layers=0, window=1, heads=heads, dropout=dropout,
                         text_dim=text_dim, query_layers=query_layers, ff_mult=ff_mult,
                         skill_query=True)
        self.kind = "single_frame"
        self.ctor = dict(z_dim=z_dim, num_classes=num_classes, q_dim=q_dim,
                         d_model=d_model, heads=heads, dropout=dropout, text_dim=text_dim,
                         query_layers=query_layers, ff_mult=ff_mult)

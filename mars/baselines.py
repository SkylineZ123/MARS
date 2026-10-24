"""Comparison routers with the same per-step interface as MARS.

GRURouter            h_t = GRU(P([z_t, q_t]), h_{t-1}), p_t = softmax(H(h_t)); hidden reset per episode
WindowedQueryRouter  a causal window of L samples t - k*stride (k = 0..L-1) instead of a dense memory:
                       windowed_tf     tokens -> time-causal Transformer -> current-step pool -> linear head
                       windowed_xattn  tokens -> candidate queries cross-attend -> shared score MLP
                       windowed_query  tokens -> time-causal Transformer -> candidate queries -> score MLP

Window samples before the episode start are clamped and flagged as padding. Every sample gives a
compact-state token (z) and, optionally, a proprioception token (q). Candidate queries come from the
frozen candidate texts, as in MARS.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from .layers import QueryBlock, SelfBlock, window_indices, window_pad
from .router import check_text_queries, init_text_queries, store_text_embeddings


def _join(z: Tensor, q: Tensor | None, q_dim: int) -> Tensor:
    if not q_dim:
        return z
    if q is None:
        raise ValueError("this router was built with proprio input; pass q")
    return torch.cat([z, q], dim=-1)


class GRURouter(nn.Module):
    kind = "gru"

    def __init__(self, z_dim: int, num_classes: int, proj_dim: int = 256, hidden: int = 256, dropout: float = 0.1,
                 q_dim: int = 0):
        super().__init__()
        self.ctor = dict(z_dim=z_dim, num_classes=num_classes, proj_dim=proj_dim, hidden=hidden, dropout=dropout,
                         q_dim=q_dim)
        self.z_dim, self.q_dim, self.num_classes = z_dim, q_dim, num_classes
        self.proj = nn.Sequential(nn.Linear(z_dim + q_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                                  nn.Dropout(dropout))
        self.gru = nn.GRU(proj_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward_seq(self, z: Tensor, q: Tensor | None = None) -> Tensor:
        """z (B, T, z_dim) -> logits (B, T, K); each sequence starts from a zero hidden state."""
        out, _ = self.gru(self.proj(_join(z, q, self.q_dim)))
        return self.head(out)

    def init_state(self, device=None) -> Tensor:
        return torch.zeros(1, 1, self.gru.hidden_size, device=device)

    @torch.no_grad()
    def step(self, z_t: Tensor, h: Tensor, q_t: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = _join(z_t.reshape(-1), None if q_t is None else q_t.reshape(-1), self.q_dim)
        out, h = self.gru(self.proj(x.reshape(1, 1, -1)), h)
        return torch.softmax(self.head(out[0, -1]), dim=-1), h


class WindowedQueryRouter(nn.Module):
    """Causal-window router with an optional temporal encoder and optional candidate queries.

    kind follows the switches: temporal only -> "windowed_tf", queries only -> "windowed_xattn",
    both -> "windowed_query".
    """

    def __init__(self, z_dim: int, num_classes: int, window: int = 6, stride: int = 10, q_dim: int = 8,
                 d_model: int = 128, heads: int = 4, temporal_layers: int = 2, query_layers: int = 1,
                 ff_mult: int = 4, dropout: float = 0.1, temporal: bool = True, skill_query: bool = True,
                 text_dim: int = 0):
        super().__init__()
        if not (temporal or skill_query):
            raise ValueError("WindowedQueryRouter needs a temporal encoder, candidate queries, or both")
        check_text_queries(skill_query, text_dim)
        self.ctor = dict(z_dim=z_dim, num_classes=num_classes, window=window, stride=stride, q_dim=q_dim,
                         d_model=d_model, heads=heads, temporal_layers=temporal_layers, query_layers=query_layers,
                         ff_mult=ff_mult, dropout=dropout, temporal=temporal, skill_query=skill_query,
                         text_dim=text_dim)
        self.z_dim, self.q_dim, self.num_classes = z_dim, q_dim, num_classes
        self.window, self.stride = window, stride
        self.temporal, self.skill_query = temporal, skill_query
        self.kind = "windowed_query" if temporal and skill_query else ("windowed_tf" if temporal else "windowed_xattn")
        self.n_mod = 2 if q_dim else 1
        d = d_model

        self.z_in = nn.Sequential(nn.Linear(z_dim, d), nn.LayerNorm(d))
        self.q_in = nn.Sequential(nn.Linear(q_dim, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)) if q_dim else None
        self.mod_emb = nn.Parameter(torch.zeros(self.n_mod, d))
        self.age_emb = nn.Parameter(torch.randn(window, d) * 0.02)
        self.pad_emb = nn.Parameter(torch.zeros(d))
        self.drop = nn.Dropout(dropout)
        # Token order: age-major ([z_0, q_0, z_1, q_1, ...]); age 0 is the current step.
        ages = torch.arange(window).repeat_interleave(self.n_mod)
        self.register_buffer("token_age", ages, persistent=False)
        # Time-causal: a token may read tokens of its own step or older ones.
        self.register_buffer("causal_block", ages[None, :] < ages[:, None], persistent=False)

        self.encoder = nn.ModuleList([SelfBlock(d, heads, ff_mult, dropout) for _ in range(temporal_layers)]) \
            if temporal else None
        self.enc_norm = nn.LayerNorm(d)
        if skill_query:
            init_text_queries(self, num_classes, text_dim, d)
            self.query_blocks = nn.ModuleList([QueryBlock(d, heads, ff_mult, dropout) for _ in range(query_layers)])
            self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    # ------------------------------------------------------------------ pieces
    def set_text_embeddings(self, emb: Tensor) -> None:
        store_text_embeddings(self, emb)

    def queries(self) -> Tensor:
        return self.text_proj(self.text_emb)

    def _step_tokens(self, z: Tensor, q: Tensor | None) -> Tensor:
        """Per-step tokens (..., n_mod, d) before windowing (projection does not depend on the window)."""
        toks = [self.z_in(z)]
        if self.q_dim:
            if q is None:
                raise ValueError("this router was built with proprio tokens; pass q")
            toks.append(self.q_in(q))
        return torch.stack(toks, dim=-2) + self.mod_emb

    def forward_windows(self, tok: Tensor, pad: Tensor, return_attn: bool = False):
        """tok (M, L, n_mod, d) per-step tokens of each window (age-major), pad (M, L) -> logits (M, K)."""
        M, L = pad.shape
        d = tok.shape[-1]
        pad_tok = pad.repeat_interleave(self.n_mod, dim=1)  # (M, N)
        x = tok.reshape(M, L * self.n_mod, d)
        x = torch.where(pad_tok[..., None], self.pad_emb.expand_as(x), x)
        x = self.drop(x + self.age_emb.repeat_interleave(self.n_mod, dim=0))
        if self.encoder is not None:
            # Real tokens never read padding; padding (always the oldest ages) may read padding,
            # so no attention row is empty.
            blocked = self.causal_block[None] | (pad_tok[:, None, :] & ~pad_tok[:, :, None])
            blocked = blocked.repeat_interleave(self.encoder[0].attn.num_heads, dim=0)
            for block in self.encoder:
                x = block(x, blocked)
        x = self.enc_norm(x)
        if not self.skill_query:
            current = x[:, : self.n_mod].mean(dim=1)  # age-0 tokens; history enters through the encoder
            return (self.head(current), None) if return_attn else self.head(current)
        q = self.queries()[None].expand(M, -1, -1)
        attn = None
        for block in self.query_blocks:
            q, attn = block(q, x, pad_tok, need_weights=return_attn)
        logits = self.score(q).squeeze(-1)
        return (logits, attn) if return_attn else logits

    # ------------------------------------------------------------------ router interface
    def forward_seq(self, z: Tensor, q: Tensor | None = None, return_attn: bool = False):
        """z (B, T, z_dim), q (B, T, q_dim) -> logits (B, T, K); windows never look past t."""
        B, T, _ = z.shape
        idx = window_indices(T, self.window, self.stride, z.device)  # (T, L)
        pad = window_pad(T, self.window, self.stride, z.device)
        tok = self._step_tokens(z, q)[:, idx]  # (B, T, L, n_mod, d)
        out = self.forward_windows(tok.reshape(B * T, *tok.shape[2:]), pad[None].expand(B, -1, -1).reshape(B * T, -1),
                                   return_attn=return_attn)
        if return_attn:
            logits, attn = out
            return logits.reshape(B, T, -1), None if attn is None else attn.reshape(B, T, *attn.shape[1:])
        return out.reshape(B, T, -1)

    def init_state(self, device=None) -> deque:
        return deque(maxlen=(self.window - 1) * self.stride + 1)

    @torch.no_grad()
    def step(self, z_t: Tensor, history: deque, q_t: Tensor | None = None) -> tuple[Tensor, deque]:
        history.append(self._step_tokens(z_t.reshape(-1), None if q_t is None else q_t.reshape(-1)))
        hist = list(history)
        n = len(hist)
        picks = [hist[max(n - 1 - k * self.stride, 0)] for k in range(self.window)]
        pad = torch.tensor([[n - 1 - k * self.stride < 0 for k in range(self.window)]], device=picks[0].device)
        logits = self.forward_windows(torch.stack(picks)[None], pad)
        return torch.softmax(logits[0], dim=-1), history

"""RL-token adapter (paper Eq. 1, Fig. 2 "RLT Representation Adapter").

Encoder: a learned readout query e_rl appended to the selected frozen image and
language hidden states; z_t is the encoder output at the e_rl position.
Decoder (training only): z_t plus N_tok learned positional queries reconstruct the
frozen states in parallel (non-autoregressive), so z_t must retain enough of the
backbone's representation to support reconstruction.

Encoder adapted from lerobot/src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py
(RLTokenEncoder) @cccf949; the decoder replaces its teacher-forced causal decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass
class AdapterSpec:
    d_in: int
    d_model: int
    n_slots: int
    enc_layers: int = 4
    dec_layers: int = 4
    heads: int = 12
    ff_mult: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.d_in, self.d_model, self.n_slots, self.enc_layers, self.dec_layers, self.heads, self.ff_mult) <= 0:
            raise ValueError("adapter dimensions and layer counts must be positive")
        if self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("d_model must be divisible by heads and dropout must be in [0, 1)")


def _encoder(d: int, layers: int, heads: int, ff_mult: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d,
        nhead=heads,
        dim_feedforward=d * ff_mult,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)


class TokenStandardizer(nn.Module):
    """Per-channel standardization of frozen hidden states (stats stored as buffers)."""

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))

    @torch.no_grad()
    def fit(self, tokens: Tensor, mask: Tensor) -> None:
        valid = tokens[mask.bool()].float()
        if not len(valid) or not torch.isfinite(valid).all():
            raise ValueError("standardizer needs finite valid tokens")
        self.mean.copy_(valid.mean(dim=0))
        self.std.copy_(valid.std(dim=0, unbiased=False).clamp_min(1e-3))

    def forward(self, tokens: Tensor) -> Tensor:
        return (tokens - self.mean) / self.std


class ReadoutEncoder(nn.Module):
    def __init__(self, spec: AdapterSpec):
        super().__init__()
        d = spec.d_model
        self.in_proj = nn.Linear(spec.d_in, d) if spec.d_in != d else nn.Identity()
        self.slot_embed = nn.Parameter(torch.zeros(1, spec.n_slots, d))
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # e_rl
        self.transformer = _encoder(d, spec.enc_layers, spec.heads, spec.ff_mult, spec.dropout)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """tokens (B, N, d_in) standardized, mask (B, N) True=valid -> z (B, d_model)."""
        B, N, _ = tokens.shape
        x = self.in_proj(tokens) + self.slot_embed[:, :N]
        x = torch.cat([x, self.readout.expand(B, -1, -1)], dim=1)
        key_padding = torch.cat([~mask.bool(), torch.zeros(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
        return self.transformer(x, src_key_padding_mask=key_padding)[:, -1]


class ParallelQueryDecoder(nn.Module):
    def __init__(self, spec: AdapterSpec):
        super().__init__()
        d = spec.d_model
        self.z_proj = nn.Linear(d, d)
        self.queries = nn.Parameter(torch.randn(1, spec.n_slots, d) * 0.02)
        self.transformer = _encoder(d, spec.dec_layers, spec.heads, spec.ff_mult, spec.dropout)
        self.head = nn.Linear(d, spec.d_in)

    def forward(self, z: Tensor, n_tokens: int) -> Tensor:
        B = z.shape[0]
        x = torch.cat([self.z_proj(z)[:, None], self.queries[:, :n_tokens].expand(B, -1, -1)], dim=1)
        return self.head(self.transformer(x)[:, 1:])


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over valid tokens and channels (padded tokens ignored)."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


class RLTAdapter(nn.Module):
    def __init__(self, spec: AdapterSpec, token_groups: list[str], group_sizes: list[int]):
        super().__init__()
        if not token_groups or len(token_groups) != len(group_sizes) or min(group_sizes) <= 0:
            raise ValueError("provide a positive size for every token group")
        if sum(group_sizes) > spec.n_slots:
            raise ValueError(f"{sum(group_sizes)} tokens exceed {spec.n_slots} slots")
        self.spec = spec
        self.token_groups = list(token_groups)
        self.group_sizes = list(group_sizes)
        self.standardizer = TokenStandardizer(spec.d_in)
        self.encoder = ReadoutEncoder(spec)
        self.decoder = ParallelQueryDecoder(spec)

    @property
    def z_dim(self) -> int:
        return self.spec.d_model

    def encode(self, tokens: Tensor, mask: Tensor) -> Tensor:
        self._check_inputs(tokens, mask)
        tokens = tokens.detach().masked_fill(~mask.bool()[..., None], 0)
        return self.encoder(self.standardizer(tokens.float()), mask)

    def _check_inputs(self, tokens: Tensor, mask: Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != self.spec.d_in or mask.shape != tokens.shape[:2]:
            raise ValueError("expected tokens (B, N, d_in) and mask (B, N)")
        if not 0 < tokens.shape[1] <= self.spec.n_slots:
            raise ValueError("token count exceeds checkpoint slots or is empty")

    def encode_features(self, features) -> Tensor:
        from .features import token_groups

        tokens, mask = token_groups(features, self.token_groups, self.group_sizes)
        device = self.standardizer.mean.device
        return self.encode(tokens.to(device), mask.to(device))

    def reconstruction_loss(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("load with drop_decoder=False to train the adapter")
        self._check_inputs(tokens, mask)
        tokens = tokens.detach().masked_fill(~mask.bool()[..., None], 0)
        target = self.standardizer(tokens.float())
        z = self.encoder(target, mask)
        pred = self.decoder(z, tokens.shape[1])
        return masked_mse(pred, target.detach(), mask), pred

    def export(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "token_groups": self.token_groups,
            "group_sizes": self.group_sizes,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_export(cls, payload: dict, drop_decoder: bool = True) -> RLTAdapter:
        adapter = cls(AdapterSpec(**payload["spec"]), payload["token_groups"], payload["group_sizes"])
        adapter.load_state_dict(payload["state_dict"])
        if drop_decoder:
            adapter.decoder = None
        return adapter.eval()


def load_adapter(path, device="cpu", drop_decoder: bool = True) -> RLTAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapter = RLTAdapter.from_export(payload, drop_decoder=drop_decoder).to(device)
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter

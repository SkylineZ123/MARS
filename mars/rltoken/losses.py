"""Specialist objectives (paper Eq. 3-5).

Critic target (Eq. 5), one chunk-level transition of L <= C executed actions:
    y = sum_{j<L} gamma^j r_{t+j} + (1 - d) gamma^L min_i Qbar_i(s_{t+L}, a')
with a' the target actor's mean at s_{t+L} plus clipped noise (TD3 smoothing).

Actor (Eq. 3-4):
    L = -lambda_Q(n) Q1(s, mu) + lambda_ref(n)/(C d_a) ||mu - a_ref||^2 + lambda_D L_D(mu)
Reference dropout zeroes the actor's reference *input*; the proximity penalty always
uses the undropped (expert-corrected) reference.

`discounted_chunk_return` is adapted from scripts/train_rlt_stage2_pi05.py @cccf949.
The actor term differs from that script on purpose: it uses the noise-free mean
and Q1 (paper), not a sampled action and min(Q1, Q2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .schedules import lambda_q, lambda_ref


def discounted_chunk_return(rewards: Tensor, valid_mask: Tensor, gamma: float) -> tuple[Tensor, Tensor]:
    """(B, C) rewards and validity -> discounted return (B, 1) and bootstrap factor gamma^L (B, 1)."""
    valid_mask = valid_mask.to(dtype=rewards.dtype)
    steps = torch.arange(rewards.shape[-1], device=rewards.device, dtype=rewards.dtype)
    weights = torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype).pow(steps)
    returns = (rewards * valid_mask * weights).sum(dim=-1, keepdim=True)
    executed = valid_mask.sum(dim=-1, keepdim=True)
    bootstrap = torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype).pow(executed)
    return returns, bootstrap


@torch.no_grad()
def critic_target(batch: dict[str, Tensor], actor_target, critic_target_net, cfg) -> Tensor:
    mu_next = actor_target(batch["next_z"], batch["next_q"], batch["next_ref"])
    noise = (torch.randn_like(mu_next) * cfg.target_sigma).clamp(-cfg.target_clip, cfg.target_clip)
    q1, q2 = critic_target_net(batch["next_z"], batch["next_q"], mu_next + noise)
    q_next = torch.min(q1, q2)
    returns, bootstrap = discounted_chunk_return(batch["rew"], batch["valid"], cfg.gamma)
    done = batch["done"].reshape(-1, 1)
    y = returns + (1.0 - done) * bootstrap * q_next
    return y.clamp(cfg.td_clamp_low, cfg.td_clamp_high)


def critic_loss(batch: dict[str, Tensor], critic, y: Tensor) -> tuple[Tensor, dict]:
    q1, q2 = critic(batch["z"], batch["q"], batch["act"])
    loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    return loss, {
        "critic_loss": loss.item(),
        "q1_mean": q1.mean().item(),
        "td_target_mean": y.mean().item(),
    }


def actor_loss(batch: dict[str, Tensor], actor, critic, n: int, cfg) -> tuple[Tensor, dict]:
    ref = batch["ref"]
    B = ref.shape[0]
    keep = (torch.rand(B, 1, 1, device=ref.device) >= cfg.ref_dropout).to(ref.dtype)
    mu = actor(batch["z"], batch["q"], ref * keep)
    q1 = critic.q1_only(batch["z"], batch["q"], mu)
    lam_q = lambda_q(n, cfg.alpha_start, cfg.alpha_span)
    lam_ref = lambda_ref(n, cfg.alpha_start, cfg.alpha_span)
    proximity = ((mu - ref) ** 2).mean(dim=(1, 2))  # ||mu - a_ref||^2 / (C d_a)
    loss = -lam_q * q1.mean() + lam_ref * proximity.mean()
    smooth = torch.zeros((), device=ref.device)
    if cfg.lambda_delta > 0 and mu.shape[1] > 1:
        smooth = ((mu[:, 1:] - mu[:, :-1]) ** 2).mean(dim=(1, 2)).mean()  # Eq. 4
        loss = loss + cfg.lambda_delta * smooth
    return loss, {
        "actor_loss": loss.item(),
        "actor_q1": q1.mean().item(),
        "ref_mse": proximity.mean().item(),
        "smooth": float(smooth),
        "lambda_q": lam_q,
        "lambda_ref": lam_ref,
    }

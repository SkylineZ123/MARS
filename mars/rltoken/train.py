"""Train the shared adapter on cached frozen tokens and specialists on replay windows.

Adapter optimization follows routerlt/adapter/train.py; VLA inference is deliberately
outside the learner, so these routines require only the base MARS dependencies.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import AdapterTrainConfig
from .learner import save_specialist


def train_adapter(adapter, tokens, mask, cfg: AdapterTrainConfig, out_dir, device="cpu", val=None, log=print):
    """Fit normalization, reconstruct masked tokens, save adapter.pt (best validation epoch).

    tokens: (N, slots, d_in), mask: (N, slots); val is an optional (tokens, mask).
    Frozen token tensors are detached, including when cached tensors require gradients.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tokens = torch.as_tensor(tokens).detach().cpu()
    mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
    adapter._check_inputs(tokens, mask)
    if len(tokens) == 0 or not mask.any(dim=1).all():
        raise ValueError("training samples must contain at least one valid token")
    if adapter.decoder is None:
        raise ValueError("adapter training requires a decoder")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adapter.pt"
    adapter.to(device).requires_grad_(True)
    # Merge per-batch moments in float64 without materializing all selected VLA tokens.
    count = 0
    mean = torch.zeros(adapter.spec.d_in, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    picks = rng.choice(len(tokens), min(len(tokens), cfg.stats_frames), replace=False)
    for start in range(0, len(picks), cfg.batch_size):
        idx = picks[start:start + cfg.batch_size]
        valid = tokens[idx][mask[idx]].double()
        if not torch.isfinite(valid).all():
            raise ValueError("normalization data contains non-finite valid tokens")
        n = len(valid)
        batch_mean = valid.mean(0)
        delta = batch_mean - mean
        m2 += ((valid - batch_mean) ** 2).sum(0) + delta.square() * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    with torch.no_grad():
        adapter.standardizer.mean.copy_(mean)
        adapter.standardizer.std.copy_((m2 / count).sqrt().clamp_min(1e-3))
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * math.ceil(len(tokens) / cfg.batch_size)

    def schedule(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    best, step, history = float("inf"), 0, []
    for epoch in range(cfg.epochs):
        adapter.train()
        losses = []
        order = rng.permutation(len(tokens))
        for start in range(0, len(tokens), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            loss, _ = adapter.reconstruction_loss(tokens[idx].to(device), mask[idx].to(device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite adapter reconstruction loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": float(np.mean(losses))}
        if val is not None:
            adapter.eval()
            vt = torch.as_tensor(val[0]).detach().cpu()
            vm = torch.as_tensor(val[1], dtype=torch.bool).cpu()
            adapter._check_inputs(vt, vm)
            if len(vt) == 0 or not vm.any(1).all():
                raise ValueError("validation data must contain valid tokens")
            error, elements = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(vt), cfg.batch_size):
                    mb = vm[start:start + cfg.batch_size].to(device)
                    loss, _ = adapter.reconstruction_loss(vt[start:start + cfg.batch_size].to(device), mb)
                    weight = int(mb.sum()) * adapter.spec.d_in
                    error += float(loss) * weight
                    elements += weight
            metrics["val_loss"] = error / elements
            if not math.isfinite(metrics["val_loss"]):
                raise ValueError("non-finite validation loss")
        score = metrics.get("val_loss", metrics["train_loss"])
        if score < best:
            best = score
            torch.save({**adapter.export(), "metrics": metrics, "train_steps": step}, path)
        history.append(metrics)
        log(metrics)
    adapter.eval()
    (out_dir / "adapter_metrics.json").write_text(json.dumps(history, indent=2))
    return path


def train_specialist(learner, replay, updates: int, output=None, log=print):
    """Additional learner updates; online loops can append newly collected windows between calls."""
    if updates <= 0 or not len(replay):
        raise ValueError("positive updates and a nonempty replay buffer are required")
    metrics = {}
    for i in range(updates):
        metrics = learner.update(replay.sample(learner.cfg.batch_size))
        if (i + 1) % 100 == 0 or i + 1 == updates:
            log(metrics)
    if output is not None:
        output = Path(output)
        save_specialist(output, learner)
        replay.save(output.with_suffix(".replay.npz"))
        output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

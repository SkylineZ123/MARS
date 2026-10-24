"""Cached-data CLI: python -m mars.rltoken --help."""

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml

from .adapter import AdapterSpec, RLTAdapter
from .config import AdapterTrainConfig, SpecialistConfig, load_config
from .episode import load_episode
from .learner import SpecialistLearner, load_specialist
from .replay import PHASE_WARMUP, TRAIN_KEYS, ReplayBuffer
from .train import train_adapter, train_specialist
from .windows import build_windows


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL Token adapter and TD3 chunk specialist training")
    commands = parser.add_subparsers(dest="command", required=True)
    adapter = commands.add_parser("train-adapter", help="train on NPZ frozen tokens and validity masks")
    adapter.add_argument("--data", required=True)
    adapter.add_argument("--val-data")
    adapter.add_argument("--config", required=True)
    adapter.add_argument("--output", required=True, help="directory receiving adapter.pt")
    specialist = commands.add_parser("train-specialist", help="train on NPZ windows; resume with paired replay")
    specialist.add_argument("--data", help="required for a new learner; omit when resuming")
    specialist.add_argument("--config", required=True)
    specialist.add_argument("--output", required=True, help="specialist checkpoint (.pt)")
    specialist.add_argument("--updates", required=True, type=int)
    specialist.add_argument("--resume")
    windows = commands.add_parser("build-windows", help="convert a normalized EpisodeLog into skill windows")
    windows.add_argument("--episode", required=True)
    windows.add_argument("--skill", required=True, help="skill predicate name; ignored for task rewards")
    windows.add_argument("--controller-id", required=True, type=int)
    windows.add_argument("--config", required=True)
    windows.add_argument("--output", required=True)
    for cmd in (adapter, specialist):
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--threads", type=int, default=2)
        cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "build-windows":
        cfg = load_config(args.config, "specialist", SpecialistConfig)
        skill = args.skill if cfg.reward == "skill" else "task"
        data, stats = build_windows(load_episode(args.episode), args.controller_id, skill, cfg.C,
                                   alignment=cfg.ref_alignment, pad_mode=cfg.pad_mode,
                                   handoff_gamma=cfg.gamma if cfg.reward == "handoff_task" else None)
        if not data:
            parser.error(f"episode contains no valid specialist windows: {stats}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **data)
        print(stats)
        return
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.command == "train-adapter":
        cfg = load_config(args.config, "adapter_train", AdapterTrainConfig)
        cfg.seed = args.seed
        with np.load(args.data, allow_pickle=False) as data:
            tokens, mask = data["tokens"], data["mask"]
            groups, sizes = data["token_groups"].tolist(), data["group_sizes"].tolist()
        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        spec_data = dict(raw.get("adapter", {}))
        spec_data.setdefault("d_in", tokens.shape[-1])
        spec_data.setdefault("n_slots", tokens.shape[1])
        unknown = set(spec_data) - {f.name for f in fields(AdapterSpec)}
        if unknown:
            parser.error(f"unknown adapter keys: {sorted(unknown)}")
        model = RLTAdapter(AdapterSpec(**spec_data), groups, sizes)
        if sum(sizes) != tokens.shape[1]:
            parser.error("group_sizes must describe the complete cached token sequence")
        val = None
        if args.val_data:
            with np.load(args.val_data, allow_pickle=False) as data:
                if data["token_groups"].tolist() != groups or data["group_sizes"].tolist() != sizes:
                    parser.error("training and validation token groups differ")
                val = data["tokens"], data["mask"]
        print(train_adapter(model, tokens, mask, cfg, args.output, args.device, val=val))
    else:
        cfg = load_config(args.config, "specialist", SpecialistConfig)
        if args.resume:
            if args.data:
                parser.error("resume restores saved replay; append online data through the Python API")
            learner = load_specialist(args.resume, cfg, args.device, resume=True)
            replay = ReplayBuffer.load(Path(args.resume).with_suffix(".replay.npz"))
        else:
            if not args.data:
                parser.error("--data is required for a new specialist")
            with np.load(args.data, allow_pickle=False) as data:
                batch = {k: data[k] for k in data.files if not k.startswith("_")}
            if set(TRAIN_KEYS) - set(batch):
                parser.error("data must contain the RL Token replay window fields")
            if batch["act"].ndim != 3 or batch["act"].shape[1] != cfg.C:
                parser.error("data action chunk length differs from specialist.C")
            replay = ReplayBuffer(cfg.buffer_capacity, seed=args.seed)
            replay.add_batch(batch, PHASE_WARMUP, episode_id=-1)
            learner = SpecialistLearner(batch["z"].shape[-1], batch["q"].shape[-1], batch["act"].shape[-1],
                                        cfg, args.device)
        train_specialist(learner, replay, args.updates, args.output)


if __name__ == "__main__":
    main()

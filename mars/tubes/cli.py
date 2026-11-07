"""Run or evaluate the released tubes simulation."""
import argparse
import json
import os
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--assets", type=Path, required=True)
        p.add_argument("--config", type=Path)
        p.add_argument("--seed", type=int)
        p.add_argument("--device", default="cuda")
        p.add_argument("--mode", choices=("mars", "base", "base_c5"), default="mars" if name == "run" else None)
        p.add_argument("--scene", type=int, default=30050)
        p.add_argument("--noise-seed", type=int, default=0)
        p.add_argument("--output", type=Path)
        if name == "run":
            p.add_argument("--max-steps", type=int)
        else:
            p.add_argument("--episodes", type=int, default=100, help="episodes per model seed (two noise seeds per scene)")
            p.add_argument("--modes", nargs="+", choices=("base", "base_c5", "mars"), default=["base", "base_c5", "mars"])
            p.add_argument("--model-seeds", nargs="+", type=int)
            p.add_argument("--noise-seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if not 1 <= args.episodes <= 100:
            parser.error("--episodes must be in 1..100")
        for name in ("model_seeds", "noise_seeds", "modes"):
            values = getattr(args, name)
            if values is not None and len(set(values)) != len(values):
                parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
        if args.episodes % len(args.noise_seeds):
            parser.error("--episodes must be divisible by the number of noise seeds")
        scenes = range(args.scene, args.scene + args.episodes // len(args.noise_seeds))
        if scenes.start < 30050 or scenes.stop > 30100:
            parser.error("evaluation scenes must be within 30050..30099")
    # Read only lightweight configuration before importing models or the renderer.
    config_path = args.config or args.assets / "simulation.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text())
        if not isinstance(cfg, dict) or cfg.get("format") != "mars-tubes-simulation-v1":
            raise ValueError("unsupported simulation config")
        available = cfg.get("seeds")
        if not isinstance(available, dict) or not available or any(type(s) is not int for s in available):
            raise ValueError("simulation config requires integer model seed keys")
        available = sorted(available)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.command == "run":
        args.seed = args.seed if args.seed is not None else available[0]
        model_seeds = [args.seed]
    else:
        model_seeds = [args.seed] if args.seed is not None else (
            args.model_seeds if args.model_seeds is not None else available)
    missing = sorted(set(model_seeds) - set(available))
    if missing:
        parser.error(f"unavailable model seeds {missing}; available model seeds: {available}")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from .runtime import SimulationRuntime, evaluate, summarize_modes, reproducibility
    from .environment import TubesEnv
    env = TubesEnv(render=True)
    try:
        if args.command == "run":
            runtime = SimulationRuntime(args.assets, args.seed, args.device, args.mode, args.config)
            result = runtime.run(env, args.scene, args.noise_seed, args.max_steps)
            result["reproducibility"] = reproducibility(runtime, [args.scene], [args.noise_seed])
            result["reproducibility"]["max_steps"] = args.max_steps or runtime.cfg["max_steps"]
        else:
            runs = []
            modes = [args.mode] if args.mode is not None else args.modes
            for mode, seed in ((mode, seed) for mode in modes for seed in (model_seeds if mode == "mars" else [available[0]])):
                runtime = SimulationRuntime(args.assets, seed, args.device, mode, args.config)
                runs.append(evaluate(runtime, env, scenes, args.noise_seeds))
                del runtime
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows = [row for run in runs for row in run["episodes"]]
            result = {"model_seeds": model_seeds, "modes": modes, "runs": runs,
                      "by_mode": summarize_modes(rows)}
    finally:
        env.close()
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()

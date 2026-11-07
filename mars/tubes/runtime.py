"""Portable simulation deployment using the public MARS controller and loaders."""
from pathlib import Path
import numpy as np
import torch
import yaml
from mars.backbones import make_backbone
from mars.controller import MARSController
from mars.rltoken.adapter import load_adapter
from mars.rltoken.controller import RoutedRLTPolicy
from mars.rltoken.learner import load_specialist
from .assets_cli import digest, safe_path, verify
from .noise import episode_key, vla_noise_key
from .protocol import SUITE


class RoundedExpert:
    """Replay stored z in fp16; actors consume those rounded values as fp32."""
    def __init__(self, expert):
        self.expert, self.C = expert, expert.C

    def act(self, z, q, ref, **kwargs):
        return self.expert.act(np.asarray(z).astype(np.float16).astype(np.float32), q, ref, **kwargs)


class SimulationRuntime:
    def __init__(self, assets, seed=0, device="cuda", mode="mars", config=None):
        self.root = Path(assets)
        self.config_path = Path(config or self.root / "simulation.yaml")
        self.cfg = yaml.safe_load(self.config_path.read_text())
        if self.cfg.get("format") != "mars-tubes-simulation-v1":
            raise ValueError("unsupported simulation config")
        if mode not in ("mars", "base", "base_c5"):
            raise ValueError("mode must be mars, base or base_c5")
        if seed not in self.cfg.get("seeds", {}):
            raise ValueError(f"unavailable model seed {seed}; available model seeds: {sorted(self.cfg.get('seeds', {}))}")
        verify(self.root)
        self.model_seed = seed
        self.mode = mode
        bb = dict(self.cfg["backbone"], device=device)
        bb["pretrained_path"] = str(safe_path(self.root, bb["pretrained_path"]))
        self.backbone = make_backbone(bb)
        if self.backbone.c_vla != 10:
            raise ValueError("release requires c_vla=10")
        if mode == "mars":
            paths = self.cfg["seeds"][seed]
            controller = MARSController.from_checkpoint(safe_path(self.root, paths["router"]),
                                                        device=device, **self.cfg["switch"])
            if controller.classes != ["vla", "pickup", "insert"]:
                raise ValueError("unexpected release controller order")
            adapter = load_adapter(safe_path(self.root, self.cfg["adapter"]), device)
            experts = {name: RoundedExpert(load_specialist(safe_path(self.root, paths[name]), device=device))
                       for name in controller.classes[1:]}
            if any(e.C != 5 for e in experts.values()):
                raise ValueError("release requires experts C=5")
            if controller.router.z_dim != adapter.z_dim:
                raise ValueError("adapter/router z_dim mismatch")
            if controller.router.q_dim not in (0, self.backbone.d_q):
                raise ValueError("router/backbone q_dim mismatch")
            for name, wrapped in experts.items():
                expert = wrapped.expert
                if (expert.z_dim, expert.q_dim, expert.d_a) != (
                        adapter.z_dim, self.backbone.d_q, self.backbone.d_a):
                    raise ValueError(f"{name} expert dimensions mismatch adapter/backbone")
            self.policy = RoutedRLTPolicy(controller, adapter, self.reference, experts,
                                          query_interval=self.cfg["query_interval"])

    def reference(self, features, q):
        bb = self.backbone
        chunk = bb.reference_chunk(features, bb.make_noise(vla_noise_key(self.key, self.t)))
        # Generalist executes ten, experts consume the first five of this reference.
        return chunk[0, :5 if self.mode == "base_c5" else bb.c_vla].float().cpu().numpy()

    @torch.no_grad()
    def run(self, env, scene=30000, noise_seed=0, max_steps=None):
        from collections import deque
        self.key = episode_key(SUITE, 0, scene, noise_seed)
        self.t = 0
        if self.mode == "mars":
            self.policy.reset()
        queue = deque()
        obs = env.reset(init_id=scene, seed=noise_seed)
        info = {}
        limit = self.cfg["max_steps"] if max_steps is None else max_steps
        if not 1 <= limit <= self.cfg["max_steps"]:
            raise ValueError("invalid episode step budget")
        for t in range(limit):
            self.t = t
            bb = self.backbone
            if self.mode == "mars" or not queue:
                features = bb.prefill(bb.preprocess(obs, env.task_description))
                q = np.asarray(bb.normalize_state(bb.state_from_raw(obs)), dtype=np.float32)
            if self.mode == "mars":
                result = self.policy.step(features, q)
                action = result.action
            else:
                if not queue:
                    queue.extend(self.reference(features, q))
                action = queue.popleft()
            obs, success, info = env.step(np.clip(bb.unnormalize_action(action), -1, 1))
            if success or info.get("terminated") or info.get("truncated"):
                break
        return {"scene": scene, "noise_seed": noise_seed, "mode": self.mode,
                "model_seed": self.model_seed,
                "success": bool(success), "steps": t + 1,
                "completed_tubes": len(set(env.completed)),
                "total_tubes": env.spec.parts,
                "stages_completed": env.stage_index,
                "termination_reason": ("success" if success else info.get("failure") or
                                       ("budget" if info.get("truncated") or t + 1 == limit else "terminated")),
                "scr": len(set(env.completed)) / env.spec.parts}


def summarize(rows):
    if not rows:
        raise ValueError("evaluation grid is empty")
    successful = [r for r in rows if r["success"]]
    return {"n": len(rows), "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "completed_tubes": sum(r["completed_tubes"] for r in rows),
            "total_tubes": sum(r["total_tubes"] for r in rows),
            "scr": sum(r["completed_tubes"] for r in rows) / sum(r["total_tubes"] for r in rows),
            "steps_all": {"mean": float(np.mean([r["steps"] for r in rows])), "n": len(rows)},
            "steps_success_only": {"mean": float(np.mean([r["steps"] for r in successful])) if successful else None,
                                   "n": len(successful)}}


def summarize_modes(rows):
    result = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected = [row for row in rows if row["mode"] == mode]
        seeds = sorted({row["model_seed"] for row in selected})
        per_seed = {str(seed): summarize([row for row in selected if row["model_seed"] == seed])
                    for seed in seeds}
        rates = [item["success_rate"] for item in per_seed.values()]
        result[mode] = {**summarize(selected), "per_model_seed": per_seed,
                        "seed_success_rate_mean": float(np.mean(rates)),
                        "seed_success_rate_sample_std": float(np.std(rates, ddof=1)) if len(rates) >= 2 else None,
                        "seed_count": len(seeds)}
    return result


def reproducibility(runtime, scenes, noise_seeds):
    import platform
    from .protocol import TubesSpec, source_fingerprint
    code = Path(__file__).parents[1]
    return {"format": "mars-tubes-evaluation-v1", "mode": runtime.mode,
            "asset_manifest_sha256": digest(runtime.root / "manifest.json"),
            "config_sha256": digest(runtime.config_path),
            "runtime_sources": {str(p.relative_to(code)): digest(p) for p in sorted(code.rglob("*.py"))},
            "environment_source_sha256": source_fingerprint(), "task": vars(TubesSpec()),
            "scenes": list(scenes), "noise_seeds": list(noise_seeds),
            "model_seed": runtime.model_seed, "max_steps": runtime.cfg["max_steps"],
            "c_vla": 5 if runtime.mode == "base_c5" else 10, "expert_C": 5,
            "scr_definition": "unique protocol-confirmed insertion/release events divided by tube count",
            "expert_z": "fp16 rounded then fp32", "noise": "blake2b episode/environment-step keyed CPU torch.randn",
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}


def evaluate(runtime, env, scenes, noise_seeds=(0, 1)):
    scenes, noise_seeds = list(scenes), list(noise_seeds)
    rows = [runtime.run(env, scene, noise) for scene in scenes for noise in noise_seeds]
    return {"episodes": rows, **summarize(rows),
            "reproducibility": reproducibility(runtime, scenes, noise_seeds)}

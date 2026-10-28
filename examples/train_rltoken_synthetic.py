"""CPU smoke run of adapter -> skill windows -> TD3 -> MARS -> routed actions.

Run from the repository root: python -m examples.train_rltoken_synthetic
Synthetic observations/rewards exercise interfaces, not robot performance.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from mars import MARSConfig, MARSController, build_router, save_router, train_router
from mars.rltoken import (
    AdapterSpec, AdapterTrainConfig, EpisodeLog, PHASE_WARMUP, PrefixFeatures, RLTAdapter,
    ReplayBuffer, RoutedRLTPolicy, SOURCE_VLA, SpecialistConfig, SpecialistLearner, build_windows,
    load_adapter, load_specialist, train_adapter, train_specialist,
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="mars-rltoken-example-") as tmp:
        output = Path(tmp)
        T, d, C, action_dim, q_dim = 12, 8, 3, 2, 3
        hidden = torch.randn(T + 1, 6, d)
        mask = torch.ones(T + 1, 6, dtype=torch.bool)
        adapter = RLTAdapter(AdapterSpec(d, d, 6, enc_layers=1, dec_layers=1, heads=2, dropout=0),
                             ["image", "image2", "lang"], [2, 2, 2])
        path = train_adapter(adapter, hidden[:T], mask[:T], AdapterTrainConfig(batch_size=4, epochs=2),
                             output, log=lambda _: None)
        adapter = load_adapter(path)
        with torch.no_grad():
            z = adapter.encode(hidden, mask).numpy()
        q = rng.normal(size=(T + 1, q_dim)).astype(np.float32)
        owner = np.array([0] * 3 + [1] * 6 + [0] * 3)
        log = EpisodeLog(meta={"end_reason": "budget"})
        log.exec_norm = list(rng.normal(size=(T, action_dim)).astype(np.float32))
        log.skill_owner, log.source = list(owner), [SOURCE_VLA] * T
        log.skill_success_after = {"pickup": [False] * T}
        log.skill_success_after["pickup"][8] = True
        log.success_after = [False] * T
        log.z, log.z_valid, log.q_norm = list(z), [True] * (T + 1), list(q)
        log.vla_t = [0, 6]
        log.vla_chunks = [np.zeros((6, action_dim), np.float32)] * 2
        windows, stats = build_windows(log, k=1, skill="pickup", C=C)
        replay = ReplayBuffer(64, seed=0)
        replay.add_batch(windows, PHASE_WARMUP, episode_id=0)
        cfg = SpecialistConfig(C=C, actor_hidden=[16, 16], critic_hidden=[16, 16], batch_size=4,
                               buffer_capacity=64)
        learner = SpecialistLearner(d, q_dim, action_dim, cfg)
        train_specialist(learner, replay, updates=4, output=output / "pickup.pt", log=lambda _: None)

        f = PrefixFeatures(hidden[:T], {"image": slice(0, 2), "image2": slice(2, 4), "lang": slice(4, 6)},
                           {g: torch.ones(T, 2, dtype=torch.bool) for g in ("image", "image2", "lang")})
        packed = z[:T]
        router_cfg = MARSConfig(z_dim=d, q_dim=q_dim, d_model=16, heads=2, dropout=0,
                                skill_query=False, ssm_layers=1, epochs=2, batch_episodes=1)
        router = build_router("mars", d, 2, router_cfg)
        # Direct classification head is the existing MARS ablation, avoiding a text model download.
        train_router(router, [(packed, owner, q[:T])], [(packed, owner, q[:T])], router_cfg,
                     log=lambda _: None)
        save_router(output / "mars.pt", router, ["vla", "pickup"])
        ctrl = MARSController.from_checkpoint(output / "mars.pt", k=1, tau=0.5)
        policy = RoutedRLTPolicy(ctrl, adapter, lambda features, q: np.zeros((6, action_dim), np.float32),
                                 {"pickup": load_specialist(output / "pickup.pt")})
        for t in range(T):
            step_features = PrefixFeatures(hidden[t:t + 1], f.slices,
                                          {g: m[t:t + 1] for g, m in f.group_masks.items()})
            result = policy.step(step_features, q[t])
            assert result.action.shape == (action_dim,) and np.isfinite(result.action).all()
        print(f"OK: trained adapter, {stats['windows']} replay windows, {learner.n_updates} TD3 updates; "
              f"trained/restored MARS and executed {T} routed actions on CPU")


if __name__ == "__main__":
    main()

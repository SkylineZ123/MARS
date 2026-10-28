# MARS: Memory-Augmented Router for atomic Skills

[中文说明](README.zh-CN.md)

MARS decides, at every control step, **which controller should act**: a frozen generalist
vision-language-action (VLA) policy or one of several atomic-skill specialists (for example
*Pickup* and *Place*). Its input at each step is the shared RL Token (which carries the images and task language
through the frozen VLA) and the normalized proprioception. It keeps a dense causal memory
of the episode and scores every candidate controller through a query built from that
controller's plain-language capability description. A k-of-k confirmation rule turns the scores
into reliable handoffs.

This repository contains the router, candidate-text encoding, skill switcher, shared RL Token
adapter, TD3 chunk specialists, replay, training and routed execution. Supply a frozen VLA and
environment through the feature and normalized-action interfaces. The RL modules are adapted
from RL-Token-Pi05-open's `routerlt` implementation; see the [integration guide](docs/rltoken.zh-CN.md).

## Method

```mermaid
flowchart LR
    O[Images, task text, proprioception] --> F[Frozen VLA + shared RL token]
    F --> M[RL Token + proprioception embedding]
    M --> S[Causal Mamba memory]
    S --> H[Last L memory states as K/V]
    T[Candidate capability descriptions] --> E[Frozen SigLIP2 text encoder, cached once]
    E --> Q[Shared text projection -> one query per candidate]
    Q --> A[Query self-attention + memory cross-attention]
    H --> A
    A --> P[Shared score MLP + softmax]
    P --> W[Threshold tau + k consecutive confirmations]
    W --> C[Run the VLA or the selected specialist]
```

1. **State embedding.** Only the shared RL Token z_t and normalized proprioception q_t
   enter the router. z_t goes through Linear + LayerNorm, q_t through a two-layer MLP + LayerNorm,
   and the two results are summed (d_model = 128) with dropout. Images and task language
   are not router inputs; they reach it through the frozen VLA and the RL Token adapter.
2. **Causal memory.** A two-layer selective state-space (Mamba) trunk advances its state by one
   step per router query (O(1) at deployment). The last L = 8 trunk outputs, plus a learned
   embedding of their age, are kept as keys and values.
3. **Candidate-skill queries.** Each candidate is described in text, for example *"grasp the black
   bowl by its rim and lift it up"*. The frozen SigLIP2 text encoder embeds the descriptions once.
   A shared trainable projection turns each embedding into a query. The queries pass through two
   blocks of self-attention among themselves, cross-attention over the memory and a feed-forward layer. A shared score MLP gives
   one logit per candidate, and a softmax gives the probabilities. The queries come only from the
   candidate texts; there is no skill-ID input or learned per-controller embedding. The candidate
   list sets the output order, and permuting it permutes the outputs.
4. **Skill switching.** The top-scoring controller takes over once its probability reaches tau
   on k consecutive queries (defaults tau = 0.6, k = 2, cooldown 0). Below the threshold the
   current controller keeps control. Every handoff discards the previous controller's queued
   action chunk and queries the new controller at the same step.

With the default configuration (`configs/mars.yaml`, SmolVLA feature width 960, three
candidates and the 768-wide SigLIP2 text embedding), MARS has 1,021,953 trainable parameters.

## Install

```bash
pip install -e ".[test]"          # torch, numpy, pyyaml (+ pytest)
pip install -e ".[siglip2]"       # only needed to encode new candidate descriptions
pytest                            # CPU unit tests
python examples/train_synthetic.py
```

## Usage

### 1. Encode the candidate descriptions

```bash
python -m mars.text --config configs/mars.yaml --classes vla pickup place --output siglip2_text.pt
```

The cache stores the encoder name, the candidate order and the exact descriptions.
`load_text_cache` refuses a cache that does not match the current configuration.

### 2. Train

```python
from mars import MARSConfig, build_router, save_router, train_router
from mars.text import candidate_embeddings

cfg = MARSConfig.from_yaml("configs/mars.yaml")
classes = ["vla", "pickup", "place"]                  # index 0 is the generalist
text, spec = candidate_embeddings(cfg, classes, out_dir="cache")
router = build_router(cfg.type, cfg.z_dim, len(classes), cfg, text_dim=text.shape[1])
router.set_text_embeddings(text)

# train_seqs / val_seqs: lists of (z (T, z_dim), y (T,), q (T, q_dim)) per episode
result = train_router(router, train_seqs, val_seqs, cfg, seed=cfg.seed)
save_router("mars.pt", router, classes, extra={"text_spec": spec})
```

Labels equal to `mars.train.IGNORE` (-100) are skipped. The loss is cross-entropy weighted by inverse label frequency. With `select_best` the epoch with
the best validation `selection_metric` (macro-F1 by default) is kept.

### 3. Deploy

```python
from mars import MARSController

ctrl = MARSController.from_checkpoint("mars.pt", cfg)
ctrl.reset()                                           # at every episode start
for t in range(max_steps):
    z_t = rl_token[0]   # (z_dim,)
    active, switched, probs = ctrl.query(z_t, q_t)
    if switched:
        action_queue.clear()                           # the old controller's chunk is invalid
    if not action_queue:
        action_queue.extend(policies[ctrl.active_name](obs))
    env.step(action_queue.popleft())
```

### 4. Offline evaluation

`mars.metrics.selector_report` reports per-step accuracy, macro-F1 and specialist F1. It also
reports per-interval IoU, false entries, missed intervals and entry/exit lag. With
`switch_grid=[(tau, k), ...]` and the per-step probabilities, it computes the same metrics for
the *deployed* owner after the switcher is replayed.

## RL Token and specialist learning

```bash
python -m examples.train_rltoken_synthetic  # CPU smoke run of the complete module chain
python -m mars.rltoken --help

python -m mars.rltoken train-adapter --data tokens.npz \
  --config configs/rltoken.yaml --output outputs/adapter --device cuda
python -m mars.rltoken train-specialist --data pickup_windows.npz \
  --config configs/rltoken.yaml --output outputs/pickup.pt --updates 5000 --device cuda
```

`mars.rltoken` provides `RLTAdapter`, `SpecialistLearner`, `ReplayBuffer`, `EpisodeLog`,
`build_windows`, checkpoint loaders, and `RoutedRLTPolicy`. Cached adapter data contains
`tokens (N, slots, d_vla)`, `mask (N, slots)`, `token_groups`, and `group_sizes` as NPZ arrays.
Specialist windows contain `z`, `q`, `ref`, `act`, `rew`, `valid`, `done`, `next_z`, `next_q`,
and `next_ref`; use `build-windows` to construct them from an episode. Resume training with
`--resume outputs/pickup.pt` and its paired `.replay.npz` file; `--updates` counts additional updates.

During deployment, `RoutedRLTPolicy` shares one frozen adapter across the router and all
specialists, calls a user-supplied VLA reference callback, and clears queued actions immediately
when MARS switches controllers. Actions and proprioception use the same normalization as training;
unnormalize actions in the environment adapter. See the [data and deployment guide](docs/rltoken.zh-CN.md).
Existing `routerlt` adapter and chunk specialist checkpoints are supported (supply the matching
config for older specialist checkpoints); the original pi05 causal-decoder and DSRL checkpoints
use different formats. VLA/robot drivers and simulator environments are supplied separately.

## Tube-transfer workcell

`mars.tubes` is a robosuite 1.4 / MuJoCo Panda workcell (`tubes_v2`) in which three tubes are moved
one by one from a fixed source rack into a randomised target rack, with a privileged scripted
controller and ground-truth skill labels (`pickup`, `insert`). It needs `pip install -e .[tubes]` and
`NUMBA_DISABLE_JIT=1`; see [docs/tubes.zh-CN.md](docs/tubes.zh-CN.md).

## Inputs

| Input | Shape | Source |
|---|---|---|
| `z_t` | `z_dim` | shared RL Token from the adapter; images and language are encoded upstream |
| `q_t` | `q_dim` | normalized proprioception (`use_q: false` drops it) |
| `y_t` | scalar | controller label, 0 = generalist |

Training sequences are per-episode tuples `(z, y, q)`. Candidate descriptions are not a per-step
input: they are encoded once into a `(num_classes, text_dim)` array and set with `set_text_embeddings`.
No phase, progress, reward or simulator state enters the router.

`pack_modalities` and the `multimodal_mars` / `multimodal_mars_gru` kinds are the earlier
four-input variant (`[z, mean(image), mean(image2), mean(language)]`, sized by `modality_dims`).
They are kept only for legacy checkpoints; `fusion` and `modality_dims` in the config apply to them alone.

## Router kinds

| Kind | Description |
|---|---|
| `mars` | MARS: RL Token + proprioception + Mamba memory + candidate-skill text queries |
| `mars_gru` | the same, with a GRU memory trunk |
| `mars` + `skill_query: false` | direct linear classifier instead of candidate queries (ablation) |
| `single_frame` | same embedding and text queries, no memory trunk: the current step only (ablation) |
| `mamba_head`, `mamba_query`, `gru_query` | recurrent trunk on the same RL Token + proprioception embedding; `mamba_head` has a linear head, the query kinds read a memory window spaced by `window_stride` |
| `gru` | GRU on the concatenated `[z, q]`, linear head (comparison) |
| `windowed_query`, `windowed_tf`, `windowed_xattn` | windowed query router: a causal window of L samples taken `window_stride` steps apart, instead of a dense memory; z and q are summed per sample (comparison) |

All kinds share the same interface: `forward_seq(z, q)` for training (causal over time), and
`init_state()` + `step(z_t, state, q_t)` for streaming. `step` matches `forward_seq` up to floating-point error.
Every kind that uses candidate queries builds them from the candidate texts. `load_router`
refuses checkpoints that use learned skill-ID queries.

## Layout

```
mars/
  router.py      MARSRouter, SingleFrameRouter, RecurrentRouter, legacy MultimodalFusionRouter, pack_modalities
  layers.py      MambaBlock, QueryBlock, SelfBlock, window helpers
  baselines.py   GRURouter, WindowedQueryRouter
  text.py        SigLIP2 candidate-text encoding and validated cache
  switch.py      SkillSwitcher (tau, k-of-k, cooldown), replay_switch
  controller.py  MARSController: streaming router + switcher
  train.py       train_router, predict, class_weights
  metrics.py     selector_report and interval metrics
  registry.py    build_router, save_router, load_router
  config.py      MARSConfig
  rltoken/       Shared adapter, TD3 specialists, replay windows, routed execution and CLI
  tubes/         Tube-transfer robosuite workcell, scripted controller and skill labels
configs/mars.yaml
configs/rltoken.yaml
examples/train_synthetic.py
tests/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

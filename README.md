# MARS: Memory-Augmented Router for atomic Skills

[中文说明](README.zh-CN.md)

MARS decides, at every control step, **which controller should act**: a frozen generalist
vision-language-action (VLA) policy or one of several atomic-skill specialists (for example
*Pickup* and *Place*). It reads the current multimodal observation, keeps a dense causal memory
of the episode, and scores every candidate controller through a query built from that
controller's plain-language capability description. A k-of-k confirmation rule turns the scores
into reliable handoffs.

This repository contains the router, the candidate-text encoding, the skill switcher, training
and offline metrics. The VLA backbone, the RL-token encoder and the specialists are not included.
MARS only needs per-step feature vectors from them (see [Inputs](#inputs)).

## Method

```mermaid
flowchart LR
    O[Images, task text, proprioception] --> F[Frozen VLA + shared RL token]
    F --> M[Multimodal attention fusion]
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

1. **Modality fusion.** The RL token z_t, mean-pooled features of two cameras and of the task
   language, and the normalized proprioception q_t are projected per modality. One Transformer
   layer mixes them, and a learned attention query pools them into one token per step.
2. **Causal memory.** A selective state-space (Mamba) trunk advances its state by one step per
   router query (O(1) at deployment). The last L = 8 trunk outputs are kept as keys and values.
3. **Candidate-skill queries.** Each candidate is described in text, for example *"grasp the black
   bowl by its rim and lift it up"*. The frozen SigLIP2 text encoder embeds the descriptions once.
   A shared trainable projection turns each embedding into a query. The queries run
   self-attention among themselves and cross-attention over the memory. A shared score MLP gives
   one logit per candidate, and a softmax gives the probabilities. There are no per-controller ID
   embeddings: the candidate list sets the output order, and permuting it permutes the outputs.
4. **Skill switching.** The top-scoring controller takes over once its probability reaches tau
   on k consecutive queries (defaults tau = 0.6, k = 2, cooldown 0). Below the threshold the
   current controller keeps control. Every handoff discards the previous controller's queued
   action chunk and queries the new controller at the same step.

With the default configuration (`configs/mars.yaml`, SmolVLA feature width 960, three
candidates), MARS has 1,382,161 trainable parameters.

## Install

```bash
pip install -e ".[test]"          # torch, numpy, pyyaml (+ pytest)
pip install -e ".[siglip2]"       # only needed to encode new candidate descriptions
pytest                            # CPU unit tests, about 15 s
python examples/train_synthetic.py
```

## Usage

### 1. Encode the candidate descriptions (once)

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
router = build_router(cfg.type, sum(cfg.modality_dims), len(classes), cfg, text_dim=text.shape[1])
router.set_text_embeddings(text)

# train_seqs / val_seqs: lists of (z (T, sum(modality_dims)), y (T,), q (T, q_dim)) per episode
result = train_router(router, train_seqs, val_seqs, cfg, seed=cfg.seed)
save_router("mars.pt", router, classes, extra={"text_spec": spec})
```

The loss is cross-entropy weighted by inverse label frequency. With `select_best` the epoch with
the best validation `selection_metric` (macro-F1 by default) is kept.

### 3. Deploy

```python
from mars import MARSController, pack_modalities

ctrl = MARSController.from_checkpoint("mars.pt", cfg)
ctrl.reset()                                           # at every episode start
for t in range(max_steps):
    z_t = pack_modalities(prefix_features, rl_token)[0]   # (sum(modality_dims),)
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

## Inputs

| Input | Shape | Source |
|---|---|---|
| `z_t` | `sum(modality_dims)` | `pack_modalities(features, z)`: [RL token, mean(camera 1), mean(camera 2), mean(language)] |
| `q_t` | `q_dim` | normalized proprioception |
| `y_t` (training only) | scalar | controller label, 0 = generalist |

`features` is the frozen VLA prefix of the current observation. It must provide `.hidden`
`(B, N, d)`, `.slices` (token ranges named `image`, `image2` and `lang`) and `.group_masks`
(token validity per range). No phase, progress, reward or simulator state enters the router.

## Router kinds

| Kind | Description |
|---|---|
| `mars` | MARS: modality fusion + Mamba memory + candidate-skill text queries |
| `mars_gru` | the same, with a GRU memory trunk |
| `mars` + `skill_query: false` | direct linear classifier instead of candidate queries |
| `mars` + `query_mode: id` / `text_id` | learned ID queries / text + ID queries |
| `mamba_head`, `mamba_query`, `gru_query` | recurrent trunk on the packed vector, without modality fusion |
| `gru` | GRU recurrent router (comparison) |
| `windowed_query`, `windowed_tf`, `windowed_xattn` | windowed query router: a causal window of L samples taken `window_stride` steps apart, instead of a dense memory (comparison) |

All kinds share the same interface: `forward_seq(z, q)` for training (causal over time), and
`init_state()` + `step(z_t, state, q_t)` for streaming. `step` matches `forward_seq` up to floating-point error.

## Layout

```
mars/
  router.py      MARSRouter, RecurrentRouter, pack_modalities
  layers.py      MambaBlock, QueryBlock, SelfBlock, window helpers
  baselines.py   GRURouter, WindowedQueryRouter
  text.py        SigLIP2 candidate-text encoding and validated cache
  switch.py      SkillSwitcher (tau, k-of-k, cooldown), replay_switch
  controller.py  MARSController: streaming router + switcher
  train.py       train_router, predict, class_weights
  metrics.py     selector_report and interval metrics
  registry.py    build_router, save_router, load_router
  config.py      MARSConfig
configs/mars.yaml
examples/train_synthetic.py
tests/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

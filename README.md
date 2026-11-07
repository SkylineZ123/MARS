
# MARS: Memory-Augmented Router for Atomic Skills
<p align="center">
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="https://huggingface.co/XDEDM7/MARS">🤗 Model weight</a> ·
  <a href="LICENSE">License</a>
</p>

## Overview

MARS coordinates a frozen vision-language-action (VLA) policy and multiple reinforcement learning specialists for long-horizon manipulation. Shared RL Tokens, Mamba memory, and natural-language skill descriptions guide controller selection. Each handoff clears the previous controller's action queue.

This repository includes the **MARS router**, **RL Token adapter**, **TD3 specialist training components**, and a **Panda tube-transfer simulator**.

## Quick Start

### 1. Installation

Use Linux, Python 3.12, and an NVIDIA driver compatible with CUDA 12.6. Run the following commands from the repository root:

```bash
conda create -n mars python=3.12 pip -y
conda activate mars
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[smolvla,siglip2,tubes]"
```

### 2. Download Pretrained Models

Weights are available at [**XDEDM7/MARS**](https://huggingface.co/XDEDM7/MARS). Download the released revision:

```bash
mars-assets download --repo-id XDEDM7/MARS \
  --revision 815fca0803f88a21fa8062574411fdfb988f2143 \
  --output assets/tubes
```

The bundle contains the frozen VLA, shared adapter, one MARS router, and one Actor checkpoint for each pickup and insertion expert. File integrity is checked automatically during download.

### 3. Run Tube Transfer

The Panda robot transfers three tubes from a source rack to their corresponding slots in a target rack.

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
mars-tubes run --assets assets/tubes --seed 0 \
  --output outputs/tubes/run.json
```

**Optional batch evaluation:** the published bundle contains model seed **0** only.

```bash
mars-tubes evaluate --assets assets/tubes --model-seeds 0 --modes mars \
  --output outputs/tubes/evaluation.json
```

Use `mars-tubes run --help` or `mars-tubes evaluate --help` for additional options.

## Documentation

| Component | Code / Guide |
|---|---|
| Router and training | [Router](mars/router.py) · [Training](mars/train.py) · [Configuration](configs/mars.yaml) |
| RL Token and TD3 specialists | [Data formats and training interfaces](docs/rltoken.zh-CN.md) |
| Tube simulation | [Environment and task protocol](docs/tubes.zh-CN.md) |
| Rack and tube meshes | [STL assets](mars/tubes/assets/README.md) |

The detailed guides are currently available in Chinese.

## License

This code is released under the [Apache License 2.0](LICENSE). External models and datasets remain subject to their respective licenses.

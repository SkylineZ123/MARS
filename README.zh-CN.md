# MARS：面向长时序机器人操作的原子技能路由

[English](README.md) · [模型权重](https://huggingface.co/XDEDM7/MARS)

MARS 在冻结的视觉–语言–动作模型（VLA）与多个强化学习技能专家之间动态选择控制器。它结合共享 RL Token、Mamba 历史记忆和技能文本描述，判断何时调用专家，并在交接时清空旧动作队列。

本仓库提供路由器、RL Token 适配器、TD3 专家训练组件，以及 Panda 机械臂的试管搬运仿真。

## 安装

推荐使用 Linux、Python 3.12 和支持 CUDA 12.6 的 NVIDIA 驱动。在仓库根目录执行：

```bash
conda create -n mars python=3.12 pip -y
conda activate mars
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[smolvla,siglip2,tubes]"
```


## 下载权重

从 [XDEDM7/MARS](https://huggingface.co/XDEDM7/MARS) 下载已发布版本：

```bash
mars-assets download --repo-id XDEDM7/MARS \
  --revision 815fca0803f88a21fa8062574411fdfb988f2143 \
  --output assets/tubes
```

资源包含冻结 VLA、共享适配器、一个 MARS 路由器，以及抓取和插入专家各一个 Actor 检查点。下载工具会校验文件完整性。

## 试管搬运仿真任务

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
mars-tubes run --assets assets/tubes --seed 0 \
  --output outputs/tubes/run.json
```

可选的批量评测命令如下：

```bash
mars-tubes evaluate --assets assets/tubes --model-seeds 0 --modes mars \
  --output outputs/tubes/evaluation.json
```

参数说明见 `mars-tubes run --help` 和 `mars-tubes evaluate --help`。

## 代码与文档

| 内容 | 入口 |
|---|---|
| 路由器与训练 | `mars/router.py`、`mars/train.py`、`configs/mars.yaml` |
| RL Token 与 TD3 专家 | [数据格式和训练接口](docs/rltoken.zh-CN.md) |
| 试管环境与任务协议 | [仿真说明](docs/tubes.zh-CN.md) |
| 试管架与试管 STL | [模型资源](mars/tubes/assets/README.md) |


## 许可证

代码采用 [Apache License 2.0](LICENSE)。外部模型与数据遵循各自的许可条款。

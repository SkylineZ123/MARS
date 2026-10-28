# MARS 中的 RL Token 与技能专家

MARS 现在包含共享表示训练、技能专家强化学习和路由执行。入口为 `mars.rltoken`，只依赖已有的
Torch、NumPy 和 PyYAML。冻结 VLA 的加载、图像预处理、观测/动作归一化及环境交互由调用方提供。
本模块可以接收 π0.5、SmolVLA 或其他 VLA 的冻结前缀特征；整个训练和执行过程必须使用同一个
VLA、同一个 adapter 和同一套归一化统计。

## 来源与迁移范围

代码来源是 `RL-Token-Pi05-open` 的 `routerlt` 实现，与 MARS 现有多专家架构对应：

| 当前模块 | 原仓库来源 | 职责 |
|---|---|---|
| `adapter.py` | `routerlt/adapter/model.py` | 读出 RL Token，使用并行位置查询重建冻结特征 |
| `train.py` | `routerlt/adapter/train.py` 的训练流程 | 缓存特征训练、标准化、验证集选优 |
| `networks.py` | `routerlt/specialist/networks.py` | 参考动作条件化的完整动作块 Actor、双 Critic |
| `losses.py`、`schedules.py` | 同名 `routerlt/specialist` 文件 | TD3 目标、参考约束、reference dropout、权重调度 |
| `learner.py`、`replay.py` | 同名 `routerlt/specialist` 文件 | 延迟 Actor 更新、Polyak 目标、经验回放和恢复 |
| `episode.py`、`windows.py` | `routerlt/runtime/episode_log.py`、`routerlt/specialist/windows.py` | 逐步记录、技能区间、重叠窗口、人工纠正 |
| `features.py` | `routerlt/backbone/base.py` 的特征约定 | 与冻结 VLA 对接，不导入原工程 |
| `controller.py`、`cli.py` | 本次新增 | 与 MARS 路由器连接、独立命令行入口 |

上游 `routerlt` 的 Actor/Critic 和更新流程进一步来自
`lerobot/src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py` 与
`scripts/train_rlt_stage2_pi05.py`。这里使用 `routerlt` 的并行解码 adapter 和双隐藏层专家，
与 π0.5 原版的因果教师强制解码器、头部配置不同。

**检查点兼容性**：可直接读取 `routerlt` 的 `adapter.pt`；已有 chunk 专家检查点可用
`load_specialist(path, cfg=对应的SpecialistConfig)` 读取。新检查点内含配置，可直接加载。
π0.5 原版的 `best_checkpoint.pt`、DSRL 噪声空间专家不属于本模块格式。VLA/机器人驱动、
LIBERO、几何标注器和仿真脚本仍由各自环境提供；这里没有复制这些环境依赖。

## 安装与快速验证

```bash
pip install -e ".[test]"
pytest
python -m examples.train_rltoken_synthetic
python -m mars.rltoken --help
```

示例使用 CPU 合成观测，完成 adapter 训练、窗口构造、4 次 TD3 更新、路由器训练、检查点加载、
12 步路由执行，只验证模块连接和数值有效性，不衡量任务成功率。

## 1. 训练共享 RL Token

冻结 VLA 的输出需要符合以下接口（也可直接传入原 `routerlt` 的 `PrefixFeatures`）：

```python
from mars.rltoken import PrefixFeatures, token_groups

features = PrefixFeatures(
    hidden=hidden,               # (B, N, d_vla)，冻结 VLA 输出
    slices={"image": image_slice, "image2": image2_slice, "lang": language_slice},
    group_masks={"image": image_mask, "image2": image2_mask, "lang": language_mask},
)
tokens, mask = token_groups(features, ["image", "image2", "lang"], group_sizes)
```

`group_sizes` 是每组的固定槽位数，训练与部署一致。函数将不足的组填充并屏蔽，超长组报错。
语言截断需在 VLA 提取阶段显式完成。`mask=True` 表示有效 token。

将多个 batch 合并为训练缓存，保存组元数据：

```python
import numpy as np

np.savez("tokens.npz", tokens=tokens.cpu().numpy(), mask=mask.cpu().numpy(),
         token_groups=np.array(["image", "image2", "lang"]), group_sizes=np.array(group_sizes))
```

```bash
python -m mars.rltoken train-adapter \
  --data tokens_train.npz --val-data tokens_val.npz \
  --config configs/rltoken.yaml --output outputs/adapter --device cuda
```

产出 `adapter.pt` 和 `adapter_metrics.json`；提供验证集时选择验证损失最低的 epoch。
缓存格式为 `tokens (N, slots, d_vla)`、`mask (N, slots)`、`token_groups`、`group_sizes`。
默认 `d_in`、`n_slots` 从缓存推断，`d_model=960` 与当前 MARS 配置匹配。
使用其他 VLA 时同时修改 `configs/rltoken.yaml` 和 `configs/mars.yaml` 的特征维度。
原 π0.5 的 mmap 缓存不由此 CLI 直接读取；可在 Python API 中传入已读取的 token/mask，
并明确声明其 token 组。图像专用缓存不应被标成包含语言的缓存。

## 2. 从 rollout 构造专家训练数据

`EpisodeLog` 区分两类索引：T 个动作对应 T 条执行记录，T+1 个观测对应 T+1 个 `z/q`。

| 字段 | 内容 |
|---|---|
| `exec_norm` | T 个实际执行的归一化动作，包含人工纠正 |
| `skill_owner` | 每个动作所属专家，0 为通用策略，与 MARS 候选索引一致 |
| `source` | `SOURCE_VLA`、`SOURCE_RL` 或 `SOURCE_EXPERT` |
| `skill_success_after` | `{技能名: T 个 bool}`，执行动作后的技能成功信号 |
| `success_after` | T 个任务成功信号，供 task/handoff_task 奖励使用 |
| `rewards` | 可选 T 个环境奖励；提供后优先于成功信号 |
| `z`、`z_valid`、`q_norm` | T+1 个共享 RL Token、有效标志、归一化本体状态 |
| `vla_t`、`vla_chunks` | 通用策略查询时间和完整归一化动作块，包括最后状态的参考来源 |
| `meta["end_reason"]` | `budget` 表示时间截断，其余结束在区间边界作为终止 |

```python
from mars.rltoken import EpisodeLog, build_windows, save_episode

log = EpisodeLog(meta={"end_reason": "budget"})
# 按上表填充每步数据；采集期间保持列表，finalize 后即为只读训练记录。
save_episode("rollout.pt", log)
windows, stats = build_windows(log, k=1, skill="pickup", C=10)
```

CLI 提供相同转换：

```bash
python -m mars.rltoken build-windows --episode rollout.pt --skill pickup \
  --controller-id 1 --config configs/rltoken.yaml --output pickup_windows.npz
```

每个技能区间内，每一步产生一个长度不超过 C 的重叠窗口。成功或交接令 `done=1`；
时间截断保留 bootstrap，下一观测必须有效。奖励按实际执行长度 L 使用 `gamma**L`
引导下一状态价值。人工纠正的动作替换对应 reference 行；窗口不跨越目标或指令边界。
`reward: handoff_task` 可在交接时使用后续任务成功的折扣信用。

## 3. 训练与恢复技能专家

```bash
python -m mars.rltoken train-specialist --data pickup_windows.npz \
  --config configs/rltoken.yaml --output outputs/pickup.pt --updates 5000 --device cuda

python -m mars.rltoken train-specialist --resume outputs/pickup.pt \
  --config configs/rltoken.yaml --output outputs/pickup.pt --updates 1000 --device cuda
```

输出包括 `pickup.pt`、`pickup.replay.npz` 和最后更新的 `pickup.metrics.json`。
恢复时两个训练文件需配套存在；恢复 Actor/Critic、目标网络、优化器、更新计数和随机数状态。
`--updates` 是本次增加的更新次数。新训练的窗口数据可以来自多个 episode 的拼接；
Python API 可保留每条数据的采集阶段和 episode ID。

在线 RL 使用同一 learner，在环境采集之后追加新窗口，再执行更新：

```python
from mars.rltoken import PHASE_ONLINE, ReplayBuffer, SpecialistLearner, train_specialist

learner = SpecialistLearner(z_dim, q_dim, action_dim, specialist_cfg, device="cuda")
replay = ReplayBuffer(specialist_cfg.buffer_capacity, seed=0)
# 每次采集后：
replay.add_batch(windows, PHASE_ONLINE, episode_id=episode_id)
train_specialist(learner, replay, updates=100, output="outputs/pickup.pt")
```

执行时 `learner.act(z, q, ref, explore=True, rng=rng)` 加入固定标准差探索噪声。
Actor 输出完整动作块；reference 作为输入和训练约束。按原实现使用 Q1 优化 Actor、
双目标 Critic 的最小值估计 TD 目标，延迟更新 Actor 并进行 Polyak 更新。
离线缓存训练属于 warm-up；持续在线学习需要环境反馈和新的 rollout。

## 4. 接入 MARS 执行

先按现有流程使用 `adapter.encode_features(features)` 生成路由训练特征，
训练并保存 MARS。然后绑定每个候选专家：

```python
from mars import MARSController
from mars.rltoken import RoutedRLTPolicy, load_adapter, load_specialist

ctrl = MARSController.from_checkpoint("mars.pt", device="cuda")
adapter = load_adapter("outputs/adapter/adapter.pt", device="cuda")
policy = RoutedRLTPolicy(
    ctrl, adapter,
    reference=reference_from_frozen_vla,  # (features, q) -> 归一化 (H, action_dim)
    specialists={"pickup": load_specialist("outputs/pickup.pt", device="cuda"),
                 "place": load_specialist("outputs/place.pt", device="cuda")},
)
policy.reset()  # 每集开始时清空路由记忆、切换器和动作队列
result = policy.step(features, q_normalized)
env.step(action_unnormalize(result.action))
```

`features` 在流式部署时 batch=1；`reference` 回调只在队列需要补充时调用，可复用当前 VLA 的 KV
cache。MARS 按 `query_interval` 查询，即使队列仍有动作也会检查交接；一旦切换，旧队列立即清空，
同一步调用新控制器。reference 的 H 必须至少为所选专家的 C。可用 `IdentitySpecialist(C)` 先验证
整条调用链。所有 `z`、`q`、`ref`、`act` 均为训练时的归一化空间；动作反归一化和环境裁剪放在
外部环境适配层。PT 文件使用 PyTorch 对象序列化，只加载可信来源的检查点。

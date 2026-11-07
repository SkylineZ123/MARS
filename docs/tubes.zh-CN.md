# 试管转移长时序实验

这是独立的 robosuite 1.4 / MuJoCo Panda 工作台，用于 MARS 的长时序路由与 RL 专家实验。
RL 专家只负责"抓取试管"（π1）和"插入试管架"（π2）。接近、搬运和复位由冻结的 VLA 完成。

## 场景与协议

场景布局、相机、控制器、观测格式和机器人起始姿态尽量对齐 LIBERO，以便从 `HuggingFaceVLA/smolvla_libero` 微调：

- **场景与控制**：
  - 桌面高 0.90 m，尺寸 1.0 × 1.2 m；机器人基座在 (-0.66, 0, 0)；
  - agentview 使用 LIBERO 桌面场景的相机位姿；
  - 控制器为 robosuite 默认 OSC_POSE（kp 150，单步输出上限 5 cm / 0.5 rad），控制频率 20 Hz；
  - 起始关节角为 LIBERO MountedPanda 的初始值，其中第 7 关节转了 90°，使手指沿 x 方向开合。这样张开的手指不会碰到相邻孔里的试管。
- **观测**：`_obs()` 返回与 lerobot `LiberoEnv(obs_type="pixels_agent_pos")` 完全相同的原始结构：
  - 图像为 robosuite 的原始帧；
  - `robot_state` 包含 eef 的 pos/quat/mat、夹爪的 qpos/qvel、关节的 pos/vel。
  - 图像 180° 旋转和四元数→轴角由 lerobot 的 `LiberoProcessorStep` 完成；本仓库不包含该预处理。
- **试管**：胶囊体管身（r = 8 mm，圆底帮助入孔对中）加管帽（r = 9.5 mm），总长 110 mm；condim = 4，防止被夹住后转动。
- **两个试管架**：型号相同，3 个孔，间距 45 mm，方孔内宽 = 16 mm + `clearance`（默认 2 mm）。
  - 底板支撑试管，插到底时插入深度约 5.2 cm；管帽高出顶板约 5.8 cm。
  - 源架（橙色）固定在机器人右侧，中心 (-0.04, -0.17)；开局第 k 支试管在源架第 k 个孔中。
  - 目标架（绿色）的名义中心为 (-0.04, 0.13)。
  - 第 1 支（最左）取出后直接移向更左侧的目标架，后续试管都不会从已插好或未取出的试管上方经过。
- **随机化**：
  - 目标架 xy 各 ±4 cm、yaw ±15°；
  - 关节初值带种子的高斯噪声，σ = 0.02 rad；
  - 源架固定不动；
  - 种子为 `physical_seed(scene, seed)`，与执行顺序无关；
  - 同一场景连续 reset 得到的物理状态逐位相同。
- **划分**：沿用装配的划分，train 0–9999、val 10000–10049、calibration 20000–20049、test 30000–30099。只有一条指令（order 0），没有留出顺序。


| # | 阶段 | 判据 | 负责者 / 标签 kind |
|---|---|---|---|
| 1/3/5 | `grasp_k` | 双指夹住第 k 支试管（从左数），且管底高出源架顶板 ≥ 1.5 cm | π1 抓取 / `pickup` |
| 2/4/6 | `insert_k` | 管底低于目标架第 k 孔口 ≥ 4.5 cm；管体中心与孔中心的水平偏差 ≤ 孔内半宽 + 2 mm（默认 11 mm）；倾角 ≤ 15°；未被夹持；夹爪开度 ≥ 6.4 cm | π2 插入 / `insert` |
| 7 | `home` | 末端距起始位姿 < 4 cm，且 3 支试管都在位 | VLA |

上表按 `TubesEnv.seated` 的实际实现记录。几何成功判据的中心距离阈值与孔、管间的物理间隙是两个量；接触约束仍由 MuJoCo 执行。

**失败**：任意试管掉到桌下，或某支未被夹持的试管倾角 > 60° 且持续 40 步。超过 `max_steps = 1200` 步截断。
指令为："Move the three test tubes from the orange rack to the green rack one by one from left to right, inserting each into the matching hole, then return to the start position."（36 个 token，低于 smolvla_libero 的 48 上限）

**路由标签**（`TubesEnv.make_labeler`，用于 oracle 路由、专家训练窗口和路由器监督）：
- `pickup`：末端进入管帽抓取点 6 cm 以内，离开半径为 8 cm（滞回）；
- `insert`：试管进入目标孔水平 5 cm 以内，离开半径为 7 cm；
- 持管但离孔较远时为 `transport`，其余为 `approach`，都归 VLA；
- `SKILL_KINDS = {grasp: pickup, insert: insert}`。

## 代码与运行

- `mars/tubes/protocol.py`：`TubesSpec`、数据划分、`physical_seed`、`source_fingerprint`，纯 Python；
- `mars/tubes/environment.py`：`TubesEnv`、`TubesEnvPool`、`make_workcell`；
- `mars/tubes/scripted.py`：特权脚本示教控制器 `ScriptedTubes`（只用于训练 / 验收）；
- `mars/tubes/privileged.py`：`PrivilegedState`、`ObjectInfo`、`PhaseLabel`。

```bash
pip install -e .[tubes]
export NUMBA_DISABLE_JIT=1   # 必须在导入 robosuite 之前设置；TubesEnv 在 JIT 模式下会直接报错
MUJOCO_GL=egl pytest tests/test_tubes.py tests/test_tubes_protocol.py
```

```python
from mars.tubes.environment import TubesEnv
from mars.tubes.scripted import ScriptedTubes

env = TubesEnv(render=False)
env.reset(10000)
teacher = ScriptedTubes(env)
while True:
    _, success, info = env.step(teacher.act())
    if info["terminated"] or info["truncated"]:
        break
```

`env.snapshot()` / `env.restore_training_snapshot()` 保存和恢复数值状态向量、部分控制量与协议状态，仅用于训练入口课程。

重复入口实验需先 `env.sim.reset()` 再恢复快照，以初始化未包含在 `MjSimState` 中的求解器内部状态；恢复函数还会把 OSC 目标重设到当前位姿。这是规范化的课程入口，不保证等同于原轨迹中尚在执行的控制器状态。新实验使用 `mars.experiments.overnight_data.restore_entry`，并在训练前用重复动作验证一致性。详见 [恢复检查](specialist_diagnosis_20261030.zh-CN.md)。
重复性检查还需与真实轨迹的观测对齐检查分开：相同 qpos/qvel 向量经过 `sim.forward()` 或恢复后，派生末端观测仍可能改变。跨入口评估应同时检查原始观测、处理后状态、RL Token 与参考动作；仅验证状态向量不足以证明训练/部署入口等价。实际测量见 [后续结果](specialist_results_20261031.zh-CN.md)。

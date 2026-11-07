# 试管搬运仿真

安装和下载权重见[中文 README](../README.zh-CN.md)，训练数据接口见
[RL Token 指南](rltoken.zh-CN.md)，图形依赖见[环境说明](../environment/SIMULATION.md)。

## 运行入口

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
mars-assets verify --assets assets/tubes
mars-tubes run --assets assets/tubes --seed 0 --output outputs/tubes/run.json
```

预训练权重发布在 [XDEDM7/MARS](https://huggingface.co/XDEDM7/MARS)，包含一套
匹配的模型：`mars/mars.pt`、`actors/pickup.pt`、`actors/insert.pt`，以及 VLA
和共享适配器。只提供模型 seed 0。

资源根目录包含 `manifest.json` 和 `simulation.yaml`；路径相对于资源根目录解析。
默认使用资源包配置，也可用 `--config configs/tubes_simulation.yaml` 指定同布局配置。
依赖在源码仓库维护，权重仓库不包含源码归档、依赖锁或历史实验记录。

## 任务与物理协议

`tubes_v2` 是基于 robosuite 1.4 / MuJoCo 的 Panda 工作台，默认设置为：

| 项目 | 设置 |
|---|---|
| 任务 | 从左到右将三支试管移至目标架对应孔位，释放后返回起始位姿 |
| 控制 | OSC_POSE，20 Hz，每回合最多 1200 控制步 |
| 相机 | `agentview`、`robot0_eye_in_hand`，256 × 256 |
| 目标架随机化 | xy 各 ±4 cm，yaw ±15° |
| 初始关节噪声 | 高斯噪声，标准差 0.02 rad |
| 默认孔间隙 | 2 mm |

`TubesSpec` 定义协议及其 fingerprint。阶段为 `(grasp, insert)` 重复三次，
最后为 `home`；对应的专家类别为 `pickup` 和 `insert`。
只有一种搬运顺序，没有 held-out order。

| 划分 | 场景 ID |
|---|---|
| train | 0–9999 |
| val | 10000–10049 |
| calibration | 20000–20049 |
| test | 30000–30099 |

`reset(init_id, seed)` 决定物理随机化；VLA 去噪噪声由回合和控制步编号生成。
路由器接收 VLA 特征与归一化本体状态，不读取真值阶段。脚本控制器和标签器
使用仿真真值，仅用于辅助检查与数据生成。

## 可选评测

```bash
mars-tubes evaluate --assets assets/tubes --model-seeds 0 --modes mars \
  --output outputs/tubes/evaluation.json
```

默认测试网格为场景 30050–30099、噪声 seed 0/1，共 100 回合。
模型种子默认根据资源配置选择；显式指定未提供的模型种子会在加载前报错。
`--modes base base_c5 mars` 可比较执行 10 步和 5 步动作块的 VLA 与 MARS。
MARS 的 VLA 执行长度为 10，专家动作块长度为 5；比较时应保持测试网格一致。

输出包含 SR、SCR，以及所有回合和仅成功回合的平均控制步数及对应样本量。
SR 要求完成完整任务；SCR 按协议确认的插入释放事件去重，除以试管总数。
无成功回合时，仅成功回合的步数均值为 null。
`success_rate` 和 `scr` 使用 0–1 比例；控制步数不是墙钟耗时。
固定轨迹 Macro-F1 通过路由评价接口计算，不是闭环 CLI 的输出。

## 快照与模型资源

完整任务应从 `reset` 启动。`snapshot()` / `restore_training_snapshot()`
用于训练入口课程，只恢复部分物理与协议状态，并重设控制器目标；
它们不是完整仿真检查点，不保证任意历史下的逐步精确续跑。

四个 STL 及尺寸说明见[模型资源](../mars/tubes/assets/README.md)。单位为毫米，
导入米制场景时缩放 `0.001`；自带仿真直接使用程序生成的几何体。

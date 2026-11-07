# RL Token：数据与组件接口

本指南说明保留的缓存数据训练工具与部署接口。
预训练试管推理入口见[试管指南](tubes.zh-CN.md)；安装见[README](../README.zh-CN.md)。
这些组件不构成论文全流程训练复现，也不承诺重现历史结果。

## 配置与命令

`configs/rltoken.yaml` 是通用组件配置；`configs/rltoken_paper.yaml` 是面向
论文动作窗口约定的预设（专家 `C=5`、窗口起点采样步长 `window_stride=2`）。
兼容旧调用的默认 `window_stride=1`。采样步长控制训练窗口的起点间隔，
不改变环境控制频率、路由查询频率或部署动作块长度。
新增预设不会追溯改变历史训练、检查点或已记录结果。

以下命令需要用户已准备好的缓存，不负责采集图像、运行 VLA 或生成标注：

```bash
python -m mars.rltoken --help
python -m mars.rltoken train-adapter --data tokens.npz \
  --config configs/rltoken.yaml --output outputs/adapter --device cuda
python -m mars.rltoken build-windows --episode episode.pt \
  --controller-id 1 --skill pickup --config configs/rltoken_paper.yaml \
  --output outputs/pickup_windows.npz
python -m mars.rltoken train-specialist --data outputs/pickup_windows.npz \
  --config configs/rltoken_paper.yaml --output outputs/pickup.pt \
  --updates 5000 --device cuda
```

`--updates` 是本次追加的更新次数。恢复专家需要检查点及同名 `.replay.npz`：

```bash
python -m mars.rltoken train-specialist --resume outputs/pickup.pt \
  --config configs/rltoken_paper.yaml --output outputs/pickup.pt \
  --updates 1000 --device cuda
```

恢复时不能同时传 `--data`；新增在线数据通过 Python API 加入回放。
配置必须与数据、检查点的维度、动作长度和归一化一致。

## 适配器缓存

`train-adapter` 用 `numpy.load(..., allow_pickle=False)` 读取 NPZ：

| 字段 | 形状与约束 |
|---|---|
| `tokens` | `(N, slots, d_vla)`，冻结 VLA 的前缀特征 |
| `mask` | `(N, slots)`，有效 token 掩码 |
| `token_groups` | 按顺序排列的组名，例如 `image`、`image2`、`lang` |
| `group_sizes` | 各组固定槽位数，总和必须覆盖缓存 token 序列 |

`--val-data` 可指定独立验证缓存，组名及槽位数须与训练缓存相同。
不要把缓存 token 的填充长度与图像分辨率或动作块长度混为一谈。

在线 `features` 对象需提供 `hidden (B,N,d)`、`slices` 与 `group_masks`。
`features.slices[group]` 标识该组 token 区间；掩码须匹配该区间的批次和长度。
`RLTAdapter.encode_features` 按检查点组顺序提取、填充并编码。
超出组槽位的输入会报错；语言截断应由上游特征提取显式完成。

## EpisodeLog 与窗口

使用 `mars.rltoken.episode` 的保存/加载接口管理 episode，避免手写不完整日志。
EpisodeLog 由 `torch.save` 序列化（示例使用 `.pt`），不是窗口数据的 NPZ 格式。
对于执行了 T 次动作的回合：

- 动作索引长度 T：`exec_norm`、`exec_raw`、`owner`、`skill_owner`、`source`，
  以及动作执行后的 `success_after`、`skill_success_after` 等记录。
- 观测索引长度 T+1：`z`、`z_valid`、`q_norm`、阶段等；观测 t 对应动作 t 之前。
- VLA 查询索引：`vla_t` 与 `vla_chunks` 保存查询时间和完整归一化参考块。

`owner` 是实际执行动作的控制器，`skill_owner` 是该步所属专家区间；
专家演示纠正与 RL 自身执行通过 `source` 区分。
试管数据应明确包含 `pickup` 与 `insert` 成功谓词，不沿用默认 `place` 名称。
阶段、真值与仿真状态只用于数据构造和审计，不送入部署路由器。

窗口 NPZ 的必要字段为：

| 字段 | 批次维 N 后的形状 |
|---|---|
| `z`, `next_z` | `(d_z,)` |
| `q`, `next_q` | `(d_q,)` |
| `ref`, `act`, `next_ref` | `(C,d_a)` |
| `rew`, `valid` | `(C,)` |
| `done` | 标量 |

窗口不跨越控制权区间；成功、交接与预算结束的终止/自举含义不同。
`valid` 区分真实执行动作与补齐行；参考对齐、补齐及奖励模式由配置决定。
`build_windows` 可构造重叠窗口，起点间隔由 `window_stride` 决定。
改变它会改变有效训练样本集合，不能视为旧训练协议的等价替换。

历史 `ref_alignment: executed` 会逐行拼接日志中当时生效的 VLA 参考动作，
所以窗口后半段可能来自起点之后的查询；它并不保证整个参考块在窗口起点已经可知。
部署专家接收的是当前观测生成的参考块，二者应在新训练实验中单独核对。
本次保留历史对齐语义以兼容已有权重，新增 C=5/stride=2 预设未改变此约定。

## 部署与归一化

`RoutedRLTPolicy` 共享一个冻结适配器，接收路由器、VLA 参考动作回调与专家映射。
VLA 参考块与专家输出均处于训练归一化动作空间；环境适配层负责反归一化。
每个 episode 调用 `reset()`，切换控制器时清空旧动作队列。
候选顺序必须与路由器检查点一致；试管发布使用 `vla,pickup,insert`。

适配器、专家与路由器必须来自匹配的特征提取及归一化协议。
部署加载器可丢弃适配器解码器，专家恢复训练则需要完整训练状态与配套回放；
仅含 actor 的发布权重不应当作可恢复训练的检查点。
旧检查点格式的支持以加载器校验为准，不推断其论文来源。

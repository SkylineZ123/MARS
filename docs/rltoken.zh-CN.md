# RL Token：数据与组件接口

本指南说明保留的缓存数据训练工具与部署接口。
预训练试管推理入口见[试管指南](tubes.zh-CN.md)；安装见[README](../README.zh-CN.md)。

## 配置与命令

`configs/rltoken.yaml` 是通用组件配置；
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

```bash
python -m mars.rltoken train-specialist --resume outputs/pickup.pt \
  --config configs/rltoken_paper.yaml --output outputs/pickup.pt \
  --updates 1000 --device cuda
```

## 适配器缓存
`train-adapter` 用 `numpy.load(..., allow_pickle=False)` 读取 NPZ：

| 字段 | 形状与约束 |
|---|---|
| `tokens` | `(N, slots, d_vla)`，冻结 VLA 的前缀特征 |
| `mask` | `(N, slots)`，有效 token 掩码 |
| `token_groups` | 按顺序排列的组名，例如 `image`、`image2`、`lang` |
| `group_sizes` | 各组固定槽位数，总和必须覆盖缓存 token 序列 |

在线 `features` 对象需提供 `hidden (B,N,d)`、`slices` 与 `group_masks`。
`features.slices[group]` 标识该组 token 区间；掩码须匹配该区间的批次和长度。
`RLTAdapter.encode_features` 按检查点组顺序提取、填充并编码。
超出组槽位的输入会报错；语言截断应由上游特征提取显式完成。


## 部署与归一化

`RoutedRLTPolicy` 共享一个冻结适配器，接收路由器、VLA 参考动作回调与专家映射。
VLA 参考块与专家输出均处于训练归一化动作空间；环境适配层负责反归一化。
每个 episode 调用 `reset()`，切换控制器时清空旧动作队列。
候选顺序必须与路由器检查点一致；试管发布使用 `vla,pickup,insert`。

适配器、专家与路由器必须来自匹配的特征提取及归一化协议。
部署加载器可丢弃适配器解码器，专家恢复训练则需要完整训练状态与配套回放；
仅含 actor 的发布权重不应当作可恢复训练的检查点。
旧检查点格式的支持以加载器校验为准，不推断其论文来源。

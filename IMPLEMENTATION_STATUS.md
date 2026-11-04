# 跨路径直接 FIR 实现与验证状态

更新日期：2026-11-04。本文件描述当前主实验；正式协议以
[CROSS_PATH_EXPERIMENT.md](CROSS_PATH_EXPERIMENT.md) 为准。

## 当前实验定义

- 路径 1–7：训练，用于 8 类噪声网络优化。
- 路径 8–9：验证，用于检查点选择和调参。
- 路径 10：测试，只在模型与超参数冻结后运行一次。
- 每个场景/路径的 Wiener 先验只用该路径的外层训练时间段统计和校准。
- 验证/测试路径可提供各自训练时间段校准得到的先验，但不提供目标控制窗口标签。

本实验检验“同一批噪声录音、未见次级声学路径”的适应能力。它不支持“未见场景”
或“跨录音”泛化结论，因为 8 个场景仍然来自同一批原始噪声录音。

## 网络与损失

```text
参考噪声观察窗口 -> RoPE Encoder -> Q
场景/路径 Wiener 先验 -> 有序频率 token -> K/V
Cross-Attention Decoder -> 时间 token 注意力池化
Filter Head -> 完整归一化 1967 taps FIR
w_hat = RMS(w_prior) * generated_fir
```

最终输出不再采用 `w_prior + residual`。先验只通过交叉注意力和系数 RMS 定标
提供环境条件；所有 FIR 系数由网络生成。

训练目标是未来控制区间的声学残差均方：

```text
y = w_hat * x
anti_noise = s_path * y
e = d_path - anti_noise
loss = mean(e^2)
```

`d_path` 和 `s_path` 不进入网络输入；`s_path` 按批内每个样本的真实路径分别选择。

## 数据规模

```text
train:      8 scenes x 7 paths x 128 windows = 7168 episodes
validation: 8 scenes x 2 paths x  16 windows =  256 episodes
test:       8 scenes x 1 path  x  32 windows =  256 episodes
```

## 已完成验证

- `python -m pytest tests -q`：**9 passed**。
- 跨路径 Wiener bank 形状已验证为 `[8,10,1967]`，次级路径为 `[10,1967]`，
  所有系数有限。
- 训练、验证、测试加载器已检查路径互斥，并验证每个 episode 携带自己的
  `prior_filter` 和 `secondary_path`。
- 直接 FIR 模式已完成前向和反向检查；旧残差输出路径不再参与主实验。
- 真实数据的一轮 smoke 训练可正常优化；该结果只证明数据、损失和设备链路可用，
  不用于判断方法是否有效。

## Smoke 结果的范围

本地一轮 smoke 的验证集平均结果约为：

```text
Transformer mean primary-band NR = -0.920 dB
scene_wiener                    = 21.759 dB
pooled_wiener                   = 13.796 dB
mean_wiener                     = 17.274 dB
```

这只训练了 112 个 episode，且网络从随机 FIR 初始化，不能说明直接 FIR 方案失败。

## 正式运行顺序

```bash
source /home/zqbai/.venvs/anc-transformer/bin/activate
cd /home/zqbai/Project/ANC_Transformer

python -m pytest tests -q
python -m data.inspect_data --config config.cross_path.yaml
python -m wiener.build_bank --config config.cross_path.yaml
python -m experiments.train --config config.cross_path.smoke.yaml
python -m experiments.compare --config config.cross_path.smoke.yaml --split validation

CUDA_VISIBLE_DEVICES=0 \
python -m experiments.train --config config.cross_path.yaml \
2>&1 | tee cross_path_train.log

CUDA_VISIBLE_DEVICES=0 \
python -m experiments.compare --config config.cross_path.yaml --split validation
```

验证集完成且超参数冻结后才运行测试路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m experiments.compare --config config.cross_path.yaml --split test
```

## 产物

- `wiener/wiener_bank_cross_path.pt`：场景 × 路径 Wiener 先验和 10 条次级路径。
- `checkpoints/cross_path_direct_best.pt`：按验证集声学损失选择的检查点。
- `results/cross_path_direct/training_history.json`：逐轮训练记录。
- `results/cross_path_direct/comparison_validation_summary.csv`：验证路径 8–9 的四方法对照。
- `results/cross_path_direct/comparison_test_summary.csv`：冻结方案后的路径 10 一次性对照。

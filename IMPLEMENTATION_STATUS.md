# 实现与验证记录

固定路径 1；8 类噪声；48 kHz；每个场景 1967 taps；无自由残差项。

已完成全部核心模块：数据检查和时间划分、train-only Wiener 求解与 bank、
频谱时序 Encoder、带场景身份的 bank Query Decoder、Softmax 凸组合 head、
因果 ANC 仿真、训练与验证选模、四种方法的配对比较。
原始资料包和录音保持原位；新实现位于本目录。

## 已完成的实际检查

- `python -m pytest tests -q`：**7 passed**。
- 使用真实数据生成路径 1 的完整 `[8,1967]` 双精度 bank。
- 所有拟合读取和 ridge 校准区间均核实在对应录音的外层 train 内。
- 特征标准化只拟合 train；模型 bank 为冻结 buffer。
- 完成真实数据两轮小样本训练（每场景 4 个 train 窗口、2 个 validation 窗口）。
- 在同一批 16 个 validation episode 上完成四种方法的比较。
- 真实录音的外层 test 音频样本未用于本次调试；test 执行链已用合成数据验证。

代码中的 Q/K/V：8 个投影后的 Wiener FIR + 场景身份嵌入为 Q；
过去参考噪声的 Encoder memory 为 K/V；head 预测 8 个权重，
再乘**未经截断的原始 FIR bank**。输入不包含真实场景标签或目标扰动 d。

## 小样本贯通结果（validation，不能用于正式性能结论）

| 方法 | 平均主频带 NR / dB | 最大反弹 / dB |
|---|---:|---:|
| 场景 Wiener（给定真实场景） | 25.595 | 13.278 |
| 重新拟合的路径汇总 Wiener | 18.854 | 7.449 |
| 8 个 Wiener 的简单平均 | 21.413 | 4.876 |
| Transformer（仅两轮、32 个训练窗口） | 20.152 | 6.935 |

Transformer 此时的 validation 场景识别准确率为 12.5%；这些结果仅证明数据、
模型、仿真与评分链已贯通，不能说明模型已学好场景选择，也不能说明它优于简单平均。
场景 Wiener 的 NR 更高同时反弹更大，符合当前“按主频带 NR 选择 ridge”协议；
它不是兼顾所有评价指标的全局最优控制器。

## 原始长度的累计能量检查

| 场景 | 达到 95% 所需系数数 | 达到 99% 所需系数数 |
|---|---:|---:|
| KTV | 258 | 1964 |
| 公交 | 1966 | 1967 |
| 厨房 | 407 | 1727 |
| 地铁 | 1966 | 1967 |
| 步行街 | 11 | 240 |
| 火车 | 1478 | 1958 |
| 车载 | 1967 | 1967 |
| 餐厅 | 15 | 1916 |

不同场景差异较大，当前不能统一截断成同一个短 FIR。第一版保留 1967 taps。
完整累计能量数组保存在 `results/path_01/wiener_bank_export.npz`；
候选 ridge、范数、相似度与 train 内跨场景性能在 `wiener_diagnostics.json`。
两两场景 FIR 的余弦相似度范围约为 0.0623–0.9862，存在明显差异，也有非常相似的组合。

## 正式实验入口

使用 `config.yaml` 运行 `python -m experiments.train`，完成 30 轮训练并按 validation
选择 `checkpoints/best.pt`。确认配置后，运行 `python -m experiments.compare`，
一次性完成外层 test 的四方法比较。不要把 `smoke.pt` 当作正式实验结果。

环境：Python 3.12.15，CPU PyTorch 2.15.0，NumPy 2.5.4，SciPy 1.18.1，
SoundFile 0.14.0。完整版本列表见 `requirements-tested.txt`。

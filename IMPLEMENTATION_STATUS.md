# 实现与验证记录

固定路径 1；8 类噪声；48 kHz；每个场景 1967 taps；带可学习 FIR 残差。

已完成全部核心模块：数据检查和时间划分、train-only Wiener 求解与 bank、
RoPE 频谱时序 Encoder、逐样本宽带 Wiener 先验的频带 tokenizer、
参考噪声 Query/Wiener 先验 Key-Value 交叉注意力、完整 FIR 残差 head、
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

新版代码中的 Q/K/V：过去参考噪声经 RoPE Encoder 形成的 90 个时间 token 为 Q；
一个宽带 Wiener FIR 经 rFFT 形成 32 个频带 token，再投影为 K/V；
head 对适配后的时间 token 做注意力池化，预测 1967 tap 残差并加回先验。
每个样本输入自己的宽带 Wiener 先验；模型内部没有固定场景数，也不混合先验。
当前 8 场景数据集用样本标签映射到对应 bank 行，但这只是数据接口兼容桥；
扩展到更多噪声时，数据集应为每条记录提供自己的 `prior_filter`。
目标扰动 d 不进入网络前向。
训练主损失是未来控制区间的 `mean((d-s*(w_hat*x))²)`；`d` 和 `s` 不进入网络前向，
只用于可微声学仿真与评价。validation 按同一残差均方选择检查点。

## 小样本贯通结果（validation，不能用于正式性能结论）

| 方法 | 平均主频带 NR / dB | 最大反弹 / dB |
|---|---:|---:|
| 场景 Wiener（给定真实场景） | 25.595 | 13.278 |
| 重新拟合的路径汇总 Wiener | 18.854 | 7.449 |
| 8 个 Wiener 的简单平均 | 21.413 | 4.876 |
| Transformer（仅两轮、32 个训练窗口） | 20.152 | 6.935 |

新版 smoke 训练只运行 2 epoch，且是 32 个训练窗口的贯通检查，不能据此判断
模型性能或比较新旧架构。正式结果必须用完整 train 重新训练。
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

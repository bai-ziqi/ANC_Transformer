# 跨路径 Wiener 先验条件化 Transformer ANC

当前默认实验使用路径 1–7 训练、路径 8–9 验证、路径 10 测试。参考噪声 Encoder
产生 Query，每个样本自己的场景/路径宽带 Wiener FIR 产生 Key/Value，Decoder
直接生成适配当前观察窗口的完整 1967 tap FIR。详细协议、样本数和解释边界见
[CROSS_PATH_EXPERIMENT.md](CROSS_PATH_EXPERIMENT.md)。保留 48 kHz、1967 个系数，
不截断、不降采样、不在线更新 FxLMS。`config.yaml` 保留为单路径对照实验入口。
严格说，1967 **taps/系数** 对应 FIR 数学阶数 1966。

## 数据与目录

保持约定的 `data/`、`wiener/`、`models/`、`anc/`、`experiments/`、
`checkpoints/`、`results/` 和 `config.yaml`。另有 `utils.py`、`tests/` 和依赖说明。
`data/` 存放处理代码，原始录音默认直接引用 `../dataset_placeholder/dataset`，
不复制数十 GB 的文件。如迁移到 `data/NOISE`、`data/EXPECTED_NOISE`、`data/sh.npy`，
只需把 `data_root` 改为 `data`。所有配置相对路径以 YAML 所在目录为基准。

- `NOISE/<scene>.wav`：参考信号 x。
- `EXPECTED_NOISE/<scene>_scene_<path>.wav`：目标扰动 d，路径编号为 01–10。
- `sh.npy`：形状 `[1967,10]` 或 `[10,1967]`，选出指定的 s。
- 八类名称和 bank/标签/token 顺序统一以 `scene_names` 为准。

## 无测试泄漏的时间协议

每个原始录音按时间划分。默认跳过前 20 s 和末尾 1 s，扣除两个 2 s 隔离带后，
按 70% / 15% / 15% 得到外层 train / validation / test。x 和 d 共用绝对样本索引。
不同场景时长不同，因此每类录音独立计算边界。所有区间为 `[start,end)`。

外层 train 再划出末尾 5 s 的 **内部校准段**，前留 2 s 间隔：

```text
| 跳过 | train: Wiener统计拟合区 ... 间隔 ... 内部校准区 | 间隔 | validation | 间隔 | test | 保护 |
```

8 个场景 Wiener 的相关统计量、岭系数选择、路径汇总基线和特征标准化，
**全部只访问外层 train**。默认每场景从拟合区均匀取 3 段、共 60 s；
候选 ridge 只在内部 train 校准区选取。外层 validation 只选择 Transformer 检查点。
test 不参与任何统计、拟合或选模。所有读取，包括卷积历史，都不得越过划分边界。

`data_manifest.json` 和 bank 中保留来源、样本边界和实际拟合读取区间。
音频用绝对路径、长度、大小、mtime 作一致性指纹；`sh.npy` 用 SHA-256。
音频指纹不等同于原始文件的完整内容哈希。文件内容/划分变化后须重建 bank。

## Transformer 输入输出

```text
过去 0.5 s 的参考噪声 x：[B,24000]
  → STFT(center=False) → log功率谱 → 每帧129维（频率轴池化，不改音频）
  → 仅train估计的标准化 → Linear → 2层 RoPE Encoder
  → Query memory：[B,90,64]  （默认参数）

每个样本对应的宽带 Wiener 先验 w_prior：[B,1967]
  → rFFT(实部/虚部)
  → 32 个有序频带 token → Linear(2→64)
  → Key/Value：[B,32,64]

Cross-Attention：Q=参考噪声时间 token，K/V=Wiener 频带 token
  → 2层 prior-conditioned Decoder
  → 时间 token 注意力池化 → [B,64]
  → Filter Head 直接生成完整归一化 FIR，再按先验 RMS 定标 → w_hat：[B,1967]
```

RoPE 只旋转时间自注意力中的 Q/K，不修改 V，也不向 token embedding
添加正弦位置向量。Decoder 定义 Q 来自参考噪声、K/V 来自 Wiener 频带表示，
因此这里的交叉注意力方向与标准 Encoder–Decoder Transformer 相反。

`model.prior_mode: sample` 要求每条训练或推理记录提供自己的 1967 tap 先验，
调用形式是 `model(observation, prior_filter=priors)`，其中 `priors` 可为
`[taps]` 或 `[B,taps]`。模型内部没有 8 类固定 embedding，也不会跨样本平均
或混合 Wiener 解；因此 batch 中有多少种噪声，就传入多少条各自的先验。

当前 8 场景数据由 `scene_id` 找到该行对应的 bank 行，仅作为数据接口兼容桥；
模型前向只接收已经解析好的逐样本 `prior_filter`。若扩展为多个噪声录音或
多个训练样本，应由数据集为每条记录返回自己的 `prior_filter`。

Wiener bank 只保留为当前数据的先验来源和评价基线；网络不读取整个 bank，也不把
输出限制为 8 条 FIR 的凸组合。最终 head 直接生成完整 FIR，不采用 `先验 + 残差`。

设置 `model.architecture: legacy_bank_fusion` 可复现旧版“8 候选 Q + 声音 memory K/V
+ 凸组合”模型。不同架构的检查点不兼容，应重新训练。
检查点格式还绑定训练目标和损失权重；FIR-loss 时代的旧检查点不能用于当前的
声学残差实验，必须重新训练。

## 训练和物理评价

默认训练损失直接是未来计分区间的声学残差均方：

```text
y = w_hat * x
anti_noise = s * y
e = d - anti_noise
loss = mean(e²)
```

该损失保留完整的因果启动协议，可经 PyTorch 卷积和模型反向传播。`d`、`s`
只用于训练目标和评价，不作为网络输入。相对 FIR 误差仍作为诊断指标记录，
但默认不参与梯度。validation 按同一声学残差损失选择检查点。
可通过 `training.loss_mode` 选择 `acoustic`、`filter` 或 `combined`。
选模依据是外层 validation 的相同损失，另报告所选模型的声学验证指标。
默认 30 epoch、batch 16、AdamW、小型 64 维 Transformer。

每个独立评价 episode：先观察过去 0.5 s 的 x，冻结预测滤波器，
然后控制并计分**紧接着的未来 0.5 s**：

```text
y = w_hat * x
anti_noise = s * y
e = d - anti_noise
```

控制开始时 FIR 可以读取已观测的 1966 点 x 历史，但次级路径状态从零启动；
不能用尚未选择的滤波器虚构历史扬声器输出。未来计分区的启动样本全部计分，
不丢弃瞬态。各方法使用相同 observation/control 起点与相同启动状态。
validation/test episode 不重叠；这是独立离线 episode，未实现连续切换或实时系统。

主比较包括：

1. `transformer`：根据过去 x 预测并融合 FIR。
2. `scene_wiener`：给定真实场景标签，使用该场景的 train-only FIR。
3. `pooled_wiener`：8类 train 统计汇总后**重新求解**的路径级基线，仅供对照；
   它不会作为 Transformer 的默认输入。
4. `mean_wiener`：8个场景 FIR 的简单算术平均。

不直接使用旧资料包的路径级滤波器作主基线：旧拟合区间可能跨过新的外层划分。
场景 Wiener 是条件参照，**并非分频带 NR 的无条件上界**。
评价保持原项目的口径：50–5000 Hz 三分之一倍频程等权平均 NR、宽带 NR、
1–8 kHz 最大反弹、平均正反弹、控制 RMS/峰值与能量比。

跨路径实验中，若系统近似 LTI 且最优逆滤波器可表示，不同噪声的最优 FIR 可能非常相近。
先检查 `wiener_diagnostics.json` 的余弦相似度和内部train跨场景性能矩阵，
再判断场景适应是否有收益；不预设 Transformer 必定优于汇总 Wiener。
同时输出每个 FIR 达到 95%/99% 累计能量所需 taps，第一版保持完整长度。

## 运行

在 `ANC_Transformer` 目录中运行以下命令。先安装 Python 3.10+ 和依赖。
CPU 环境可先从 PyTorch 官方 CPU 索引安装 torch，再安装其余 requirements。

本工作区已创建可直接使用的隔离环境 `../.venv-anc`。当前 Windows 无需先配置
`python` 到 PATH，也不必激活环境，例如：

```powershell
cd F:\比赛\expert_training_ppt_materials\ANC_Transformer
$ancPython = "..\.venv-anc\Scripts\python.exe"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
& $ancPython -m data.inspect_data --config config.cross_path.yaml
& $ancPython -m wiener.build_bank --config config.cross_path.yaml
& $ancPython -m experiments.train --config config.cross_path.yaml
& $ancPython -m experiments.compare --config config.cross_path.yaml --split validation
```

当前已用正式配置生成 `wiener/wiener_bank.pt`。`checkpoints/smoke.pt` 来自
`config.smoke.yaml` 的两轮小样本贯通检查，**不是正式训练检查点**；
正式配置仍为 30 轮，需训练得到 `checkpoints/best.pt` 后再做最终 test。
本次真实数据调试仅访问 train/validation，没有读取外层 test 音频样本。
验证过程和诊断见 `IMPLEMENTATION_STATUS.md`。环境实测版本见 `requirements-tested.txt`。

```powershell
python -m pip install -r requirements.txt
python -m data.inspect_data --config config.cross_path.yaml
python -m wiener.build_bank --config config.cross_path.yaml
python -m experiments.train --config config.cross_path.yaml
# 验证集用于选模和调参；冻结方案后才运行 test。
python -m experiments.compare --config config.cross_path.yaml --split validation
```

`python -m experiments.test` 可单独测试 Transformer；与 compare 使用相同 protocol。
不要根据 test 结果反复修改参数、重选模型。调试比较使用
`python -m experiments.compare --split validation`。

主要产物：

- `wiener/wiener_bank.pt`：8×1967 双精度 FIR、汇总 FIR、s、normalizer、全部来源。
- `checkpoints/best.pt`：选定 Transformer、配置、bank SHA-256 和 epoch。
- `results/path_01/data_manifest.json`：实际数据与时间划分。
- `wiener_diagnostics.json` / `wiener_bank_export.npz`：能量、相似度、候选性能和系数。
- `training_history.json`、验证结果和 `comparison_test_*.csv/json`：学习过程与分场景结果。

```powershell
python -m pytest tests -q
```

测试覆盖 train-only bank 读取、时间隔离、Wiener 互相关方向、物理卷积/启动状态、
已知增益的评分口径、RoPE/宽带先验频带 token、完整训练-加载-四方法配对评估。
测试使用合成数据，保留真实的 48 kHz 和 1967 taps。

快速复查本次贯通流程（使用已有 bank；仍只比较 validation）：

```powershell
& $ancPython -m experiments.train --config config.cross_path.smoke.yaml
& $ancPython -m experiments.compare --config config.cross_path.smoke.yaml --split validation
& $ancPython -m pytest tests -q
```

当前结论范围是**8个已知场景、同录音的时间隔离测试、一个已知路径**。
大量窗口不是大量独立录音，不据此宣称未知场景泛化或真实在线稳定性。
`anc/fxlms.py` 仅为后续阶段预留，第一版不调用。

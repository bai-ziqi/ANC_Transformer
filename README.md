# 固定路径的八场景 Wiener Bank + Transformer ANC（第一版）

固定一条已知次级路径；从参考噪声预测 8 个场景 Wiener FIR 的融合权重。
默认 `path_id: 1`，可在 `config.yaml` 修改。保留 48 kHz、1967 个系数，
不截断、不降采样、不做频带分解、不增加自由 FIR 残差、不在线更新 FxLMS。
严格说，1967 **taps/系数** 对应 FIR 数学阶数 1966。

## 数据与目录

保持约定的 `data/`、`wiener/`、`models/`、`anc/`、`experiments/`、
`checkpoints/`、`results/` 和 `config.yaml`。另有 `utils.py`、`tests/` 和依赖说明。
`data/` 存放处理代码，原始录音默认直接引用 `../dataset_placeholder/dataset`，
不复制数十 GB 的文件。如迁移到 `data/NOISE`、`data/EXPECTED_NOISE`、`data/sh.npy`，
只需把 `data_root` 改为 `data`。所有配置相对路径以 YAML 所在目录为基准。

- `NOISE/<scene>.wav`：参考信号 x。
- `EXPECTED_NOISE/<scene>_scene_01.wav`：固定路径的目标扰动 d。
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
  → 仅train估计的标准化 → Linear → 时间位置编码
  → 2层 Transformer Encoder
  → memory：[B,90,64]  （默认参数）

固定 Wiener Bank：[8,1967]，不参与梯度更新
  → 共享缩放 + Linear(1967→64) + 场景身份Embedding
  → 8个候选 Query：[B,8,64]
  → 2层 Transformer Decoder：Q=候选，K/V=Encoder memory
  → [B,8,64]
  → 每token共享Linear(64→1) → [B,8] logits
  → Softmax → alpha：[B,8]
  → alpha @ 原始Wiener Bank → w_hat：[B,1967]
```

Decoder 含候选 token 的 self-attention 和单向 cross-attention，
不是自回归文本解码，因此无三角因果 mask。Encoder 只能看到已观测窗口，
不包含后续计分数据；不接收 d、路径标签以外的新环境信息或真实噪声类别。
场景身份 embedding 标识每一个**候选**，不泄漏当前样本的真实场景。
`scene_id` 仅用于监督和带场景标签的参照基线。

权重满足 alpha>=0、sum(alpha)=1，输出被限制在 8 个 Wiener FIR 的凸包内。
这是选择/融合实验；不是无约束系数生成。第一版不输出 Delta w。

## 训练和物理评价

训练损失为 CE + 0.1 × 相对 FIR 平方误差，后者用目标 FIR 能量归一化，
避免不同场景系数量级直接主导误差。标签和目标 FIR 都来自 train 构建的 bank。
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
3. `pooled_wiener`：8类 train 统计汇总后**重新求解**的路径级 FIR。
4. `mean_wiener`：8个场景 FIR 的简单算术平均。

不直接使用旧资料包的路径级滤波器作主基线：旧拟合区间可能跨过新的外层划分。
场景 Wiener 是条件参照，**并非分频带 NR 的无条件上界**。
评价保持原项目的口径：50–5000 Hz 三分之一倍频程等权平均 NR、宽带 NR、
1–8 kHz 最大反弹、平均正反弹、控制 RMS/峰值与能量比。

固定路径下，若系统近似 LTI 且最优逆滤波器可表示，不同噪声的最优 FIR 可能非常相近。
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
& $ancPython -m data.inspect_data --config config.yaml
& $ancPython -m experiments.train --config config.yaml
& $ancPython -m experiments.compare --config config.yaml
```

当前已用正式配置生成 `wiener/wiener_bank.pt`。`checkpoints/smoke.pt` 来自
`config.smoke.yaml` 的两轮小样本贯通检查，**不是正式训练检查点**；
正式配置仍为 30 轮，需训练得到 `checkpoints/best.pt` 后再做最终 test。
本次真实数据调试仅访问 train/validation，没有读取外层 test 音频样本。
验证过程和诊断见 `IMPLEMENTATION_STATUS.md`。环境实测版本见 `requirements-tested.txt`。

```powershell
python -m pip install -r requirements.txt
python -m data.inspect_data --config config.yaml
python -m wiener.build_bank --config config.yaml
python -m experiments.train --config config.yaml
# 完成选模后运行一次完整对照；该命令同时测试四种方法。
python -m experiments.compare --config config.yaml
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
已知增益的评分口径、冻结 bank/凸组合、完整训练-加载-四方法配对评估。
测试使用合成数据，保留真实的 48 kHz 和 1967 taps。

快速复查本次贯通流程（使用已有 bank；仍只比较 validation）：

```powershell
& $ancPython -m experiments.train --config config.smoke.yaml
& $ancPython -m experiments.compare --config config.smoke.yaml --split validation
& $ancPython -m pytest tests -q
```

当前结论范围是**8个已知场景、同录音的时间隔离测试、一个已知路径**。
大量窗口不是大量独立录音，不据此宣称未知场景泛化或真实在线稳定性。
`anc/fxlms.py` 仅为后续阶段预留，第一版不调用。

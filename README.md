# ANC Transformer：SFANC 噪声与环境 Wiener 先验

当前实验使用 16 kHz SFANC WAV 噪声语料（训练 80000、验证 2000、测试 2000 条），
主路径 512 点、次级路径 256 点，输出控制 FIR 为 1024 taps（1023 阶）。
每个环境使用独立宽带校准信号计算一条 `w_prior`；每条噪声在对应环境中生成自己的
`w_target`。网络根据噪声和环境先验直接生成完整 FIR，不输入场景编号或固定子带编号。

当前配置只有 `path_1`，检验的是同一环境下的未见噪声适配；跨路径泛化需要新增环境
并使用互不重叠的路径划分。当前整段观察、整段评分是离线实验。

## 数据与模型

```text
WAV x ── LogSpectrum ── RoPE Encoder ── observation query
独立环境 w_prior ── prior tokenizer ── cross attention ── Filter Head ── w_hat

离线标签：d=P*x，z=S*x；用完整片段纯有限窗口最小二乘求解 w_target
训练损失：归一化 FIR 误差；评估残差：e=d-S*(w_hat*x)
```

PP/SP 只用于标签构造与声学仿真，不作为额外网络输入。
默认 1 秒输入形状为 `[B,16000]`，先验和输出均为 `[B,1024]`。
逐样本标签是正则化候选网格中实际残差 MSE 最小的解，不宣称精确有限窗口全局最优。

## 运行

在项目根目录、已激活的 Python 环境中执行：

```bash
python -m pip install -r requirements.txt
python -m models.inspect_model --config config.sfanc.yaml
python -m data.sfanc --config config.sfanc.smoke.yaml --limit 8
python -m wiener.build_priors --config config.sfanc.smoke.yaml
python -m wiener.build_targets --config config.sfanc.smoke.yaml --limit 8 --resume
CUDA_VISIBLE_DEVICES=0 python -m experiments.train --config config.sfanc.smoke.yaml
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare --config config.sfanc.smoke.yaml --split validation
python -m wiener.diagnose_labels --config config.sfanc.smoke.yaml --count 2
```

正式运行将配置换成 `config.sfanc.yaml` 并移除 `--limit`。
最终确定模型和参数后才运行 `--split test`；测试结果不参与 epoch 选择。
详细数据要求、服务器路径、最小二乘标签求解与缓存续算见 [SFANC_WIENER_WORKFLOW.md](SFANC_WIENER_WORKFLOW.md)。

## 目录说明

```text
ANC_Transformer/
├── config.sfanc.yaml              # 正式实验：数据、路径、先验、模型及训练参数。
├── config.sfanc.smoke.yaml        # 少量样本检查，使用独立输出位置。
├── utils.py                      # 配置解析、随机种子、设备与文件签名。
├── requirements.txt              # 项目依赖与版本范围。
├── requirements-tested.txt       # 本地验证过的依赖快照，勿覆盖服务器 CUDA 环境。
├── README.md                     # 当前实验概览、运行入口与目录索引。
├── SFANC_WIENER_WORKFLOW.md       # 数据处理、Wiener 生成和训练的详细说明。
├── .gitignore                    # 排除缓存、标签、权重和运行结果。
├── data/
│   ├── environments.py           # 读取并校验 PP/SP MAT 路径与采样率信息。
│   ├── sfanc.py                  # 扫描 WAV、生成清单、配对环境并提供 Dataset。
│   ├── sfanc_config.py           # 校验新实验配置及各模块的尺寸契约。
│   └── preprocess.py             # 将噪声转换为归一化对数频谱。
├── wiener/
│   ├── calculate_wiener.py       # 计算相关统计、求解 Wiener 候选并评分。
│   ├── build_priors.py           # 独立宽带校准得到每个环境的一条先验。
│   ├── build_targets.py          # 生成并验证逐样本纯有限窗口最小二乘 FIR 标签。
│   ├── finite_window_ls.py       # 矩阵算子和带收敛门槛的纯 LSMR 标签求解核心。
│   └── diagnose_labels.py        # 只读比较缓存标签、先验和有限窗口候选/精确解。
├── models/
│   ├── anc_transformer.py        # 组装网络并校验波形、先验及输出尺寸。
│   ├── encoder.py                # 编码噪声时频特征并形成查询。
│   ├── decoder.py                # 将先验编码为 token 并与噪声交叉注意力融合。
│   ├── filter_head.py            # 将融合特征回归为完整 FIR。
│   ├── rope.py                   # 提供作用于 Q/K 的旋转位置编码与注意力。
│   └── inspect_model.py          # 用模拟张量检查网络结构、形状和参数量。
├── anc/
│   ├── anc_simulator.py          # 以因果卷积计算控制信号和误差残差。
│   └── metrics.py                # 计算 NR、残差能量和频带回升指标。
├── experiments/
│   ├── common.py                 # 共享数据加载、损失、模型加载与评估逻辑。
│   ├── train.py                  # 训练并从 epoch 0 开始选择验证最优权重。
│   ├── compare.py                # 比较零滤波器、环境先验、样本 Wiener 和模型。
│   └── test.py                   # 加载已选权重评估最终测试集。
├── tests/
│   ├── test_sfanc.py             # 验证数据、先验、标签和训练流程。
│   ├── test_model_entry.py       # 验证模型入口尺寸、类型和设备处理。
│   └── test_signal_processing.py # 验证 Wiener、仿真、频谱与注意力计算。
├── checkpoints/                  # 运行时保存模型权重。
└── results/                      # 运行时保存清单、指标和对照结果。
```

各 Python 包的 `__init__.py` 用于标识包；生成的先验和标签位于配置指定的
`wiener/sfanc_*` 文件或目录中，不属于源代码。

## 检查

```bash
python -m pytest tests -q
```

本仓库不保存服务器 WAV 数据与原始路径文件。MAT 中的路径采样率需由来源确认，
配置中的 16 kHz 声明不能替代实际确认。

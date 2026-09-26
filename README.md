# UQGNN

**Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction**

[![Venue](https://img.shields.io/badge/ACM%20SIGSPATIAL-2025-blue)](https://dl.acm.org/doi/10.1145/3748636.3762709)
[![arXiv](https://img.shields.io/badge/arXiv-2508.08551-b31b1b)](https://arxiv.org/abs/2508.08551)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3748636.3762709-orange)](https://doi.org/10.1145/3748636.3762709)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Spatiotemporal prediction usually returns a single number per region and time step, which says nothing about how much that number can be trusted. UQGNN predicts a **full multivariate Gaussian** for every region and forecast step, jointly over the *M* interacting urban phenomena (e.g. taxi / TNP / bike demand):

```
p(y_n | X) = N( mu_n , Sigma_n ),    mu_n in R^M,   Sigma_n in R^{MxM}  (positive definite)
```

so accuracy and uncertainty come out of the same forward pass, and the off-diagonal terms of `Sigma_n` capture how the phenomena co-vary at that location.

## Method

| Component | What it does |
| --- | --- |
| **ISTE** — Interaction-aware SpatioTemporal Embedding | A spatial branch (**MDGCN**, multivariate diffusion graph convolution over the forward/reverse random-walk supports `A_q`, `A_h`) and a temporal branch (**ITCN**, interaction-aware temporal convolution with a learnable inter-variable embedding) each produce an embedding of shape `(B, N, M, e)`; the two are fused by a Hadamard product `E = E_s ⊙ E_t`. |
| **MPP** — Multivariate Probabilistic Prediction | Two heads map the fused embedding to the mean `mu` and to a lower-triangular vector that is assembled into a positive-definite covariance `Sigma` (Algorithm 1: symmetric fill → eigendecomposition → eigenvalue clamping at `min_vec` → reconstruction). |
| **Objective** | Multivariate-Gaussian negative log-likelihood (`MGAU` in `base/metrics.py`). |

`forward(X)` takes `(B, seq_len, N, M)` and returns `(mu, Sigma)`; the engine consumes the second element as the per-prediction covariance. Because the model emits its own distribution rather than a quantile triple, it sets `cqr_compatible = False` and the `--cqr` flag is rejected with a clear message.

Reported metrics: `MGAU`, `MAE`, `MAPE`, `RMSE`, `CRPS`, `KL` — all computed per forecast horizon in the original (inverse-transformed) data space.

## Repository layout

```
UQGNN/
  src/flow/uqgnn/
    uqgnn_model.py      MDGCN + ITCN + covariance assembly (the UQGNN model)
    main.py             Entry point: model args, adjacency setup, run_experiment()
  base/                 Shared framework
    runner.py           run_experiment(): the single experiment driver
    model.py            BaseModel contract
    engine.py           Training / validation / test loop, checkpointing, export
    CQR_engine.py       Conformalized Quantile Regression engine
    metrics.py          MAE / RMSE / MAPE / CRPS / KL / MGAU / interval metrics
    efficiency.py       Hardware info, memory, inference time, FLOPs
  utils/
    args.py             Common CLI arguments, path config, set_seed
    dataloader.py       Dataset / DataLoader, dataset registry lookup
    generate.py         Raw array -> his.npz / info.json / split indices
    registry.yaml       Dataset name -> data & adjacency paths
    graph_algo.py       Adjacency normalizations (sym / transition / Chebyshev / Laplacian)
    get_adj_mat.py      Build an adjacency matrix from geographic shapefiles
    log.py              Logger
    res.py              Result collection / comparison CLI
  jobs/train.sh         Slurm submission script
```

## Installation

```bash
conda create -n st python=3.10 -y
conda activate st

# PyTorch (CUDA 12.8 build)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Data

Point the framework at your data root (defaults to `<repo>/datasets`):

```bash
export POPST_DATA=/path/to/datasets      # where datasets live
export POPST_RESULT=/path/to/result      # where logs & checkpoints go
```

Each dataset folder has this layout, and every entry is produced by `utils/generate.py`:

```
<POPST_DATA>/<dataset>/
  <adj_name>.npy        Adjacency matrix (N x N)
  <years>/
    his.npz             Normalized data + scaler parameters
    info.json           Shape, scaler, split sizes, seq_length_x / seq_length_y
    meta.json           Scaler parameters and raw data shape
    idx_train.npy       Training sample indices
    idx_val.npy         Validation sample indices
    idx_test.npy        Test sample indices
    idx_all.npy         All sample indices
```

Generate it from a raw array of shape `(T, N, M)`:

```bash
python utils/generate.py --data_path /path/to/raw.npy --dataset chicago_15min --years 2018 --fmt NDT
```

Supported input layouts: `NDT` (T×N×D), `NTD` (N×T×D), `NT` (N×T, D=1 appended). Then register the dataset in `utils/registry.yaml`:

```yaml
chicago_15min:
  data: chicago_15min
  adj: chicago_15min/chicago.npy
```

The node count `N` is read from `info.json` at runtime, and `seq_len` / `horizon` / `input_dim` / `output_dim` are auto-filled from the same file unless given on the command line.

## Usage

```bash
# Train
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018

# Test from a checkpoint
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018 \
    --mode test --model_path /path/to/UQGNN_<timestamp>.pt

# Test and export prediction archives
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018 --mode test --export

# Group results under result/<proj>/
python src/flow/uqgnn/main.py --dataset chicago_15min --proj MyExperiment
```

Slurm:

```bash
sbatch jobs/train.sh
DATASETS="chicago_15min nyc_manhattan_15min" YEARS=2018 sbatch jobs/train.sh
```

Compare runs:

```bash
python utils/res.py --path result/MyExperiment
python utils/res.py --path result/MyExperiment --select RMSE
python utils/res.py --log result/MyExperiment/UQGNN/chicago_15min/<timestamp>.log
```

Results land in `result/<proj>/UQGNN/<dataset>/<timestamp>.log` next to `UQGNN_<timestamp>.pt`.

## Arguments

### Model (UQGNN)

| Argument | Default | Description |
| --- | --- | --- |
| `--hidden_dim_s` | `64` | Hidden width of the spatial (MDGCN) branch |
| `--hidden_dim_t` | `64` | Hidden width of the temporal (ITCN) branch |
| `--emb_dim` | `32` | Interaction-aware embedding dimension `e` |
| `--kernel_size` | `3` | Temporal convolution kernel size |
| `--temporal_layers` | `2` | Number of ITCN layers |
| `--min_vec` | `1e-6` | Eigenvalue floor used when clamping `Sigma` to positive definite |

### Training

| Argument | Default | Description |
| --- | --- | --- |
| `--bs` | `64` | Batch size |
| `--max_epochs` | `2000` | Maximum epochs |
| `--patience` | `30` | Early-stopping patience on validation loss |
| `--lrate` | `1e-3` | Learning rate (Adam) |
| `--wdecay` | `5e-4` | Weight decay |
| `--dropout` | `0.5` | Dropout |
| `--clip_grad_norm` | `5` | Gradient-norm clipping |
| `--step_size` | `200` | StepLR decay interval |
| `--gamma` | `0.95` | StepLR decay factor |
| `--seed` | `2025` | Random seed |

### Data & system

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `chicago_15min` | Dataset name (must exist in `registry.yaml`) |
| `--years` | `2018` | Data sub-folder |
| `--seq_len` / `--horizon` | auto | Input length / forecast steps, auto-filled from `info.json` |
| `--input_dim` / `--output_dim` | auto | Number of variables `M`, auto-filled from `info.json` |
| `--no_normalize` | -- | Disable MinMax normalization (on by default) |
| `--device` | `cuda` | Device |
| `--mode` | `train` | `train` or `test` |
| `--model_path` | -- | Checkpoint to load in test mode |
| `--export` | off | Save prediction archives with the final evaluation |
| `--proj` | -- | Sub-folder name for grouping results |

## Implementation details

Experiments were run on a Linux server (Intel Xeon, 64 GB RAM) with an NVIDIA A100 GPU. The reference results in the paper use PyTorch 2.3.0 / CUDA 11.8; this release is pinned to PyTorch 2.8.0 / CUDA 12.8.

## Baselines

Deterministic baselines follow
[STGCN](https://github.com/hazdzz/STGCN),
[DCRNN](https://github.com/chnsh/DCRNN_PyTorch),
[GWNET](https://github.com/nnzhan/Graph-WaveNet),
[StemGNN](https://github.com/microsoft/StemGNN),
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN),
[AGCRN](https://github.com/LeiBAI/AGCRN), and
[SUMformer](https://github.com/Chengyui/SUMformer).

Probabilistic baselines follow
[TimeGrad](https://github.com/zalandoresearch/pytorch-ts),
[STZINB](https://github.com/ZhuangDingyi/STZINB),
[DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch),
[CF-GNN](https://github.com/snap-stanford/conformalized-gnn), and
[DiffSTG](https://github.com/wenhaomin/DiffSTG).

Runnable implementations of all of them, on the same runner and metric pipeline used here, are available in [POPST](https://github.com/UFOdestiny/POPST).

## Citation

```bibtex
@inproceedings{yu2025uqgnn,
  title     = {UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction},
  author    = {Yu, Dahai and Zhuang, Dingyi and Jiang, Lin and Xu, Rongchao and Ye, Xinyue and Bu, Yuheng and Wang, Shenhao and Wang, Guang},
  booktitle = {Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems},
  pages     = {52--65},
  year      = {2025},
  doi       = {10.1145/3748636.3762709}
}
```

## Related work

- [POPST](https://github.com/UFOdestiny/POPST) — the unified spatiotemporal benchmarking framework this release is extracted from (~30 flow models, ~17 OD models, shared conformal-prediction engines)
- [EnergyMamba](https://github.com/UFOdestiny/EnergyMamba) (KDD 2026) — graph-enhanced selective state space model with adaptive sequential CQR
- [TrustEnergy](https://github.com/UFOdestiny/TrustEnergy) (AAAI 2026) — memory-augmented spatiotemporal GNN with sequential CQR
- [HealthMamba](https://github.com/UFOdestiny/HealthMamba) (IJCAI 2026) — graph state space model with three-mechanism uncertainty quantification

## License

Released under the [MIT License](LICENSE).

---

# 中文说明

## UQGNN：用于多变量时空预测的图神经网络不确定性量化

论文发表于 ACM SIGSPATIAL 2025：[论文页面](https://dl.acm.org/doi/10.1145/3748636.3762709) · [arXiv](https://arxiv.org/abs/2508.08551) · [DOI](https://doi.org/10.1145/3748636.3762709)。

时空预测通常只为每个区域、每个时间步输出一个数值，无法说明这个数值有多可信。UQGNN 为每个区域和预测时间步输出一个**完整的多元高斯分布**，联合描述 *M* 种相互影响的城市现象，例如出租车、网约车（TNP）和单车需求：

```text
p(y_n | X) = N(mu_n, Sigma_n)
mu_n 属于 R^M，Sigma_n 属于 R^{M×M}，且 Sigma_n 为正定矩阵
```

因此，一次前向传播即可同时得到预测值及其不确定性；`Sigma_n` 的非对角元素刻画了该位置不同现象之间的协同变化关系。

## 方法

| 组件 | 作用 |
| --- | --- |
| **ISTE：交互感知时空嵌入** | 空间分支 **MDGCN** 在正向和反向随机游走支撑矩阵 `A_q`、`A_h` 上进行多变量扩散图卷积；时间分支 **ITCN** 使用交互感知时间卷积和可学习的变量间嵌入。两个分支分别输出形状为 `(B, N, M, e)` 的嵌入，再通过逐元素乘积融合：`E = E_s ⊙ E_t`。 |
| **MPP：多变量概率预测** | 两个输出头分别将融合后的嵌入映射为均值 `mu` 和下三角元素向量，后者用于构造正定协方差矩阵 `Sigma`。算法 1 的流程为：对称填充 → 特征值分解 → 将特征值下限限制为 `min_vec` → 重建矩阵。 |
| **优化目标** | 多元高斯负对数似然，对应 `base/metrics.py` 中的 `MGAU`。 |

`forward(X)` 接收形状为 `(B, seq_len, N, M)` 的输入，返回 `(mu, Sigma)`；训练引擎将第二个返回值视为各预测对应的协方差。模型直接输出自身的预测分布，而不是三个分位数，因此设置了 `cqr_compatible = False`，并会明确拒绝使用 `--cqr` 参数。

报告的指标包括 `MGAU`、`MAE`、`MAPE`、`RMSE`、`CRPS` 和 `KL`。所有指标均针对各预测步长，在逆变换后的原始数据空间中计算。

## 仓库结构

```text
UQGNN/
  src/flow/uqgnn/
    uqgnn_model.py      MDGCN、ITCN 和协方差矩阵构造，即 UQGNN 模型
    main.py             入口：模型参数、邻接矩阵设置、run_experiment()
  base/                 共用框架
    runner.py           run_experiment()：统一实验驱动函数
    model.py            BaseModel 基类接口约定
    engine.py           训练、验证、测试循环，以及检查点保存和结果导出
    CQR_engine.py       保形化分位数回归引擎
    metrics.py          MAE、RMSE、MAPE、CRPS、KL、MGAU 和区间评价指标
    efficiency.py       硬件信息、内存、推理时间和浮点运算量
  utils/
    args.py             通用命令行参数、路径配置和随机种子设置
    dataloader.py       数据集、DataLoader 和数据集注册表查询
    generate.py         原始数组转换为 his.npz、info.json 和数据划分索引
    registry.yaml       数据集名称到数据及邻接矩阵路径的映射
    graph_algo.py       邻接矩阵归一化：对称、转移、切比雪夫和拉普拉斯
    get_adj_mat.py      根据地理 shapefile 文件构建邻接矩阵
    log.py              日志工具
    res.py              结果收集与比较的命令行工具
  jobs/train.sh         Slurm 作业提交脚本
```

## 安装

```bash
conda create -n st python=3.10 -y
conda activate st

# 安装 CUDA 12.8 构建版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## 数据

设置框架使用的数据根目录，默认是 `<repo>/datasets`：

```bash
export POPST_DATA=/path/to/datasets      # 数据集存放位置
export POPST_RESULT=/path/to/result     # 日志和模型检查点存放位置
```

每个数据集目录采用以下结构；英文原文说明这些条目由 `utils/generate.py` 生成：

```text
<POPST_DATA>/<dataset>/
  <adj_name>.npy        邻接矩阵，形状为 N × N
  <years>/
    his.npz            归一化数据和缩放器参数
    info.json          数据形状、缩放器、各划分大小、seq_length_x / seq_length_y
    meta.json          缩放器参数和原始数据形状
    idx_train.npy      训练样本索引
    idx_val.npy        验证样本索引
    idx_test.npy       测试样本索引
    idx_all.npy        全部样本索引
```

从形状为 `(T, N, M)` 的原始数组生成数据：

```bash
python utils/generate.py --data_path /path/to/raw.npy --dataset chicago_15min --years 2018 --fmt NDT
```

支持的输入布局为：`NDT`（T×N×D）、`NTD`（N×T×D）和 `NT`（N×T，自动补充大小为 1 的 D 维）。随后在 `utils/registry.yaml` 中注册数据集：

```yaml
chicago_15min:
  data: chicago_15min
  adj: chicago_15min/chicago.npy
```

程序在运行时从 `info.json` 读取节点数 `N`。如果命令行未指定，`seq_len`、`horizon`、`input_dim` 和 `output_dim` 也会从该文件自动填充。

## 使用方法

```bash
# 训练
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018

# 从模型检查点加载并测试
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018 \
    --mode test --model_path /path/to/UQGNN_<timestamp>.pt

# 测试并导出预测结果文件
python src/flow/uqgnn/main.py --dataset chicago_15min --years 2018 --mode test --export

# 将结果归入 result/<proj>/ 目录
python src/flow/uqgnn/main.py --dataset chicago_15min --proj MyExperiment
```

使用 Slurm 提交作业：

```bash
sbatch jobs/train.sh
DATASETS="chicago_15min nyc_manhattan_15min" YEARS=2018 sbatch jobs/train.sh
```

比较不同运行的结果：

```bash
python utils/res.py --path result/MyExperiment
python utils/res.py --path result/MyExperiment --select RMSE
python utils/res.py --log result/MyExperiment/UQGNN/chicago_15min/<timestamp>.log
```

结果日志保存到 `result/<proj>/UQGNN/<dataset>/<timestamp>.log`，对应的模型检查点 `UQGNN_<timestamp>.pt` 位于同一目录。

## 参数

### UQGNN 模型参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--hidden_dim_s` | `64` | 空间分支 MDGCN 的隐藏层宽度 |
| `--hidden_dim_t` | `64` | 时间分支 ITCN 的隐藏层宽度 |
| `--emb_dim` | `32` | 交互感知嵌入维度 `e` |
| `--kernel_size` | `3` | 时间卷积核大小 |
| `--temporal_layers` | `2` | ITCN 层数 |
| `--min_vec` | `1e-6` | 将 `Sigma` 约束为正定矩阵时使用的特征值下限 |

### 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--bs` | `64` | 批量大小 |
| `--max_epochs` | `2000` | 最大训练轮数 |
| `--patience` | `30` | 验证损失不再改善时，早停机制允许等待的轮数 |
| `--lrate` | `1e-3` | Adam 优化器的学习率 |
| `--wdecay` | `5e-4` | 权重衰减系数 |
| `--dropout` | `0.5` | Dropout 丢弃概率 |
| `--clip_grad_norm` | `5` | 梯度范数裁剪阈值 |
| `--step_size` | `200` | StepLR 学习率衰减间隔 |
| `--gamma` | `0.95` | StepLR 学习率衰减因子 |
| `--seed` | `2025` | 随机种子 |

### 数据与系统参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `chicago_15min` | 数据集名称，必须已在 `registry.yaml` 中注册 |
| `--years` | `2018` | 数据子目录 |
| `--seq_len` / `--horizon` | 自动 | 输入序列长度和预测步数，自动从 `info.json` 填充 |
| `--input_dim` / `--output_dim` | 自动 | 变量数量 `M`，自动从 `info.json` 填充 |
| `--no_normalize` | -- | 禁用默认开启的 MinMax 归一化 |
| `--device` | `cuda` | 计算设备 |
| `--mode` | `train` | 运行模式：`train` 或 `test` |
| `--model_path` | -- | 测试模式下加载的模型检查点 |
| `--export` | 关闭 | 在最终评估时保存预测结果文件 |
| `--proj` | -- | 用于分组保存结果的子目录名称 |

## 实现细节

实验在配备 Intel Xeon 处理器、64 GB 内存和 NVIDIA A100 GPU 的 Linux 服务器上运行。论文参考结果使用 PyTorch 2.3.0 / CUDA 11.8；本发布版本固定使用 PyTorch 2.8.0 / CUDA 12.8。

## 基线模型

确定性预测基线采用以下项目的实现：
[STGCN](https://github.com/hazdzz/STGCN)、
[DCRNN](https://github.com/chnsh/DCRNN_PyTorch)、
[GWNET](https://github.com/nnzhan/Graph-WaveNet)、
[StemGNN](https://github.com/microsoft/StemGNN)、
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN)、
[AGCRN](https://github.com/LeiBAI/AGCRN) 和
[SUMformer](https://github.com/Chengyui/SUMformer)。

概率预测基线采用以下项目的实现：
[TimeGrad](https://github.com/zalandoresearch/pytorch-ts)、
[STZINB](https://github.com/ZhuangDingyi/STZINB)、
[DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch)、
[CF-GNN](https://github.com/snap-stanford/conformalized-gnn) 和
[DiffSTG](https://github.com/wenhaomin/DiffSTG)。

这些模型采用与本项目相同的实验驱动和指标计算流程，其可运行实现均可在 [POPST](https://github.com/UFOdestiny/POPST) 中找到。

## 引用

如需引用本文，可使用以下 BibTeX 条目；书目信息保留英文原文：

```bibtex
@inproceedings{yu2025uqgnn,
  title     = {UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction},
  author    = {Yu, Dahai and Zhuang, Dingyi and Jiang, Lin and Xu, Rongchao and Ye, Xinyue and Bu, Yuheng and Wang, Shenhao and Wang, Guang},
  booktitle = {Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems},
  pages     = {52--65},
  year      = {2025},
  doi       = {10.1145/3748636.3762709}
}
```

## 相关工作

- [POPST](https://github.com/UFOdestiny/POPST)：本发布版本所源自的统一时空预测基准框架，包含约 30 种流量模型、约 17 种起讫点（OD）模型，以及共享的保形预测引擎。
- [EnergyMamba](https://github.com/UFOdestiny/EnergyMamba)（KDD 2026）：结合自适应序贯 CQR 的图增强选择性状态空间模型。
- [TrustEnergy](https://github.com/UFOdestiny/TrustEnergy)（AAAI 2026）：结合序贯 CQR 的记忆增强时空图神经网络。
- [HealthMamba](https://github.com/UFOdestiny/HealthMamba)（IJCAI 2026）：具有三机制不确定性量化的图状态空间模型。

## 许可证

本项目依据 [MIT 许可证](LICENSE) 发布。

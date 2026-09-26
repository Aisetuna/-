# Chicago 实验数据：已准备完成

当前数据覆盖 2022-09-01（含）至 2022-12-01（不含），时间间隔 15 分钟，77 个社区按编号 1–77 排列。
通道顺序固定为出租车下车数、单车还车数、犯罪事件数。数组形状为 `(8736, 77, 3)`。

| 通道 | 纳入的事件数 | 处理 |
| --- | ---: | --- |
| taxi | 1,589,104 | 目标期间共 1,757,927 条，排除 168,770 条无有效下车社区的记录，再排除 53 条下车早于上车的记录 |
| bike | 1,575,097 | 按还车时间筛选，使用还车坐标匹配社区；目标期间排除 22,652 条缺少坐标或未匹配到社区的记录 |
| crime | 64,464 | 按事件时间和社区编号统计 |

## 文件位置

- `datasets/source_files/`：源 CSV、ZIP 和地理边界；源数据说明见目录内 README。
- `datasets/raw/chicago_paper_2022/`：三类计数数组、质量统计、合并数组及 `provenance.json`。
- `datasets/chicago_15min/chicago.npy`：77×77 邻接矩阵。
- `datasets/chicago_15min/2022/`：训练可直接读取的 `his.npz`、`info.json`、`meta.json`、四个索引文件及 `verification.json`。

## 实验设置

输入过去 12 步（3 小时），预测未来 1 步（15 分钟）。按样本时间顺序约 8:1:1 划分：训练 6,980 个窗口，验证 872 个，测试 872 个。相邻窗口允许共享历史输入，预测目标按时间分开。

每个通道独立进行 Min-Max 归一化，仅使用训练区间拟合，无 log1p。所有社区共享同一通道的缩放参数。
空时间格填零，无效位置记录排除。时间沿用本地墙钟时间，夏令时回拨的重复小时合并。
邻接矩阵为 EPSG:26916 投影下的社区质心距离高斯核，低于 0.1 的权重置零。

## 开始训练（PowerShell）

```powershell
cd E:\Desktop\UQGNN-main
.\.venv\Scripts\python.exe src/flow/uqgnn/main.py --dataset chicago_15min --years 2022 --bs 32 --max_epochs 2 --patience 30 --lrate 0.001 --seed 2025 --proj chicago_trial
```

上面是两轮试跑。正式实验可将 `--max_epochs` 改为 `200`，并设置不同 `--proj` 区分实验。当前模型入口使用 `cuda:0`。通道数、输入长度和预测长度自动从 `info.json` 读取，不需要额外指定。
输出位于 `result/chicago_trial/UQGNN/chicago_15min/`。

## 重新生成与核验

```powershell
.\.venv\Scripts\python.exe utils/prepare_chicago_paper.py --taxi-archive "datasets/source_files/taxi/2022/Taxi_Trips_(2013-2023)_20260920.csv"
.\.venv\Scripts\python.exe utils/generate.py --data_path datasets/raw/chicago_paper_2022/chicago_taxi_bike_crime_2022_09_11_TND.npy --dataset chicago_15min --years 2022 --fmt TND --seq_length_x 12 --seq_length_y 1 --per_channel --train_only_scaler
.\.venv\Scripts\python.exe utils/verify_chicago_data.py
```

已验证数组形状、有限非负计数、邻接矩阵、顺序划分、训练区间缩放、逆变换，以及 RTX 4060 Ti 上真实数据批次的前向、Gaussian NLL、有限梯度和一次优化器更新。没有进行完整模型训练，验证报告中的损失是随机初始化的一次测试值，不代表预测效果。

这是一份公开源重建数据，不保证与论文原始实验数据逐条一致。犯罪镜像可能缺少后续补报，边界快照、到达口径和邻接构建也可能与原实验不同。用于研究报告时应明确记录这些设置。

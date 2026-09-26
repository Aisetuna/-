# Chicago 数据准备状态

**最新状态：三通道数据已准备完成，并通过 GPU 前向、反向和一次参数更新验证。当前说明及启动命令见 [CHICAGO_READY.md](CHICAGO_READY.md)。以下内容为历史过程记录，其中“缺少出租车”“下载阻塞”等已解决。**

## 目录管理更新

源文件统一存放于 `datasets/source_files/`，分类说明见该目录的 `README.md`。
`datasets/raw/chicago_paper_2022/` 现在只存放处理结果及质量统计，不再存放 CSV、ZIP 或 GeoJSON 源文件。
预处理脚本已更新为读取新源目录。下文关于旧下载位置的记录仅供追溯。
后续出租车处理必须用 `--taxi-archive` 指定完整且覆盖目标日期的 ZIP，不能使用 `incomplete/` 下的断点文件。

## 用户提供压缩包的核验结果（2026-09-19）

已读取 `E:/Desktop/archive.zip`，内含 `Taxi_Trips__2013-2023__20250414.csv`。
共 6,494,157 行，按下车时间筛选 2022-09-01 至 2022-12-01 仅 1 条，无法覆盖目标时间段。
文件名不能用于确认实际年份。详细月份统计见 `datasets/raw/chicago_paper_2022/taxi_quality.json`。
原压缩包保留不变；试运行产生的不合格出租车数组、合并数组及邻接矩阵已移除，可由脚本重新生成。
预处理现在检查每天是否有记录，发现整日缺失会中止，避免把未提供的数据误当作零需求。
仍需要实际包含 2022 年 9—11 月记录的出租车文件。

已为 `generate.py` 添加可选 `--train_only_scaler`，后续生成时应使用该参数，避免用测试集拟合归一化范围。

目标：2022-09-01（含）至 2022-12-01（不含），77 个社区，每 15 分钟聚合。
这是公开记录重建数据，并非作者发布的原始实验数据，不能保证严格复现论文指标。

## 已完成

- Divvy 2022 年 9、10、11 月官方 ZIP 下载并成功读取。
- 犯罪记录历史镜像下载并成功读取，日期覆盖至 2022-11-30。
- 单车按还车时间和还车坐标所在社区统计：1,575,097 条有效记录。
- 犯罪按事件日期和 Community Area 统计：64,464 条有效记录。
- 两个数组均为 `(8736, 77)`，社区编号按 1–77 排序，91 天每天都有记录。
- `datasets/raw/chicago_paper_2022/bike_counts.npy` 和 `crime_counts.npy` 已保存。
- 同目录 `bike_quality.json`、`crime_quality.json` 保存计数、排除量和每日总数。

来源：

- https://divvy-tripdata.s3.amazonaws.com/202209-divvy-tripdata.zip
- https://divvy-tripdata.s3.amazonaws.com/202210-divvy-tripdata.zip
- https://divvy-tripdata.s3.amazonaws.com/202211-divvy-tripdata.zip
- https://raw.githubusercontent.com/RandomFractals/chicago-crimes/main/data/crimes-2022.csv

## 当前阻塞

出租车镜像：https://www.kaggle.com/datasets/javiertorresvergara/taxi-trips-chicago

`taxi_2020_2025.zip` 仅下载 33,947,648 字节，预期约 876 MB，不能解压或用于完整实验。
2026-09-19 尝试断点续传、多次重试、替代 API 域名及 Python 标准库后，下载连接仍被重置。
没有生成完整三通道数据、邻接矩阵或训练文件，也没有执行三通道 GPU 验证。

## 恢复步骤（项目根目录 PowerShell）

```powershell
curl.exe -L --fail --retry 5 --retry-all-errors --connect-timeout 20 --speed-limit 10240 --speed-time 30 -C - --output datasets/raw/chicago_paper_2022/taxi_2020_2025.zip https://www.kaggle.com/api/v1/datasets/download/javiertorresvergara/taxi-trips-chicago
.\.venv\Scripts\python.exe utils/prepare_chicago_paper.py --channels taxi --taxi-archive datasets/source_files/taxi/taxi_2022_09_11.zip
.\.venv\Scripts\python.exe utils/generate.py --data_path datasets/raw/chicago_paper_2022/chicago_taxi_bike_crime_2022_09_11_TND.npy --dataset chicago_15min --years 2022 --fmt TND --seq_length_x 12 --seq_length_y 1 --per_channel
```

出租车字段和时间覆盖尚待完整下载后核验，脚本遇到不匹配字段会报错。
最终顺序为 taxi / bike / crime。单车与出租车采用到达口径；缺失或区域外坐标被排除。
时间为来源中的当地墙钟时间，夏令时回拨的重复小时合并；无记录的格子填零。
邻接矩阵采用投影坐标社区质心距离的高斯核，阈值 0.1，是明确的重建选择。
历史犯罪镜像可能缺少之后补报或更正的数据。

注意：现有 `generate.py` 在全量数据上拟合归一化参数，包括验证/测试时间段。
用于严谨的预测研究时应改为只用训练区间拟合，并记录该选择；上述命令保留当前仓库行为。

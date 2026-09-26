# Git 版本管理说明

本项目适合把代码和实验配置放进 Git，把数据、模型权重、日志和运行环境留在本地。GitHub 普通仓库对单个文件有大小限制，即使使用 Git LFS，也不适合把数 GB 的原始出租车 CSV 和完整训练数据直接放进代码仓库。

## 已配置的提交范围

根目录 `.gitignore` 会排除：

- `datasets/`：原始数据、聚合数组、归一化训练数据和邻接矩阵；
- `result/`、`*.pt`、`*.pth`、`*.log`：模型权重、日志和导出结果；
- `.venv/`、`.python310/`：Python 运行环境；
- `.chicago_work/`、临时 CSV/ZIP/NPY 文件；
- `obsidian_import_stage/`、`obsidian_restructure_backup/`：与模型无关的本地资料迁移目录。

因此，第一次提交只包含源代码、配置、依赖说明、许可证和数据处理文档，体积应保持在 GitHub 可接受范围内。

## 新电脑上的复现实验

1. 从 GitHub 克隆代码。
2. 按 `requirements.txt` 创建 Python 环境。
3. 从 `CHICAGO_READY.md` 记录的公开来源下载数据，放入本地 `datasets/source_files/`。
4. 运行 `utils/prepare_chicago_paper.py` 和 `utils/generate.py` 生成本地数据。
5. 运行训练命令，结果只写入本地 `result/`。

数据文件不随 Git 提交，因此克隆仓库后需要按说明重新获取；这是为了避免把隐含的大文件推到 GitHub。

## 如果确实要版本化小型模型文件

可以安装 Git LFS 后只跟踪经过筛选的小文件：

```powershell
git lfs install
git lfs track "artifacts/*.pt"
git add .gitattributes artifacts/
```

不建议用 LFS 保存数 GB 的出租车 CSV、原始 ZIP 或整个 `result/`。这类文件适合保留在本地、对象存储或数据集平台，并在 Markdown 中记录下载地址和 SHA-256。

## 提交前检查

```powershell
git status --short
git add -A
git diff --cached --stat
git diff --cached --name-only
```

如果看到数据文件、`.pt`、`.log` 或虚拟环境路径，先检查 `.gitignore`，不要直接推送。

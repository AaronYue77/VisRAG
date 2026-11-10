# 8B 启动与复现

当前 A/B 交互使用 Qwen3-VL-8B-Instruct。C 的验收版本是 120 道训练、50 道开发的 4-bit NF4 QLoRA，使用相同的 8B 底座、提示和证据与 60/25 版对照。[当前发布配置](../8B发布配置.json) · [20 类照片选版报告](../覆盖与选版.md)

## 在 Kaggle 使用

- 日常交互启动文件是 [kaggle_8b_chat.ipynb](kaggle_8b_chat.ipynb)。在 Notebook 开启 GPU，运行初始化单元格，再通过 `ask("你的问题")` 查商品或追问。[已验收的交互 Notebook](https://www.kaggle.com/code/aaronyue77/abo-rag-8b-interactive-chat)
- C 阶段候选训练使用 [kaggle_8b_upgrade.py](kaggle_8b_upgrade.py)，新增 15 类的两版对照使用 [kaggle_8b_breadth.py](kaggle_8b_breadth.py)。[120/50 训练和主集对照](https://www.kaggle.com/code/aaronyue77/abo-rag-8b-120-example-candidate) · [75 道跨类目照片对照](https://www.kaggle.com/code/aaronyue77/abo-rag-8b-breadth75)
- 正式任务在 Tesla T4 上运行，挂载 Kaggle 官方模型 `qwen-lm/qwen-3-vl/transformers/8b-instruct/1`。需要匹配版本的证据夹具和离线运行依赖。交互初始化还使用 ABO 元数据、图像索引和字体。已跑任务及其夹具为私有输入，打开时需要相应账号权限。

## 本地工作区布局

把本仓库放在项目根目录下的 `upstream/VisRAG`，将本目录的 PowerShell 启动文件放进项目根目录的 `scripts`。脚本使用项目根目录的 `.venv`，CPU 建索引用 SQLite FTS5，模型调用需要 CUDA 环境和官方 8B 快照。默认模型目录是 `models/Qwen3-VL-8B-Instruct`，也可通过 `ABO_QWEN8_MODEL_DIR` 指定。

将官方 ABO 商品归档 `archives/abo-listings.tar` 和图像索引 `images/metadata/images.csv.gz` 保存到 `data/ecommerce_rag/abo-source` 对应子目录。索引可从源文件重建，数据来源和署名见仓库首页。[官方 ABO 数据](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)

```powershell
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\full_catalog.py
.\scripts\run_stage_b.ps1 -Command search -Question "black backpack"
.\scripts\run_stage_b.ps1 -Command chat
```

发布内容包括源码、启动文件、协议和评测摘要。完整 SQLite 商品库、8B 底座、LoRA 二进制适配器、逐题原始输出和私有 Kaggle 输入保存在原工作区或云端运行归档。

# VisRAG-Ret 原版核心推理复现

独立 `.venv-visrag`（Python 3.11.18），锁定原版核心推理依赖，详见 [requirements.inference.txt](requirements.inference.txt) 与 [requirements.freeze.txt](requirements.freeze.txt)。与 Qwen 问答环境分进程运行。

本模块直接调用上游 `visrag_scripts/demo/visrag_pipeline/utils.py` 的 `encode`，保留加权池化与 L2 归一化，使用官方固定 revision 的 VisRAG-Ret，页面以 200 DPI 渲染。查询 instruction 与该仓库 demo 保持一致。

适配仅包括：单页批次；把检索不使用的 `llm.lm_head` 放到 CPU；Windows 中文路径的 SentencePiece 词表通过相同 protobuf 字节加载。模型权重没有量化，远程模型代码在本地核对哈希后加载，没有改动。

工作区已有 `scripts/run_visrag_reproduction.ps1`，顺序执行环境检查、27 页向量缓存、固定 10 题查询编码与 Top-2、本地 Qwen 图像回答、自动验收。各阶段必须退出后才启动下一模型。此脚本位于 Git 仓库外，也可直接运行模块：

```powershell
# 从 E:\大模型项目自学E\doc-vqa-assistant 执行
.\.venv-visrag\Scripts\python.exe .\upstream\VisRAG\personal\visrag_repro\check_environment.py
.\.venv-visrag\Scripts\python.exe .\upstream\VisRAG\personal\visrag_repro\retrieve.py index --index-name my-pages
.\.venv-visrag\Scripts\python.exe .\upstream\VisRAG\personal\visrag_repro\retrieve.py retrieve --index-name my-pages --retrieval-name my-queries --top-k 2
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\visrag_repro\generate.py --retrieval-name my-queries --run-name my-visual-run
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\visrag_repro\verify_results.py --index-name my-pages --retrieval-name my-queries --run-name my-visual-run
```

当前数据是原 A 基线的两份技术 PDF，用于核验检索实现和相同题对照。业务方向已改为电商，入口见 [电商 RAG](../ecommerce_rag/README.md)。这些历史题不能代替电商评测。

产物位于工作区 `data/visrag_repro/<index-name>/reps.npy`、`page-map.jsonl`、页面图片，以及 `<retrieval-name>/query_reps.npy`、`similarities.npy`、`retrieval.jsonl`。每行向量与物理页码对应。问答结果在 `reports/a_baseline/<run-name>/逐题结果.txt` 和 JSONL；验收 TXT 为 `记录/VisRAG原版检索复现验收与对照.txt`。

未安装 DeepSpeed、decord、pytrec_eval：它们不参与本次 PDF 图像推理；不将该环境表述为完整 Linux 训练环境。问答仍使用 Qwen3.5-2B，不能据此声称复现了原论文所有生成器或其报告成绩。

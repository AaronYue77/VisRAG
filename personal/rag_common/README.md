# 当前多模态 RAG 的公共运行模块

这里保存当前电商项目复用的 PDF/OCR、图片渲染、VLM 装载、引用解析、指标、逐题 TXT、缓存和操作记录能力。

- `core.py`：文档、图像、模型与评测辅助函数。
- `audit.py`：只汇总 `reports/ecommerce_rag/` 的逐题结果。
- `common.py`：E 盘缓存、JSONL、哈希与连续日志。
- `test_core.py`：现有评测/记录保护检查。

连续日志按用户原约定仍保存在 `personal/a_baseline/docs/操作方法与结果记录.txt` 和 `问题与解决记录.txt`；该旧目录只保留日志，不再保留旧论文业务程序。

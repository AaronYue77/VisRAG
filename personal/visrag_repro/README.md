# VisRAG-Ret 页面视觉编码工具

保留官方固定 revision 的编码器、原版核心依赖与 Windows 中文路径兼容。旧论文十题的生成/比较脚本和缓存已删除；工具改为从电商文档清单构建索引，按请求型号/店铺/有效期限制候选。

```powershell
# 从 E:\大模型项目自学E\doc-vqa-assistant 执行
.\scripts\run_visual_index.ps1 -IndexName my-visual-pages -RetrievalName my-visual-queries
```

该命令依次执行独立 `.venv-visrag` 环境检查、页面编码、问题编码和候选排序，各进程顺序退出。输出在 `data/visrag_repro/<name>/`，包含向量、页码/问题映射、配置、源码快照和检索记录。

`retrieve.py index --corpus-dir data/ecommerce_rag/demo-v1` 默认读取电商 `documents.json`；`retrieve` 使用同目录的开发题。只接受 `smoke_dev/dev`，不自动运行最终测试。业务路由不使用参考答案或参考页；实时请求等不编码。

这只是视觉检索工具，尚未替换电商应用中的文字检索，也没有自动调用 VLM 回答、实现融合或训练。当前应用入口仍在 `personal/ecommerce_rag/`。

模型权重不量化，直接调用上游 `utils.encode`；未参与检索前向的 lm_head 放 CPU。只通过 Python 读取原词表 protobuf 字节，解决旧 SentencePiece 中文路径问题。模型/编码规则不变。

依赖见 `requirements.inference.txt` 与 `requirements.freeze.txt`。模型来源、哈希和环境记录保存在 `reports/visrag-repro/`，这些是后续视觉能力的运行资产，不是旧业务数据。

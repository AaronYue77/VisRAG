# VisRAG-Ret 页面视觉编码工具

保留官方固定 revision 的编码器、原版核心依赖与 Windows 中文路径兼容。旧论文十题的生成/比较脚本和缓存已删除；工具改为从电商文档清单构建索引，按请求型号/店铺/有效期限制候选。

当前电商入口通过 B 阶段对全量文本候选启用视觉重排。

```powershell
.\scripts\run_stage_b.ps1 -Visual
```

B 先用文字检索召回商品，再用官方 VisRAG-Ret 对最多 6 个候选的照片重排。重排任务与结果缓存在全量索引目录的 `visual-tasks` 中，最终回答由电商应用的 8B 生成器处理。

独立 `retrieve.py` 仍可在有文档清单的资料集上构建视觉索引，并在开发题上评估。

模型权重不量化，直接调用上游 `utils.encode`；未参与检索前向的 lm_head 放 CPU。只通过 Python 读取原词表 protobuf 字节，解决旧 SentencePiece 中文路径问题。模型/编码规则不变。

依赖见 `requirements.inference.txt` 与 `requirements.freeze.txt`。模型来源、哈希和环境记录保存在 `reports/visrag-repro/`，这些是后续视觉能力的运行资产，不是旧业务数据。

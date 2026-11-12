# 电商多模态 RAG · Qwen3-VL-8B-Instruct

A 构建 147,702 条 ABO 商品的全量 SQLite 库、检索索引和证据页。B 从全库搜索并选择商品，按问题从原始字段、商品照片或组合证据回答。普通对话和需要读图的商品问题共享 8B 生成模型。C 在 Kaggle 用固定证据比较原提示、改进提示和 QLoRA。

- [A–C 流程图与代码对应](docs/A-C架构与流程.md)
- [8B 启动、Kaggle 交互和微调](docs/kaggle-8b/launch/README.md)
- [20 类照片与 120/50 适配器选版](docs/kaggle-8b/覆盖与选版.md)
- [代码阅读地图](docs/代码阅读地图.md)

本地有完整 ABO 源和索引时，可从工作区根目录用 PowerShell 检查 A、搜索 B、查询 C 的云运行状态。

```powershell
.\scripts\run_stage_a.ps1 -Command check
.\scripts\run_stage_b.ps1 -Command search -Question "running shoes"
.\scripts\run_stage_c.ps1 -Command cloud-status
```

B 可选稠密、融合、交叉编码器重排及小候选集内的 VisRAG 图片重排。冻结协议和已完成结果保留在本地数据与报告目录，发布的成绩按对应实验版本解读。

# 电商商品知识与售后服务多模态 RAG

当前主线是电商 RAG，后续再接电商 Agent。[代码阅读地图](docs/代码阅读地图.md)列出每一步的实际入口；共享能力位于 [rag_common](../rag_common/README.md)。旧论文数据、旧业务程序、旧实验与一次性生成脚本已清理。

## 已实现

PDF 原生文字→BGE 页面文本向量缓存与 BM25→型号/店铺/有效期过滤→RRF Top-2→页面图像→本地 Qwen3.5-2B→引用与拒答保护→JSONL/TXT。

业务路由会对缺型号/冲突澄清、未知资料拒答、实时请求返回 `requires_tool`。未连接真实订单、库存、价格或交易服务。输出保护不是通用语义忠实度判别器。

## 本机命令

从 `E:\大模型项目自学E\doc-vqa-assistant` 执行；每次使用新运行名：

```powershell
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\run.py run --mode hybrid --run-name ecom-my-check
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\ask.py "DN-R100 的接口是什么颜色？" --sku DN-R100 --run-name ecom-my-question
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\test_retrieval.py
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\rag_common\test_core.py
.\.venv\Scripts\python.exe .\scripts\verify_setup.py
```

资料不变时不要重复生成或覆盖索引。新资料另建数据版本与评测协议；`prepare_demo.py` 只是模拟资料生成器。当前数据为 4 份、8 页、16 题开发样例，没有独立电商测试集合。

## 后续多模态能力

[VisRAG-Ret 工具](../visrag_repro/README.md)保留独立环境、视觉编码与业务过滤，并改为读取电商资料；它独立保存向量和候选，尚未替换当前应用检索器，也没有完成视觉—文字融合或 LoRA 对照。

先补真实商品来源与独立评测，再根据错误验证视觉融合、重排、最多一次改写重试。正确证据诊断后再确定训练目标。当前 BGE 是文本编码器，图片由 VLM 阅读，不把文本 RRF 说成视觉融合。

## 保存位置

- 工作区 `reports/ecommerce_rag/<run-name>/逐题结果.txt`、`results.jsonl`、配置和源码快照。
- 汇总：`记录/逐题结果汇总.txt`；清理清单与验证：`记录/项目清理清单与验证结果.txt`。
- 连续日志：[操作方法与结果](../a_baseline/docs/操作方法与结果记录.txt)、[问题与解决](../a_baseline/docs/问题与解决记录.txt)。旧目录名仅为保持日志路径兼容。

工作区 `记录/电商多模态RAG实施流程_A-B-C修订版.txt` 是唯一当前实施路线。已有电商原始结果、配置、输入图片与快照保留；旧论文内容不再作为启动或验证依赖。

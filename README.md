# 电商商品知识多模态 RAG

面向商品资料查阅、规格表和商品图片问答的个人 AI 应用项目。先完成 RAG，后续再接电商 Agent。

## A 阶段真实数据基线

- 官方 Amazon Berkeley Objects（ABO）元数据/原图，24 个椅子商品，转换为48页来源可追溯PDF资料卡。转换PDF、152道题和开发/测试划分为本项目自建，不是官方问答基准或厂家原始手册。
- 原生文字块/坐标 + 必要时 RapidOCR → BM25文字块检索 → 父页面去重 → 本地 Qwen3.5-2B 看页面图片 → 引用/拒答/逐题记录。
- 商品编号、站点、快照日期范围规则路由；澄清、资料不足、需要实时工具、场外分支。当前没有执行订单、价格、库存或退款工具。
- 两个单因素改进：整页→文字块（修复跨商品证据漏召回，也暴露引用退化）；再单独澄清引用标签与物理页码。全部收益与退化保留。
- 冻结的12个独立商品测试组共76题：56道可回答题中55道满足答案及引用要求；12道未知题、8道业务边界题动作正确；零运行异常。一次裸No未满足JSON/引用协议，系统拒答，未追分修补。
- 39项单元/回归检查通过；本地HTTP健康检查、输入校验、真实图片问答与2槽并发保护通过。

这些是小型、自建、含字段提示的资料卡任务；单商品2页而Top-2全覆盖，高召回有任务设计因素。不能作为普遍真实客服准确率、线上A/B收益或生产上线证明。原始属性也有质量问题，出处可靠不代表每字段正确。

## 运行与证据

工作区 `E:\大模型项目自学E\doc-vqa-assistant`，所有模型、数据、环境和缓存留在E盘。

```powershell
.\scripts\run_stage_a.ps1 -Command ask -Question 'B07124WN69 的规格表宽度是多少？保留单位。' -Sku B07124WN69
.\scripts\run_stage_a.ps1 -Command serve -Port 8765
.\scripts\run_stage_a.ps1 -Command dev -RunName a-my-dev-01
```

- [A阶段面试覆盖与代码对应](personal/ecommerce_rag/docs/A阶段实施与面试覆盖.txt)
- [A阶段用法](personal/ecommerce_rag/README.md)
- [操作方法与结果](personal/a_baseline/docs/操作方法与结果记录.txt)
- [问题与解决](personal/a_baseline/docs/问题与解决记录.txt)
- 本地真实测试逐题TXT：`reports/ecommerce_rag/abo-a-test-final-v1/逐题结果.txt`；总表：`记录/逐题结果汇总.txt`。
- 本地验收：`记录/A阶段完成情况与验收报告.txt`；数据来源与署名：`data/ecommerce_rag/abo-cards-v1/ATTRIBUTION.txt`。
- 运行保留JSONL、原始输出、图片、配置/代码SHA、输入清单与代码快照；gold不进入正常Reader，oracle仅开发诊断。

## 后续与历史回归

B拟比较BGE、VisRAG-Ret、文字视觉融合/重排；C拟做有目标LoRA；电商Agent后续再接业务工具。A没有从零训练模型、强化学习、视频或多Agent。

旧的4份PDF/16题虚构路由器样例保留作功能回归，不能与真实ABO测试混报。原模拟应用的BM25+BGE/RRF和独立VisRAG视觉工具仍保留，不表示真实商品视觉融合已完成。

## 来源

上游起点 `OpenBMB/VisRAG@7205787862e35d70cee7ebfd8f2b58d3eb5d563b`。[原作者README](README.upstream.md)、[LICENSE](LICENSE)和原视觉算法代码保留。A的BM25链路是个人应用模块，不宣称复现原论文视觉检索指标。

ABO源数据/图片归Amazon.com，数据集作者署名和CC BY 4.0原文随本地数据保存。来源：[官方数据页](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)。基础模型与上游成果归各原作者；本项目业务路由、资料转换、分块/引用实验、评测和记录按实际实现说明。

当前开发分支 `feat/ecommerce-rag`；默认master仍保留上游基线。

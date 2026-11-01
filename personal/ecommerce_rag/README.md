# 电商商品知识与售后服务多模态 RAG

2026-11-01 用户确认的新业务方向。先实现可追踪的 RAG，后续将这个检索问答能力作为电商 Agent 的一个工具。当前支持商品参数、接口示意图、排障说明、政策与资料版本问答。

项目首页见 [仓库 README](../../README.md)。第一次阅读先看 [电商代码阅读地图](docs/代码阅读地图.md)，可按 `ecom-03` 追踪索引、检索、图像输入、回答和记录。

## 已实现

- PDF 页面文本与图像：文字用于检索，实际页面图像交给本地 Qwen3.5-2B 作答，答案附 PDF 页码引用。
- 在检索前过滤店铺、SKU 和资料生效日期。对比两种型号时保留双方候选；旧版资料只用于明确指定的历史日期。
- BM25 对照和 BGE-small-zh-v1.5 中文稠密检索，经 RRF 融合；向量预先保存，编码在 CPU 运行。
- 未提供型号或上下文冲突时澄清；未知型号/缺少资料时拒答；价格、库存、订单请求返回 `requires_tool`，明确工具尚未接通。
- JSON、页码范围与显式拒答保护。这里没有实现通用语义忠实度判别器，也没有自我反思检索循环。
- 逐题 JSONL、TXT、配置、源代码快照、检索排序、引用、耗时和模型 revision。`answer_source` 区分模型、规则路由和输出保护。

## 本地运行

在工作区 `E:\大模型项目自学E\doc-vqa-assistant` 的 PowerShell 中执行。环境、模型和缓存均已放在 E 盘；新运行必须用新名字，旧结果不会被覆盖。

```powershell
# 数据与向量已经准备好；更换数据后另建数据版本并重建索引
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\run.py index

# 固定 16 个模拟功能题
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\run.py run --mode hybrid --run-name ecom-my-check

# 任意提问，不输入参考答案；支持重复 --sku 以比较型号
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\ask.py "DN-R100 的 WAN 接口是什么颜色？" --sku DN-R100 --run-name ecom-my-question

# 不占用 GPU 的业务边界检查
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\ecommerce_rag\test_retrieval.py
```

`--as-of YYYY-MM-DD` 表示资料查询日期；当前演示默认 2026-11-01。`--store` 默认 `demo_store`，该字段是演示上下文，不是生产认证/权限系统。

## 数据与已知边界

演示语料是**自行编写的虚构商品与政策**：两个路由器型号、R100 新旧版本、一个模拟店铺政策，共 4 份 PDF、8 页、16 个开发问题。品牌、参数、退货期限全部是测试夹具，不能作为真实商家承诺。生成脚本为 [prepare_demo.py](prepare_demo.py)，图片与 PDF 已人工检查。

数据位于工作区 `data/ecommerce_rag/demo-v1/`。11 道题有可召回证据，其余检查未知信息、澄清与实时工具边界。它们全部是开发题，没有独立测试集；小样例的高命中率不能代表线上能力。

`ecom-bm25-smoke-v1` 与 `ecom-hybrid-smoke-v1` 使用同一提示词：均 11/11 Top-2 证据命中，动作 15/16，严格文本匹配 14/16。**没有观察到混合检索收益**。第 11 题说“未报告”却返回普通 answer；v2 强化提示仍失败，后续用显式拒答保护修正。第 1 题“2个”与参考“2 个”语义一致，却被当前英文式匹配器判错；旧标签和结果保持不变。后续指标须支持中文规范化/人工复核，不能用这一错误声称模型能力不足或提升。

生成使用 GPU，BGE 使用 CPU；模型未同时占满显存。页面图像回答属于多模态应用，但当前文本向量检索不等于视觉向量检索。原版 VisRAG-Ret 的独立复现入口在 [../visrag_repro](../visrag_repro/README.md)，其历史论文数据仅用于技术回归，未冒充电商评测。

## 后续 RAG 工作顺序

1. 固定一个真实商品类目，收集可引用的官方手册、图示、FAQ、带版本的店铺政策；记录来源、许可、SKU、日期与页码。扩充同义问法、型号近似、跨页、图示、过期资料和未知问题；按原文档/型号家族拆开发与测试，不把改写重复题跨集合。
2. 固定相同数据和生成器，比较 BM25、稠密、RRF；再按失败证据引入重排、段落/父页面映射与视觉检索。对比 Recall@k、MRR、引用支持、拒答、端到端正确性、P50/P95 耗时与成本；按原文档聚类做配对 bootstrap，数据充足后报告区间。
3. 若仍有可修复的漏召回，再增加“证据充分性检查→最多一次改写检索→回答或拒答”，记录每一步、最大调用次数与收益/成本。它是有界 RAG 工作流；不能因为多了一次大模型调用就宣称是多 Agent。
4. RAG 验收后，才接电商 Agent 的工具调用、会话状态、人工转接，以及模拟订单/库存接口。知识回答与订单状态分开取证；任何交易执行需要另行设计和验收。

当前没有实现重排、查询改写、在线 A/B、GRPO、奖励模型、MCP、Agent 或真实业务接口。离线相同题对照是配对实验，不称为在线 A/B。

## 面试中可以讲什么

围绕实际产物讲：为什么 SKU/日期先过滤，为什么 BM25/BGE 用排名融合，跨型号如何保留证据，纯文字能否读图示，引用为何不能证明语义正确，未知问题如何拒答，实时信息为什么不能从旧手册生成。用原始失败题、代码与配置快照说明改动；不能声称模拟数据上的结果是线上收益。

参考的是 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 的检索、评测与有界反思思路；[AgentGuide](https://github.com/adongwanai/AgentGuide) 作为后续工程化学习资料；[面试资料仓库](https://github.com/bcefghj/ai-agent-interview-guide) 仅用于整理表达，其 Python 示例明确是最小骨架。本模块独立实现，保留上游 VisRAG 许可证和引用。

## TXT 保存位置

- 每次运行：工作区 `reports/ecommerce_rag/<run-name>/逐题结果.txt`、`results.jsonl`；固定题批次还有 `运行报告.txt`。
- 所有运行汇总：工作区 `记录/逐题结果汇总.txt`。
- 方法与验证：[操作方法与结果记录.txt](../a_baseline/docs/操作方法与结果记录.txt)。
- 失败与修复：[问题与解决记录.txt](../a_baseline/docs/问题与解决记录.txt)。
- 方向及阶段验收：工作区 `记录/电商RAG调整与分阶段验收方案.txt`。

目录与商品追问修复：自然目录查真实统计，类目搜索先规范化并过滤，“还有呢”承接最近任务；结构化概览按源字段递进，事实引用与普通开放对话保留。

[方法、完整会话实测和TXT位置](../../../../记录/目录查询_鞋类检索与商品追问修复_20261108.txt)

开放对话范围：普通内容没有预设题目或主题白名单，结合近期上下文由模型回答；具体商品知识按需取证。新话题无需逐一加到程序。

[本次范围修正与新话题真实实测](../../../../记录/开放对话范围与新话题实测_20261108.txt)

当前对话入口修复（2026-11-08）：**可以自由聊天，具体商品资料才检索ABO库并引用。** 支持近期聊天追问、查商品后换话题，再回到商品取证。两个默认A/B入口均已修复；正在运行的旧进程需要先输入“退出”再启动。

[修复方法、真实对话验收与记录位置](../../../../记录/全量版自由对话修复与实测_20261108.txt)

---
# 全量ABO A/B阶段（2026-11-08，当前默认）

使用完整147,702条官方本地元数据，涵盖576个原始类目、32个站点，保留145,615个不同商品编号及站点身份。`full_catalog.py`无损入库/FTS，`full_dense.py`全量E5向量，`full_retrieval.py`四模式对照，`full_cards.py`按需官方原图/派生PDF，`full_app.py`商品搜索选择/本地VLM读图/引用/会话与HTTP，`full_evaluate.py`冻结评测，`full_report.py`报告。

从项目根目录运行：

```powershell
.\scripts\run_stage_a.ps1 -Command check
.\scripts\run_stage_b.ps1 -Command chat
```

两个默认入口均已切换全量。B可选`-Mode dense/hybrid/rerank`，`-Visual`只在最多6个文字候选内做真实VisRAG图片重排。开发集选择BM25默认；保留融合/重排退化及首次BGE失败，不宣称复杂方案必然更好。图片/PDF按需，不是全库原图视觉索引。

[全量完成验收](../../../../记录/A_B阶段全量ABO升级完成与验收报告_20261108.txt)、[全量启动与TXT](../../../../记录/全量ABO启动与记录位置_20261108.txt)、[面试代码映射](../../../../记录/全量ABO面试覆盖与代码对应_20261108.txt)。旧24商品样本与历史冻结题集未覆盖；A用`-Mode enhanced`，B用`-Corpus sample`返回历史应用。以下旧“默认”说法以本节为准。

---
# 历史：B阶段检索对照与独立应用入口（2026-11-04）

B完成 BM25、BGE、文字RRF、VisRAG页面视觉、多模态RRF、独立交叉重排和最多一次查询扩展。200道自建回归题，按开发选择、全部失败保留。BGE优于本次原问题BM25，后补A主题规范化参考更强，**不宣称超过现有A增强应用**。A启动器保留。

```powershell
.\scripts\run_stage_b.ps1 -Command check
.\scripts\run_stage_b.ps1 -Command chat
```

[面试覆盖与取舍](docs/B阶段面试覆盖与取舍.txt)、[链路与代码地图](docs/B阶段链路与代码地图.txt)、[完成与验收](../../../../记录/B阶段完成情况与验收报告.txt)、[启动与TXT位置](../../../../记录/B阶段启动与记录位置.txt)。静态与交互Reader分别验证，不能混用准确率。尚未实施C微调或电商工具Agent。

---
以下A说明与历史保留。

当前入口已升级 **a-dialogue-v3.3**：自然目录请求直接列出24个真实编号/原名，覆盖范围、取消/接续和需求保护已补充；商品字段仍取本轮页面证据。普通知识质量仍有限。[最新修复与验收TXT](../../../../记录/A阶段最新会话问题修复与验证_20261103.txt)。以下保留之前版本说明，启动仍用 `scripts/run_stage_a.ps1 -Command chat`。

# A阶段开放对话与商品RAG入口

2026-11-03 本地增强版为 `a-dialogue-v3.0`。一般知识、科学、编程、写作和日常交流由现有本地Qwen自然回答；最近6轮/6000字符普通历史用于追问。选过商品之后仍可换到普通话题。

具体商品事实继续用商品过滤、BM25、页面图片及原文引用校验；缺资料不转闲聊编造。没有实时业务接口，不是自主Agent。独立21轮流程检查与18项接口/恢复检查通过，但人工发现一般知识解释错误，不能称答案全正确或与ChatGPT质量等同。

从项目根目录运行：
```powershell
.\scripts\run_stage_a.ps1 -Command check
.\scripts\run_stage_a.ps1 -Command chat
```

启动后在“你：”提示符输入；默认chat/ask/serve均为增强版。原基线使用 `-Mode baseline`；index/dev/test/grid仍指原静态流程。原152题、已有报告及基线代码保留。

当前说明：[开放对话实现与验收TXT](../../../../记录/A阶段开放对话_实现与验收说明_20261103.txt)。每会话仍有对话记录.txt、逐题结果.txt及results.jsonl，记录范围/上下文/证据/调用/耗时。

---
以下保留2.8及更早历史说明，能力与启动命令以以上当前入口为准。

# A阶段对话增强入口

2026-11-03 本地版本为 `a-dialogue-v2.8`：动作/需求分开识别，保留组合查询，补充来源冲突提示与意图确认闭环。纯致谢使用有限控制语法，开放表述继续走本地模型。新18轮独立验收为17/18；唯一动作差异是记录需求时顺带进行了有证据的回答，未回改评分。

当前交付与限制见工作区 `记录/A阶段剩余完善_完成与验收记录_20261103.txt`。原2.7的22/24、原2.6的20/23及静态基线报告均保留，成绩分别报告。

默认 `chat/ask/serve` 使用独立的 `dialogue_app.py` 增强版本。原 A 基线 Python 实现和冻结语料保持不变，用 `-Mode baseline` 进入。

```powershell
.\scripts\run_stage_a.ps1 -Command check
.\scripts\run_stage_a.ps1 -Command chat
.\scripts\run_stage_a.ps1 -Mode baseline -Command chat
```

新增语义分流、受限短期上下文、澄清接续、需求更新、原文证据校验；同一本地 Qwen 实例复用。新资料是同 24 个商品的 73 页来源卡，保留完整 159 条英文产品要点，独立于旧 48 页/152 题。

工作区 `记录/A阶段对话应用完善_使用说明.txt` 与 `记录/A阶段对话应用完善_验收说明.txt` 说明当前结果和限制；新会话保留 `abo-dialogue-*/对话记录.txt`、`逐题结果.txt`、`results.jsonl`。不称自主 Agent，不确认实时业务或现行店铺政策。

下面保留原规则会话和静态基线说明；其交互命令请显式附加 `-Mode baseline`。`index/dev/test/grid` 仍指原静态基线，不会变成新版对话评测。

# 原版终端连续聊天

```powershell
.\scripts\run_stage_a.ps1 -Command chat
```

看到“你：”后直接提问，输入“退出”结束。支持问候、商品选择、代词追问、重置及异常后继续；本地模型同进程只加载一次。
记录保存在每次启动显示的 `abo-chat-*/对话记录.txt` 和 `逐题结果.txt`，原始输出/引用保留。HTTP `/ask` 返回 `session_id`，后续带同一ID继续，独立会话隔离商品上下文。旧 `ask` 保留单次问答。

[使用范围与验证](docs/终端连续聊天.txt)

# A阶段真实商品入口

真实数据主入口是 `stage_a.py`、`stage_a_index.py`、`stage_a_app.py`；原 `run.py`/`ask.py` 仍是模拟回归，不要混报。

从项目工作区运行：

```powershell
.\scripts\run_stage_a.ps1 -Command ask -Question 'B07124WN69 的宽度是多少？保留单位。' -Sku B07124WN69
.\scripts\run_stage_a.ps1 -Command serve -Port 8765
.\scripts\run_stage_a.ps1 -Command index
```

HTTP：`GET /health`；`POST /ask`，JSON含question和request。request仅允许skus数组、store_id、as_of。服务只绑定127.0.0.1；Ctrl+C结束；未云端上线。

数据重建：在新语料版本、测试冻结前，依次执行 `abo_data.py download/profile/select/cards/annotate`、`stage_a.py index`。cards需reportlab和pymupdf；下载可用能稳定连接官方S3的Python环境，来源和SHA全部记录。

现有 `abo-cards-v1` 已冻结，不要覆盖题集或资料；新实验保留不同运行名。详情见 [面试覆盖与代码对应](docs/A阶段实施与面试覆盖.txt) 及工作区 `记录/A阶段运行与记录位置.txt`。

---
以下是保留的旧模拟回归说明：

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

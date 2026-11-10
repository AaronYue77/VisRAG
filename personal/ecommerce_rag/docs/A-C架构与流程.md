# A–C 项目架构与流程

本项目把商品资料问答拆为三个可独立验收的阶段。A 统一商品身份、索引和证据格式，B 负责多轮交互时的检索与证据选择，C 用固定证据检验回答、引用和拒答。三阶段共享来源标识与逐题记录。D、E 的接口和实现留待后续阶段。

## A · 商品数据、证据和检索基线

```mermaid
flowchart TD
    src["官方 ABO 商品元数据<br/>图像清单和可按需读取的原图"] --> load["无损解析 · 校验源文件与商品记录<br/>ASIN 和站点共同确定商品身份"]
    load --> store["SQLite 全量库<br/>147,702 条 · 576 类 · 32 站点"]
    store --> fts["FTS5 BM25<br/>名称、型号、类目和站点筛选"]
    store --> cards["按需生成派生 PDF<br/>属性、原图、原始产品要点"]
    cards --> pages["页号、商品身份、来源与 SHA256<br/>文字页抽取和图像输入"]
    docs["历史 A 基线的派生资料卡"] --> text["原生文字块优先<br/>需要时用 RapidOCR"]
    text --> blocks["块级 BM25 → 父页面去重"]
    query["用户问题"] --> intent{"先判断商品意图和范围"}
    intent -->|普通话题| chat["8B 开放对话 · 近期普通历史"]
    intent -->|按名称搜索| fts
    intent -->|已选或已指定身份| select
    fts --> select["候选排序 · 选商品或按身份限定"]
    select --> cards
    pages --> reader["证据 Reader · 直接字段或 8B 读页<br/>答案、引用、缺失处理"]
    blocks --> legacy["历史 Reader 对照<br/>整页与文字块、引用标号"]
    reader --> audit["A 验收：字段、引用、边界<br/>保存逐题 TXT 和 JSONL"]
    legacy --> audit
    audit --> next["交给 B：全量索引、页面和来源标识<br/>可复用 Reader 与回归数据"]
```

| 模块 | 对应实现 |
| --- | --- |
| 全量导入、商品家族、站点身份和索引 | [full_catalog.py](../full_catalog.py)、[full_catalogue.py](../full_catalogue.py) |
| 官方图像缓存、资料卡和来源信息 | [full_cards.py](../full_cards.py) |
| 历史 A 分块、OCR、父页检索与 Reader | [stage_a_index.py](../stage_a_index.py)、[stage_a.py](../stage_a.py) |
| 统一交互、路由和逐题记录 | [full_app.py](../full_app.py)、[full_conversation.py](../full_conversation.py)、[full_audit.py](../full_audit.py) |

原 24 商品/152 题属于历史 A 基线。全量检索覆盖 147,702 条元数据，原图与派生页按使用需要生成。照片能提供哪些证据取决于源图实际展示的内容。

## B · 全量检索、对话和自适应取证

```mermaid
flowchart TD
    a["A 的全量库、资料卡和引用协议"] --> query["会话入口<br/>目录、搜索、选商品、追问"]
    query --> route{"先检查明确规则<br/>模糊意图由 8B 分类"}
    route -->|普通话题| open["8B 开放回答<br/>普通历史与商品上下文分开"]
    route -->|浏览目录| catalogue["真实类目统计与分页<br/>支持类目短选"]
    catalogue -->|选类目后搜索| retrieve
    route -->|搜索| retrieve["按 ASIN、站点、类目约束<br/>查询规范化与候选排序"]
    retrieve --> rank["BM25 默认<br/>可对照 E5 Dense、RRF、BGE 重排"]
    rank --> visual["可选 VisRAG 页面视觉重排<br/>限 6 个文字候选"]
    rank --> chosen["展示候选并确认商品"]
    visual --> chosen
    route -->|已明确的商品问题| evidence{"本轮需要什么证据？"}
    chosen --> evidence
    evidence --> fields["明确字段<br/>直接读取官方源值与单位"]
    evidence --> photo["图片属性<br/>按需原图页与 8B 读图"]
    evidence --> mixed["字段与图混合或解释类<br/>组合来源或用证据 Reader"]
    photo --> guard["引用校验 · 缺证据明确拒答<br/>可选最多一次换页"]
    mixed --> guard
    fields --> result["统一输出<br/>回答、引用、路由、耗时"]
    guard --> result
    open --> result
    route -->|售后快照或业务动作| policy["只读政策快照带时间范围<br/>交易、退款、实时订单标记需工具"]
    policy --> result
    rank --> bench["检索离线对照<br/>120 道查询 · 同题 Top-5 池化"]
    result --> c["交给 C：证据 Reader、页格式<br/>引用与拒答协议"]
```

| 模块 | 对应实现 |
| --- | --- |
| 商品对象、站点过滤、最近任务和普通聊天 | [full_app.py](../full_app.py)、[full_conversation.py](../full_conversation.py) |
| 文本 BM25、多语言 E5、RRF 与独立重排 | [full_retrieval.py](../full_retrieval.py)、[full_dense.py](../full_dense.py)、[full_encoder.py](../full_encoder.py) |
| 候选内视觉重排 | [full_visual.py](../full_visual.py) |
| 结构化字段、原图页、换页预算和政策快照 | [full_evidence.py](../full_evidence.py)、[full_policy.py](../full_policy.py) |
| 全库检索对照与新题池化 | [full_evaluate.py](../full_evaluate.py)、[c8_retrieval_eval.py](../c8_retrieval_eval.py) |

全库 BM25 是当前默认。Dense、RRF 和重排属于可选实验路径。Kaggle 交互验收用全量 SQLite BM25。Kaggle 固定证据评测与日常应用的在线取证分别计分。

## C · 8B 微调和同题配对评测

```mermaid
flowchart TD
    b["复用 B 的 Reader、证据标识<br/>和 JSON 回答协议"] --> freeze["冻结商品家族、答案、页序与哈希<br/>同型号家族隔离"]
    freeze --> train["训练 60 → 120 道<br/>图文、单位、引用和拒答"]
    freeze --> dev["开发 25 → 50 道<br/>新旧适配器同题"]
    freeze --> test["210 道问答 + 100 道照片<br/>原集 180+90，补测 30+10"]
    freeze --> breadth["新增 15 类共 75 张<br/>与原照片分开统计"]
    model["Kaggle Tesla T4<br/>Qwen3-VL-8B-Instruct · NF4 · FP16"] --> c0["C0 · 原提示"]
    model --> c1["C1 · 加字段、单位与引用核对"]
    model --> qlora["QLoRA · 1 epoch<br/>末两层文本注意力 q/v · r8 · alpha 16"]
    train --> qlora
    qlora --> saved["保存适配器 · 记录 loss、显存和 SHA256"]
    saved --> c2["C2 · 同一 C1 提示 + 重载适配器"]
    model -.-> c2
    dev --> paired["初版 C0/C1/C2<br/>新版用旧、新 C2 配对"]
    test --> paired
    breadth -->|仅旧、新适配器| paired
    c0 --> paired
    c1 --> paired
    c2 --> paired
    paired --> score["严格检查 JSON、原值单位与引用<br/>单列正确拒答和误拒答"]
    score --> gate["主集至少一项提高且分层不降<br/>新增照片与开发集保护"]
    gate --> export["归档两版模型和逐题记录<br/>通过后切换 C 默认配置"]
```

| 模块 | 对应实现 |
| --- | --- |
| 训练/开发/测试家族隔离与派生表格 | [c8_protocol.py](../c8_protocol.py) |
| 原图照片、人工标注与新增 40 题 | [c8_photos.py](../c8_photos.py)、[c8_supplement.py](../c8_supplement.py) |
| 8B 4-bit 加载、QLoRA、相同输入三组预测 | [qwen8_runtime.py](../qwen8_runtime.py)、[c8_experiment.py](../c8_experiment.py) |
| 评分器和逐题配对退化记录 | [stage_c.py](../stage_c.py)、[合并验收](kaggle-8b/合并验收.txt) |
| 120/50 训练和 20 类选版 | [c8_upgrade.py](../c8_upgrade.py)、[c8_visual_breadth.py](../c8_visual_breadth.py)、[选版报告](kaggle-8b/覆盖与选版.md) |

210 道问答包含 60 道单字段、50 道双商品、40 道单位/表格/图表、30 道引用证据和 30 道缺失拒答。原 5 类各 20 张，另有 15 类各 5 张，合计 20 类 175 张。原照片集含一张真图的受控失焦版本。固定证据用于观察 Reader 的变化，日常检索质量由 B 的全库对照衡量。

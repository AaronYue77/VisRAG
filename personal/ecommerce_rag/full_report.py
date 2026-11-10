"""Produce reports from completed frozen full-catalog experiments, without tuning."""
import ast
import json
from pathlib import Path
import numpy as np
from full_catalog import ROOT
from full_protocol import EVAL
from common import WORKSPACE,write_json,read_jsonl,operation,ISSUES,timestamp


def main():
    maintenance=WORKSPACE/'reports/maintenance/abo-full-20261108'
    record=WORKSPACE/'记录';record.mkdir(exist_ok=True)
    report_root=WORKSPACE/'reports/ecommerce_rag'
    integrity=json.loads((maintenance/'integrity.json').read_text('utf-8'))
    manifest=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    selected=json.loads((EVAL/'selection.json').read_text('utf-8'))
    modes=['bm25','dense','hybrid','rerank']
    summaries={(s,m):json.loads((report_root/f'abo-full-v2-ret-{s}-{m}/summary.json').read_text('utf-8')) for s in ['dev','test'] for m in modes}
    qa={(s,a):json.loads((report_root/f'abo-full-v2-qa-{s}-{a.lower()}-r2/summary.json').read_text('utf-8')) for s in ['dev','test'] for a in ['A','B']}
    checks={kind:json.loads((report_root/f'abo-full-{kind}-acceptance-v1/checks.json').read_text('utf-8')) for kind in ['api','model']}
    assert all(x['passed'] for rows in checks.values() for x in rows)
    assert all(s['questions']==80 for s in summaries.values())
    assert all(s['questions']==114 and s['runtime_errors']==0 for s in qa.values())
    rows={m:read_jsonl(report_root/f'abo-full-v2-ret-test-{m}/results.jsonl') for m in modes}
    families=sorted({r['reference_family'] for r in rows['bm25']})
    rng=np.random.default_rng(20261108);resamples=rng.integers(0,len(families),(5000,len(families)))
    vectors={m:np.array([np.mean([r['metrics']['family_hit_at_5'] for r in rr if r['reference_family']==f]) for f in families]) for m,rr in rows.items()}
    ci={m:{'mean':float(v.mean()),'percentile_95_interval':np.quantile(v[resamples].mean(1),[.025,.975]).tolist(),
           'paired_difference_vs_bm25':float((v-vectors['bm25']).mean()),
           'paired_difference_95_interval':np.quantile((v-vectors['bm25'])[resamples].mean(1),[.025,.975]).tolist()} for m,v in vectors.items()}
    ci['method']={'clusters':len(families),'bootstrap_replicates':5000,'seed':20261108,
       'caution':'Exploratory paired source-family cluster bootstrap. No population probability sample; English/image-available/title-template selection, incomplete relevance labels and potential related-name dependence limit inference. Not online A/B.'}
    write_json(maintenance/'统计对照.json',ci)
    stats='全量检索冻结测试：按原始关联家族聚类，配对Bootstrap，5000次，seed=20261108。\n每组80个名称查询、40个原始关联家族；报告Hit@5差值及区间，不据测试调参。\n'+json.dumps(ci,ensure_ascii=False,indent=2)
    (maintenance/'统计对照与局限.txt').write_text(stats,'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    x=np.arange(4)
    for offset,split,label in [(-.18,'dev','开发集'),(.18,'test','冻结测试')]:
        vals=[summaries[(split,m)]['family_hit_at_5'] for m in modes]
        bars=axes[0].bar(x+offset,vals,.36,label=label)
        axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[0].set(xticks=x,xticklabels=modes,ylim=(0,1.08),ylabel='关联家族 Hit@5',title='全147,702条中全局召回；每划分80个模板查询')
    axes[0].legend();axes[0].grid(axis='y',alpha=.15)
    top=list(manifest['product_types'].items())[:12]
    axes[1].barh([t for t,n in top][::-1],[n for t,n in top][::-1],color='#387aa5')
    axes[1].set(xlabel='完整保留的商品记录数',title='576个原始类目中的前12类')
    for suffix in ['png','svg']:fig.savefig(maintenance/f'全量检索与类目分布.{suffix}',dpi=160)
    plt.close(fig)
    lines=['A、B阶段全量ABO升级完成与验收报告（2026-11-08）','='*68,
     '一、这次具体完成了什么',
     '已从官方本地abo-listings.tar逐条导入147,702条完整商品元数据；不是把24个椅子改个显示数量。',
     f'包含145,615个不同商品编号、576种原始product_type、32个来源站点。复合身份是ASIN@domain，不能只按ASIN误删站点记录。',
     'A：全部记录进入SQLite FTS5 BM25；B：全部147,702条生成384维多语言E5语义向量，并接通dense、RRF混合、独立交叉重排四模式。',
     f'开发集冻结选择：{selected["selected_mode"]}。允许简单方案胜出；没有为了“技术多”强行默认更差的模式。',
     '默认A、B启动器均已切到full；旧24商品/73页和旧冻结题集、代码、结果保留，可显式返回sample/enhanced/baseline。',
     '', '二、完整数据与图片/PDF的区别',
     '数据原文、缺失字段、来源成员/行号、源SHA全部保留。数据库147702行和FTS147702行已与官方源逐条核对。',
     '398212是原图路径/宽高的图片元数据映射数，不是本次下载的原图数量。575条缺可解析主图的商品仍保留，不能假称都有图片。',
     f'当前实际缓存原图{integrity["cached_photo_files"]}张、按需商品卡PDF{integrity["cached_pdfs"]}份；80个抽评商品固定为238页。后续新商品搜索可直接查全库，问具体事实时才下载官方原图并生成PDF。',
     '商品卡是官方真实商品元数据与原图派生的自建应用PDF，包含属性、照片和原始文案；不是Amazon官方说明书，也不是官方问答基准。',
     '没有预生成147702份PDF，也没有下载全部约110GB原图。这样保留全量检索，同时适合16GB内存/8GB显存电脑。',
     f'全量目录abo-full-v1当前占{integrity["full_v1_directory_gib"]:.3f}GiB，含旧失败向量及新增缓存；E5权重额外约0.44GiB，原有Qwen/VisRAG/环境不重复下载。',
     'E5代表文本最大512 tokens，4014条命中截断上限；原始JSON和稀疏索引继续保留完整字段，不宣称语义向量无损。',
     '', '三、应用链路与路由',
     '输入 → 问候/目录/业务边界 → 全量名称或品牌检索 → 选择ASIN@domain → 原始字段+官方图片派生PDF → 原生文字选页 → 本地Qwen读取页面图片 → 引用校验 → JSONL/TXT记录。',
     '全量元数据生成的PDF已有文字层，因此这条主链无需OCR；原A原生PDF/必要OCR能力仍在旧入口保留。',
     '同一ASIN多站点时先澄清；商品未选时不猜型号；每轮重新取事实证据，旧回答不作为事实；新搜索清空旧商品上下文。',
     '实时价格、库存、订单和退款需求进入requires_tool边界，但不执行任何业务工具；没有把历史文案当成现行店铺售后承诺。',
     'B的-Visual为最多6个全局文字候选的真实VisRAG原图重排，属于候选内条件视觉检索。无法找回文字候选之外的图片，不是全库147702张图视觉向量索引。',
     '-Retry最多一次公开词语扩展/换页，默认关闭；不是自主反思或循环Agent。HTTP仅监听127.0.0.1，有输入约束、会话隔离、容量/并发限制。',
     '', '四、检索对照（全量语料，抽样评测）',
     '新评测20类、80个商品、80个关联家族；每类2个开发/2个测试，旧24商品排除。开发/测试按共享主图或品牌型号的关联组隔离。',
     '每划分80个完整/截短商品名查询，全部在147702条中检索，不用SKU、类目或站点缩小检索范围。',
     '关联家族Hit@5是Top5是否含同一原图/型号关联组；exactHit@5才是具体ASIN@domain命中。MRR在前5内计算，不能把单目标Hit直接称完整相关集合Recall。',
     '模式       开发家族Hit@5  测试家族Hit@5  测试exactHit@5  测试MRR@5  测试中位秒']
    for m in modes:
        d=summaries[('dev',m)];s=summaries[('test',m)]
        lines.append(f'{m:<10} {d["family_hit_at_5"]:>12.4f} {s["family_hit_at_5"]:>15.4f} {s["exact_hit_at_5"]:>15.4f} {s["family_mrr_at_5"]:>11.4f} {s["retrieval_p50_seconds"]:>11.3f}')
    lines += ['方法在开发集选定后再运行测试，没有根据测试回调权重、阈值、Top-k或默认模式。四模式共640条逐题记录。',
     '查询来自源商品英文名称模板，抽评要求英文名称、可解析主图等条件，因此不代表全部576类、全部语言、自然口语和真正看图问题的准确率。',
     f'原图/型号关联组隔离仍不能保证所有语义近重复被识别；相同字面标题跨划分诊断数{integrity["same_literal_title_cross_split_count"]}。商品变体和未穷举相关标签可能影响指标，未据此重新标注或抬高成绩。',
     'Bootstrap区间在维护目录“统计对照与局限.txt”；它是探索性配对组重采样，不是线上流量A/B或总体成功率保证。',
     '耗时包含首次CPU模型加载/缓存差异，实验间有其他本地检查并行，不作严格性能基准或吞吐承诺。',
     '', '五、VLM问答与接口验收',
     '每划分114题：80道可回答字段题、10道缺字段拒答、24道实时业务/现行政策等边界题。每阶段共228题，A/B四组共456条评测记录。',
     '字段题输入明确的商品编号和站点，再选该商品的证据页；与不带编号的全局搜索是两个任务。答案按源字段子串/数值单位及引用页判定，不是完整语义蕴含人工评分。']
    for (s,a),q in qa.items():
        lines.append(f'{s}/{a}：字段正确率{q["answer_accuracy"]:.1%}，证据对齐{q["evidence_aligned_accuracy"]:.1%}，缺字段拒答{q["unknown_refusal"]:.1%}，边界{q["boundary_accuracy"]:.1%}；运行错误{q["runtime_errors"]}；新模型预测{q["actual_new_predictions"]}，同输入预测缓存{q["generation_cache_hits"]}。')
    lines += ['A/B字段题使用相同证据和同一Reader，相同输入明确复用已保存预测。实际新VLM字段预测160次，B复用160次；这些对照不能用于宣称B提升生成准确率。',
     '原字段模板容易，不足以证明一般视觉能力；图上细节、多商品自然比较、长文档图表、英文以外语言需另建独立人工题集。',
     f'14项功能单元检查通过；真实HTTP元数据验收{sum(x["passed"] for x in checks["api"])}/{len(checks["api"])}，真实模型/条件视觉验收{sum(x["passed"] for x in checks["model"])}/{len(checks["model"])}。',
     '模型接口验收额外取80个抽评商品之外的AUTO_OIL（B07C5PRQHN@amazon.ca），实际Qwen读取商品品牌并返回引用；真实背包图片经原VisRAG独立环境重排，随后释放/重载Qwen，避免8GB显存同时驻留。',
     '', '六、真实问题、处理与剩余短板',
     'ISSUE-117：首个BGE配置的全局开发Hit@5为0，保留失败向量与结果。CPU/GPU同文档向量余弦0.99999964，排除主要索引错行/精度错误；改用固定版本多语言E5并修复误删普通10字符单词的SKU正则。两项同时变化，不能把全部收益单独归因给换模型。',
     '初次A开发字段题80次缺version运行失败：保留原目录abo-full-v2-qa-dev-a；在新全量页面适配层补充Reader必需version/store_id，新增契约检查，使用-r2新目录重新完成；同编号问题日志追加已解决验证。',
     '数据库连接ResourceWarning：增加显式close；最终14项测试在ResourceWarning作为错误的条件下通过。',
     'Windows缺少causal_conv1d/flash-linear-attention使用正确但较慢的PyTorch回退；未声称安装了Linux优化内核。原模型版本不强行升级，下载均固定revision并验SHA。',
     '重排与融合未超过当前BM25默认；保留退化和耗时。源分类存在噪声（如GROCERY下出现洗手液），没有静默改写官方字段。',
     'ISSUE-198（未解决质量短板）：测试品牌Amazonベーシック(AmazonBasics)被VLM转录成Amazonベースック(AmazonBasics)，引用正确但日文单字错误。按原规则计错，保留79/80；未修改测试或Reader抬分。',
     '缺图/网络下载失败会明确记录图片资产状态，仅使用真实可用证据，不补造产品图；未对全部147702商品逐一做VLM回答验收。',
     '', '七、文件位置与历史保护',
     f'全量库：{ROOT}',f'固定评测：{EVAL}',
     f'历史保护：{integrity["protected_historical_files"]}个既有文件SHA核对通过，变化0；80个固定评测PDF SHA不变。',
     '已有A/B的24商品样本继续是历史回归版本，不会混写成全量成绩；本次没有提交或推送Git，保留用户原有未提交改动。',
     '启动入口和每次TXT位置见“全量ABO启动与记录位置_20261108.txt”；面试代码对应见“全量ABO面试覆盖与代码对应_20261108.txt”。',
     'C微调、订单/退款执行、多Agent、长期用户记忆、全库原图向量均未实施，不能写入已完成简历。',
     '官方来源：https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html',
     '保留ABO源README/LICENSE及每条原图URL、源行号；数据来自官方公开快照，不代表实时Amazon商品目录。']
    completion=record/'A_B阶段全量ABO升级完成与验收报告_20261108.txt'
    completion.write_text('\n'.join(lines)+'\n','utf-8')
    launch='''全量ABO启动与TXT记录位置（2026-11-08）
====================================================================
一、在哪运行
打开Windows PowerShell，输入：
cd 'E:\\大模型项目自学E\\doc-vqa-assistant'

二、检查与连续聊天
.\\scripts\\run_stage_a.ps1 -Command check
.\\scripts\\run_stage_a.ps1 -Command chat
.\\scripts\\run_stage_b.ps1 -Command check
.\\scripts\\run_stage_b.ps1 -Command chat

现在默认都是full，检查应显示records=147702、types=576、mode=bm25。
在“你：”处输入以下对话内容，不能把这些中文当PowerShell命令：
商品列表
搜索 黑色双肩包
第一个
品牌是什么？
重置
搜索 running shoes
第一个
商品资料里记录的颜色是什么？
退出

商品列表只展示前25类；完整576类可在HTTP /categories或manifest中查看。
选编号时同时保留列表给出的站点，避免跨站点混淆。比如单次真实资料查询：
.\\scripts\\run_stage_b.ps1 -Command ask -Question 'B07C5PRQHN@amazon.ca 的品牌是什么？'
先不调用模型、仅全量检索：
.\\scripts\\run_stage_b.ps1 -Command search -Question 'running shoes'

三、可选B对照（已实际构建全量语义索引）
.\\scripts\\run_stage_b.ps1 -Command chat -Mode dense
.\\scripts\\run_stage_b.ps1 -Command chat -Mode hybrid
.\\scripts\\run_stage_b.ps1 -Command chat -Mode rerank
.\\scripts\\run_stage_b.ps1 -Command chat -Visual
-Visual只重排最多6个文字候选图片，首次下载/VisRAG加载较慢；不能叫全库图片索引。
-Retry最多一次查询扩展，默认不开，不是Agent。原图联网按需下载到E盘，模型权重本地固定。
HTTP服务：.\\scripts\\run_stage_b.ps1 -Command serve -Port 8768
只监听本机http://127.0.0.1:8768；GET /health、/categories，POST /ask、/search。

四、回到历史样本（不覆盖旧数据）
.\\scripts\\run_stage_a.ps1 -Mode enhanced -Command chat
.\\scripts\\run_stage_a.ps1 -Mode baseline -Command chat
.\\scripts\\run_stage_b.ps1 -Corpus sample -Command chat
旧样本仅24商品/73页；以前的旧B数值与新全量成绩不可直接混比。

五、TXT具体保存位置
每次聊天启动会直接显示实际目录；每个会话均有：
E:\\大模型项目自学E\\doc-vqa-assistant\\reports\\ecommerce_rag\\abo-full-<日期时间>-<会话号>\\对话记录.txt
同目录的逐题结果.txt、results.jsonl、config.json和session.json。
逐题结果.txt含问题ID、问题、参考答案、参考证据页、召回页及排名、模型答案、引用、检索/生成耗时、模型/配置版本。
交互题无人工标准答案时明确写“交互未标注”，不会伪造参考答案。
全局商品检索只有元数据记录时明确标metadata，不能伪造PDF物理页码；模型问答才记录实际PDF页号。

固定测试逐题结果：
reports\\ecommerce_rag\\abo-full-v2-ret-test-bm25\\逐题结果.txt
reports\\ecommerce_rag\\abo-full-v2-qa-test-a-r2\\逐题结果.txt
reports\\ecommerce_rag\\abo-full-v2-qa-test-b-r2\\逐题结果.txt
其余三模式检索测试在ret-test-dense、hybrid、rerank相同文件名。
逐题总表：记录\\逐题结果汇总.txt（含历史/失败，判断正式成绩请用本次-r2目录）。
操作方法、结果和验证：upstream\\VisRAG\\personal\\a_baseline\\docs\\操作方法与结果记录.txt
问题、解决和修复验证：upstream\\VisRAG\\personal\\a_baseline\\docs\\问题与解决记录.txt
原错误不删除，修复后按同编号追加。首次BGE失败和首次Reader失败目录保留。
完整性核验、统计与图：reports\\maintenance\\abo-full-20261108\\
'''
    launch_path=record/'全量ABO启动与记录位置_20261108.txt';launch_path.write_text(launch,'utf-8')
    mapping='''全量ABO版：A/B面试覆盖与代码对应（2026-11-08）
====================================================================
定位：AI应用开发。个人改进是全量可追溯数据工程、可对照检索、跨商品证据、应用路由与工程验收，不宣称发明底座算法。

1. 数据从哪里来、到底有多少、如何证明不是样本展示？
方法：full_catalog.py/source_records/build；官方147702条，lossless raw_json、复合身份、FTS和dense行号一一对应。
证据：维护目录/全量数据与索引完整性核验.txt。逐条源JSON+SHA核验，不只读manifest。

2. 为什么要分块、怎么保持图文与引用对应？
方法：full_cards.py/evidence_fields/ensure_card/choose_pages；结构化属性页、真实照片页、分段文案页，保留doc_id/page_id/pdf_sha256/来源行。
范围：全量元数据派生PDF有原生文字，不必OCR；原始制造商PDF版面复杂解析不是本次新增工作。旧A有PDF/OCR基线，不能冒称本次全库用了OCR。

3. 稀疏与稠密区别、为什么BM25、如何对齐检索对象？
方法：full_retrieval.py；SQLite FTS5；full_encoder.py/mean pooling+query/passages前缀+L2；147702x384 E5向量与products.id固定对应。
证据：四模式开发/测试，BM25默认，负收益也保留。不能把文字语义向量说成图文统一视觉空间。

4. 混合分数怎么融合、为何重排、Top-k过大有什么问题？
方法：full_retrieval.py/RRF c=60；候选20独立交叉重排，输出5；固定模型/预算，不直接相加不同比例原始分数。
证据：四组正式检索结果和耗时。重排没有净增益即不默认；没有用测试去重新找参数。

5. 多模态在哪里、视觉检索和生成有什么区别？
方法：full_app.py/EvidenceReader复用原Reader，实际Qwen读商品页面图；full_visual.py+visrag_repro/full_candidates.py用独立原VisRAG环境编码原图页和问题。
边界：-Visual只在6个文字候选内，未建立全库原图向量；无法恢复候选外商品。模型HTTP验收确实调用过这两个模型，按顺序释放显存。

6. 路由、意图识别、会话状态、失败兜底怎样设计？
方法：full_app.py/FullSession.dispatch：问候、类目、搜索、选择、事实、缺字段、跨站点澄清、实时业务边界；规则可解释，非训练分类器。
状态：会话商品与候选、30分钟TTL、64会话上限；历史回答不当事实，不称长期用户记忆。
证据：真实HTTP会话隔离、冲突身份、非法字段、未知商品、实时库存/现行政策验收。

7. 如何降低幻觉、Prompt与引用如何验证？
方法：固定Reader证据约束/结构化结果、引用白名单，字段缺失优先拒答，实时资料不能从旧快照推测；页面契约version检查。
证据：228题/阶段，80个开发+80个测试新VLM字段预测，A/B相同输入缓存有明确标记。
限制：字段模板正确率不等于通用视觉准确率；基于源值的自动判分不是完整人工语义审核。

8. 如何评测、判断优化真的有效、如何定位bad case？
方法：full_protocol.py按图/型号家族划分；full_evaluate.py先dev选择再test；full_report.py配对家族Bootstrap、类目宏平均、逐阶段耗时与来源版本。
证据：全量检索640条+字段/拒答/边界456条；失败BGE和Reader错误保留；不是在线A/B，也不是147702道问题都答过。

9. 本地8GB显存如何运行、依赖隔离、异步和并发怎样处理？
方法：离线全量向量、CPU查询/交叉重排、Qwen与VisRAG顺序驻留；两个环境保留旧依赖；ThreadingHTTPServer+2入口槽+模型串行锁。
边界：未做生产负载测试，不宣称大并发；GPU生成是计算密集，网络下载是I/O，不能笼统说整个RAG只有一种瓶颈。

10. 知识更新、索引版本和工程复现？
方法：固定原始归档SHA、模型revision、配置/题集SHA、源文件快照；新增full目录而不是覆盖旧样本。
边界：旧B样本的增量更新实验保留，本次全量库版本不可变；全量原子增量更新尚未实现，不可把样本实验说成全量生产验收。

留给C/后续：自然口语/看图难题、必要时轻量微调、独立盲测；电商工具Agent再做Function Calling/MCP/工具参数校验、订单授权、流程和记忆。
不为覆盖算法面经强加PPO/GRPO/OPD/MoE训练；这些未实施，不能从规则路由或独立模型推断出来。
Transformer/注意力/LoRA基础、数据结构和手写题仍需单独学习；项目能提供场景和证据，不能替代基础知识。
'''
    mapping_path=record/'全量ABO面试覆盖与代码对应_20261108.txt';mapping_path.write_text(mapping,'utf-8')
    (ROOT/'ATTRIBUTION.txt').write_text('数据来自Amazon Berkeley Objects (ABO)官方公开元数据和原图。\n官方入口：https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html\n本目录保留源README/LICENSE，商品库为逐条无损副本及派生索引；按需PDF是本项目生成，不是制造商原PDF。\n每条原文保存来源归档成员/行号与SHA；每张缓存图有官方URL、image_id、SHA。使用/分发须保留原ABO许可和署名。\n官方题集并未用于本次成绩：20类抽样QA和名称查询均为本项目自建模板，不能写成ABO官方问答分数。\n','utf-8')
    for path in Path(__file__).parent.glob('full_*.py'):ast.parse(path.read_text('utf-8'))
    with ISSUES.open('a',encoding='utf-8') as f:
        f.write(f'\n[{timestamp()}] ISSUE-117（全量E5开发/冻结测试完成更新）\n处理：保留原BGE失败，新增固定多语言E5全147702条，修复误把普通10字符单词当商品ID的正则；先开发后测试，无测试调参。\n验证：E5开发家族Hit@5={summaries[("dev","dense")]["family_hit_at_5"]:.4f}，测试={summaries[("test","dense")]["family_hit_at_5"]:.4f}；全量向量SHA/行号/归一化通过；默认由开发集选择BM25。\n状态：已解决检索异常，质量局限保留（两项同时改变，不能单独归因换模型；E5未超过BM25）。\n')
    operation('全量A/B升级正式报告与面试映射','读取已完成四模式开发/冻结测试及四组Reader验收，配对家族Bootstrap5000次，生成TXT/图与来源声明。',
       '147702条全量库/向量，576类32站点；默认BM25；保留失败、限定视觉候选范围、模板和缓存边界。',
       [completion,launch_path,mapping_path,maintenance/'统计对照与局限.txt',maintenance/'全量检索与类目分布.png'],
       '640条检索、456条QA、14项单元与真实HTTP/模型验收完成；旧文件保护SHA核验不变。')
    print(json.dumps({'completion':str(completion),'selected':selected['selected_mode'],'qa':list(qa.values()),'integrity':integrity},ensure_ascii=False))


if __name__=='__main__':main()

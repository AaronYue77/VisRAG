"""Write the B handoff index without replacing historical A documentation."""
import ast
from pathlib import Path
import b_protocol
from common import WORKSPACE,operation,issue,ISSUES,timestamp


def main():
    code=Path(__file__).resolve().parent;records=WORKSPACE/'记录'
    launch=r'''项目：电商多模态RAG - B阶段启动与记录位置
##############################################################################################################
当前B完成检索对照及本机应用接入。默认BGE来自开发集六模式选择；不宣称超过A增强版的主题规范化检索。A入口保留。
B是带范围路由的检索问答工作流，有限重试不是自主Agent。C微调和后续电商工具另做。
——————————————————————————————————————————————————————————————————
1. 在Windows PowerShell先进入项目工作区：
cd 'E:\大模型项目自学E\doc-vqa-assistant'

2. 完整性检查（不加载GPU）：
.\scripts\run_stage_b.ps1 -Command check

3. 连续聊天：
.\scripts\run_stage_b.ps1 -Command chat
看到“你：”后输入：商品列表
输入：选择 B07124WN69
输入：它的宽度是多少？请保留原单位。
输入：它的重量是多少？
输入：退出
已在Windows PowerShell 5.1真实执行并正常退出；脚本ASCII避免旧版PowerShell将无BOM中文误解析。

4. 单次查询：
.\scripts\run_stage_b.ps1 -Command ask -Question 'B07124WN69 的宽度是多少？' -Sku B07124WN69

5. 可选检索模式：
-Mode bm25 / bge / text_rrf / visual / multimodal_rrf / rerank
默认auto读取冻结选择。视觉模式新问题需分时加载VisRAG和Qwen，冷启动较慢；问题向量可缓存。
-Retry启用最多一次中英词项扩展；相同候选不再读图。开发修复1题，回归测试无净收益，因此默认不开。
重排当前明显退化，保留作对照，不建议作为默认。

6. 本机HTTP：
.\scripts\run_stage_b.ps1 -Command serve -Port 8767
GET http://127.0.0.1:8767/health
POST http://127.0.0.1:8767/ask
请求：{"question":"宽度是多少？","request":{"skus":["B07124WN69"]}}
继续对话：{"question":"重量呢？","session_id":"上一步返回的ID"}
返回stage、retrieval_mode、record_txt、conversation_txt。仅本机接口，未云端部署；库存/订单/退款仍说明工具边界。

7. 复查已完成静态对照：
.\scripts\run_stage_b.ps1 -Command dev
.\scripts\run_stage_b.ps1 -Command test
完整同名运行会跳过，不覆盖。新实验另建协议版本与运行名，不修改冻结测试或拿它调参。

8. 主要位置：
data/ecommerce_rag/abo-dialogue-v2：24商品73页，官方ABO历史属性/原图派生PDF，不是厂家手册。
data/ecommerce_rag/b-evaluation-v1：200题自建标签，开发/回归测试各100，旧A映射题不能称全新独立测试。
data/ecommerce_rag/b-index-v1：文字向量、检索计划、预测缓存。
data/visrag_repro/b-abo73-pages-v1：真实视觉页面向量与页码映射。
.venv：Qwen/BGE/重排；.venv-visrag：原版VisRAG兼容环境。全部保存在E盘。
scripts/run_stage_a.ps1：原A独立入口。原48页/152题及现有A源码、历史报告保留。
——————————————————————————————————————————————————————————————————
TXT绝对路径：
'''
    paths=[records/'B阶段完成情况与验收报告.txt',records/'B阶段面试覆盖与取舍_20261104.txt',
       code/'docs/B阶段链路与代码地图.txt',WORKSPACE/'reports/ecommerce_rag/b-test-bge-v1/逐题结果.txt',
       WORKSPACE/'reports/ecommerce_rag/b-test-bge-v1/缓存与有限重试记录.txt',
       WORKSPACE/'reports/ecommerce_rag/b-test-a-topic-reference-v1/辅助基线说明.txt',
       WORKSPACE/'reports/ecommerce_rag/b-dev-oracle-v1/正确证据诊断.txt',
       WORKSPACE/'reports/ecommerce_rag/b-app-acceptance-v1/应用验收记录.txt',
       WORKSPACE/'reports/ecommerce_rag/b-visual-live-acceptance-v1/应用验收记录.txt',
       WORKSPACE/'reports/maintenance/b-stage-20261104/PowerShell5真实连续聊天.txt',
       records/'逐题结果汇总.txt',code.parent/'a_baseline/docs/操作方法与结果记录.txt',ISSUES]
    launch+='\n'.join(str(p) for p in paths)+'\n'
    launch+='''
每次会话打印实际路径：reports/ecommerce_rag/abo-dialogue-<时间>-<随机ID>/对话记录.txt、逐题结果.txt、B阶段检索与重试轨迹.txt、results.jsonl。
静态对照：b-dev-模式-v1或b-test-模式-v1下的逐题结果.txt、运行报告.txt、缓存与有限重试记录.txt、results.jsonl。
逐题TXT有问题/参考答案/证据页/召回排名/模型输出/引用/检索生成耗时/模型revision/配置SHA。普通聊天没有人工标签会标未标注，不伪造参考答案。
静态固定JSON Reader与交互原文摘录Reader分别验证，准确率不能混用。
'''
    (records/'B阶段启动与记录位置.txt').write_text(launch,'utf-8')
    names={
     'b_protocol.py':['build','rewrite','query_key'],
     'b_retrieval.py':['build_text','CrossEncoder','BRetriever','dense_query','visual_scores','reserve','search'],
     'b_evaluate.py':['prepare','CachedReader','run','summary_for','freeze'],
     'b_app.py':['BDialogueRetriever','BSession','_answer_product','_save','BApplication','b_handler_for'],
     'b_incremental.py':['ChunkCache','verify'],'b_diagnostics.py':['budget','oracle'],
     'b_report.py':['paired','main'],'b_verify_app.py':['main'],'b_a_reference.py':['run']}
    lines=['项目：B阶段实际链路与代码地图','#'*100,
     'PDF -> 复用A解析/OCR文字块及原页码 -> 稀疏/文字向量/页面视觉三路 -> 型号/站点过滤 -> 可选RRF -> 可选交叉重排 -> 固定1/2页预算 -> VLM -> 引用/拒答 -> JSONL及TXT。',
     '范围路由先挡实时/缺编号；A管理商品上下文和语义分流。RRF按同一个page_id融合，不跨空间混算512/2304维向量。',
     '证据不足最多加一次词项扩展；候选不变停止。不是ReAct规划，没有MCP业务工具。',
     '静态为stage_a.Reader；交互为dialogue_model.LocalDialogueModel.read。摘录及冲突保护保留，协议分别验证。','']
    for file,selected in names.items():
        p=code/file;lines.append(str(p))
        for node in ast.walk(ast.parse(p.read_text('utf-8'))):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in selected:lines.append(f'  第{node.lineno}行：{node.name}')
        lines.append('')
    lines+=['视觉页面编码使用personal/visrag_repro/retrieve.py原始encode；b_queries.py编码公开问题，b_one_query.py为新问题启动独立环境。',
     '个人实现：受控协议、块/页映射与缓存、融合/重排适配、有限重试、评测统计、强基线核对、应用接入和验收。预训练能力与VisRAG算法归开源作者。',
     '未完成：LoRA-SFT、订单退款库存工具、长期记忆、生产向量数据库并发增量、视频、RL/OPD/MoE训练。',
     '结果：BGE强于raw-query BM25，A主题规范化参考更强；重排退化，重试测试无净收益。']
    (code/'docs/B阶段链路与代码地图.txt').write_text('\n'.join(lines)+'\n','utf-8')
    coverage=code/'docs/B阶段面试覆盖与取舍.txt';coverage_text=coverage.read_text('utf-8')
    heading='五、2026-11-04完成后的实际面试证据'
    if heading not in coverage_text:
        with coverage.open('a',encoding='utf-8') as f:f.write('''
——————————————————————————————————————————————————————————————————
五、2026-11-04完成后的实际面试证据
- 14组主对照1400条预测；后补A规范化参考200条、开发正确页诊断7条分别报告。
- 测试80题可回答：raw-query BM2562/80、BGE74/80、文字RRF69/80、VisRAG69/80、多模态RRF71/80、重排35/80、有限重试74/80（答案+参考页对齐）。
- A主题规范化参考77/80，是后补分析、相同静态Reader，不是整个A应用。不能宣传BGE超过现有A。
- BGE未知题11/12按协议拒答，边界8/8。info not available却action=answer错误保留。
- 28/28聊天/API检查、6/6视觉新查询检查、PowerShell5.1真实聊天正常结束。不是生产部署或任意语义准确率保证。
- 7道开发失败给正确页后6道修复；1道无扶手图仍回答Yes。多数先处理检索，剩余视觉读取作为C诊断起点。
- B最值得讲：比较方法后发现强基线规范化有价值；新增组件未必更好，用证据取舍、记录退化及成本。
- B没有另做完整分辨率/局部裁剪消融；长上下文位置、线上压测、HyDE、生产增量仍未完成或按需选做。
- 可视化及探索性商品组配对统计在b-stage-acceptance-v1。小样本重复模板和旧标签限制需主动说明。
''')
    (records/'B阶段面试覆盖与取舍_20261104.txt').write_text(coverage.read_text('utf-8'),'utf-8')
    readme=code/'README.md';original=readme.read_text('utf-8')
    if not original.startswith('# B阶段检索对照'):
        intro='''# B阶段检索对照与独立应用入口（2026-11-04）

B完成 BM25、BGE、文字RRF、VisRAG页面视觉、多模态RRF、独立交叉重排和最多一次查询扩展。200道自建回归题，按开发选择、全部失败保留。BGE优于本次原问题BM25，后补A主题规范化参考更强，**不宣称超过现有A增强应用**。A启动器保留。

```powershell
.\\scripts\\run_stage_b.ps1 -Command check
.\\scripts\\run_stage_b.ps1 -Command chat
```

[面试覆盖与取舍](docs/B阶段面试覆盖与取舍.txt)、[链路与代码地图](docs/B阶段链路与代码地图.txt)、[完成与验收](../../../../记录/B阶段完成情况与验收报告.txt)、[启动与TXT位置](../../../../记录/B阶段启动与记录位置.txt)。静态与交互Reader分别验证，不能混用准确率。尚未实施C微调或电商工具Agent。

---
以下A说明与历史保留。

'''
        readme.write_text(intro+original,'utf-8')
    plan=WORKSPACE.parent/'总共要搞清楚的问题_电商RAG新版.txt'
    if 'B阶段实际完成更新（2026-11-04）' not in plan.read_text('utf-8'):
        with plan.open('a',encoding='utf-8') as f:f.write('''
——————————————————————————————————————————————————————————————————
B阶段实际完成更新（2026-11-04）：
同24商品73页，6种检索和有限一次改写，200题开发/回归测试。固定VLM、提示、像素、单商品1页/双商品2页。
测试80题答案+证据：原问题BM25 62、BGE74、文字RRF69、VisRAG69、多模态RRF71、重排35、有限重试74。
后补A主题规范化参考77/80；B尚未证明超过A增强版。BGE只是在预声明六组里按开发选择，不掩盖强基线或退化。
向量、模型版本/缓存、分时GPU、逐题引用/代价及统计均保存；聊天/API28项、视觉6项通过，PS5.1实际聊天正常结束。
原A源码、题集、历史报告保留；B另有run_stage_b.ps1。C、生产云部署、工具Agent尚未实施。映射旧题是回归集，不是官方QA或全新独立测试。
逐图面试取舍：doc-vqa-assistant/记录/B阶段面试覆盖与取舍_20261104.txt。
完整实验：doc-vqa-assistant/记录/B阶段完成情况与验收报告.txt；启动和TXT：doc-vqa-assistant/记录/B阶段启动与记录位置.txt。
''')
    with ISSUES.open('a',encoding='utf-8') as f:f.write(f'\n[{timestamp()}] ISSUE-110 后续验证\n测试重排35/80、融合71/80、BGE74/80。A主题规范化参考77/80。退化保留，没有据测试重新调参。\n状态：质量短板保留，强基线归因已补充，不默认重排。\n')
    quality=issue('BGE照片召回不足、未知答复协议错误，正确页仍有视觉误读',
     '6道测试扶手题未召回主图；一个未知题info not available却action=answer。开发B07GNF6Q9L正确页仍回答Yes，人工看图确认没有独立扶手。',
     '保留全部输出，分别报告检索/读取/协议。C只从开发错误诊断，不以测试答案训练或补造缺失事实。',
     '7道开发失败正确页诊断6道修复，余1道真实读取错误。原题/模型参数未修改。见b-dev-oracle-v1/正确证据诊断.txt。',status='未解决的质量短板')
    tooling=issue('临时文档生成脚本ModuleNotFoundError: common',
     '临时脚本只加入电商目录，未初始化rag_common的Python搜索路径，未产生修改。',
     '改为可复用b_docs.py，先导入b_protocol初始化路径，再导入common。',
     '本脚本已写出启动索引、代码地图、面试更新和README。',status='已解决')
    operation('B应用适配和交付说明','新增独立入口、B health字段及TXT重试轨迹；同候选重试累计两次检索耗时，B定向9项通过；真实聊天/API/视觉和PS5.1核验。',
     '聊天/API28/28、视觉6/6、PS5.1退出0；先前163项含8项B，新增1项后总164项。更新README、原计划进度和面试映射。',
     [records/'B阶段启动与记录位置.txt',records/'B阶段面试覆盖与取舍_20261104.txt',code/'docs/B阶段链路与代码地图.txt'],
     '图表已目视检查无裁切；GNF图片确认读取错误。没有改冻结数据或推送用户A改动。',related_issues=quality+'; '+tooling)
    print('B documentation saved.')


if __name__=='__main__':main()

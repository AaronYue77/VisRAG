"""B acceptance report, paired product-cluster statistics and immutable result audit."""
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import statistics
from b_protocol import ROOT,CORPUS,MODES,INDEX
from common import WORKSPACE,read_jsonl,write_json,file_hash,operation
from stage_a import percentile

REPORTS=WORKSPACE/'reports/ecommerce_rag'


def paired(a,b,metric):
    a={r['question_id']:r for r in a if r['reference_pages']};b={r['question_id']:r for r in b if r['reference_pages']}
    assert set(a)==set(b)
    parent={r['family_key']:r['family_key'] for r in a.values()}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for r in a.values():
        for k in r.get('document_families',[]):parent[find(k)]=find(r['family_key'])
    groups={}
    for key,r in a.items():groups.setdefault(find(r['family_key']),[]).append(b[key]['metrics'][metric]-r['metrics'][metric])
    keys=sorted(groups);rng=random.Random(20261104);samples=[]
    for _ in range(3000):samples.append(statistics.mean(x for k in rng.choices(keys,k=len(keys)) for x in groups[k]))
    totals=[sum(v) for v in groups.values() if sum(v)!=0]
    p=sum(abs(sum(s*v for s,v in zip(signs,totals)))>=abs(sum(totals))-1e-12 for signs in itertools.product((-1,1),repeat=len(totals)))/(2**len(totals)) if totals else 1.0
    return {'questions':len(a),'connected_product_clusters':len(groups),'mean_difference':statistics.mean(x for v in groups.values() for x in v),
      'cluster_bootstrap_95ci':[percentile(samples,.025),percentile(samples,.975)],'cluster_sign_flip_p_unadjusted':p,
      'gains':[k for k in a if b[k]['metrics'][metric]>a[k]['metrics'][metric]],'regressions':[k for k in a if b[k]['metrics'][metric]<a[k]['metrics'][metric]],
      'assumption':'Cluster symmetry under null; small custom regression set, labels historically observed. No population/online business guarantee.'}


def main():
    out=REPORTS/'b-stage-acceptance-v1';out.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'selection.json').read_text('utf-8'))['selected_mode']
    modes=MODES+['retry'];summaries={};rows={};verification={}
    for split in ('dev','test'):
        summaries[split]={};rows[split]={}
        for mode in modes:
            run=REPORTS/f'b-{split}-{mode}-v1';items=read_jsonl(run/'results.jsonl');config=json.loads((run/'config.json').read_text('utf-8'))
            assert len(items)==100
            for row in items:
                for name in ('question_id','question','reference_answer','reference_pages','retrieved_pages','answer','citations','retrieval_seconds','generation_seconds','model_revision','config_sha256'):assert name in row
                assert row['config_sha256']==file_hash(run/'config.json')
                assert all((WORKSPACE/p).is_file() for p in row.get('image_paths',[]))
                assert set(row['citations'])<=set(p['page_id'] for p in row['retrieved_pages'])
            verification[run.name]={'records':len(items),'results_sha256':file_hash(run/'results.jsonl'),'complete':True}
            rows[split][mode]=items;summaries[split][mode]=json.loads((run/'summary.json').read_text('utf-8'))
    contrasts={}
    for name,base,other in [('selected_vs_bm25','bm25',selected),('reranker_vs_fusion','multimodal_rrf','rerank'),('retry_vs_selected',selected,'retry')]:
        contrasts[name]=paired(rows['test'][base],rows['test'][other],'evidence_aligned_fact')
    # Descriptive, related contrasts; no claim of preregistered confirmatory testing.
    ordered=sorted(contrasts,key=lambda x:contrasts[x]['cluster_sign_flip_p_unadjusted']);running=0
    for i,key in enumerate(ordered):
        running=max(running,min(1,(len(ordered)-i)*contrasts[key]['cluster_sign_flip_p_unadjusted']))
        contrasts[key]['p_holm_adjusted_3_contrasts']=running
    write_json(out/'paired-statistics.json',contrasts);write_json(out/'matrix.json',summaries);write_json(out/'verification.json',verification)
    auxiliary={split:json.loads((REPORTS/f'b-{split}-a-topic-reference-v1/summary.json').read_text('utf-8')) for split in ('dev','test')}
    write_json(out/'auxiliary-a-reference.json',auxiliary)
    lines=['项目：电商多模态RAG - B阶段完成与验收','日期：2026-11-04','#'*100,
      '一、范围与公平对照','基于既有24商品73页资料；原A48页/152题与独立对话代码保留。B预声明200题（映射152题+48自然措辞），按原商品组开发/回归测试各100题。',
      '旧A标签曾经看过，B映射题只能称回归对照。自然措辞仍共享商品和答案，不能当独立样本；本次没有新增官方数据或构造新的厂家事实。',
      '六模式：BM25、BGE、文字RRF、VisRAG-Ret页面视觉、文字/视觉RRF、独立bge-reranker-base交叉重排。另对开发集选中的模式做最多一次查询扩展/重试。',
      '每组固定相同Qwen3.5-2B、Reader提示、786432像素/页、greedy与160输出token；单商品1页、双商品2页。A旧成绩用2页/48页语料，不能直接与B数字相减；B内重新跑BM25作基线。',
      '正常检索、重排、改写不读参考答案。公开视觉问题文件仅text/hash。缓存只复用完全相同问题/请求/页面顺序/模型/提示/源代码/像素预算的预测；保留首次实测时间，明确命中标记。',
      '', '二、开发集结果与选择']
    for mode in modes:
        s=summaries['dev'][mode];lines.append(f"{mode}: Recall={s['recall_at_k']:.4f}; 答案={s['answer_accuracy']:.4f}; 证据对齐={s['evidence_aligned_accuracy']:.4f}; 实际新增Reader调用={s['actual_new_reader_calls']}")
    lines+=['开发集选择：'+selected+'；只在预声明的六组内按开发证据对齐、答案和成本选择，测试不回流选参。B独立入口默认复用该选择。',
      '这不是对已有A增强应用的优胜结论：A入口的主题规范化也是一个强检索方法。B保留单独启动器，不覆盖A。','', '三、完整回归测试结果（各100题、其中80题可回答）']
    for mode in modes:
        s=summaries['test'][mode];lines.append(f"{mode}: Recall={s['recall_at_k']:.4f}; 答案={s['answer_accuracy']:.4f}; 证据对齐={s['evidence_aligned_accuracy']:.4f}; 拒答={s['unknown_refusal']:.4f}; 路由={s['route_accuracy']:.4f}; 异常={s['runtime_errors']}")
    lines+=['', '三补、现有A主题规范化检索参考组（后补分析）',
      '该组在B六组选择后补充，以免把原问题直接检索的BM25当作现有A全部能力；不是事先六组之一，不用于回改冻结参数。',
      '调用现有A DialogueRetriever，以当前问题字面主题推断，使用同一个B静态Reader。未运行完整A语义分流，不能当整个A应用得分。']
    for split,s in auxiliary.items():lines.append(f"{split}: A主题规范化参考 证据对齐={s['evidence_aligned_accuracy']:.4f}，答案={s['answer_accuracy']:.4f}，Recall={s['recall_at_k']:.4f}")
    lines+=['A规范化参考超过BGE。只能说BGE优于本次raw-query BM25，不能说新增向量组件超过了现有A；现有A仍可作为优先使用入口。',
      '', '四、探索性配对统计',json.dumps(contrasts,ensure_ascii=False,indent=2),
      '区间按商品连接组bootstrap3000次；跨商品比较连接相关组，自然措辞不当作独立商品。p值为组符号翻转，依赖对称假设，并对本报告三组比较作Holm校正。比较不是预注册的验证性统计；小样本/历史标签可见的限制保留，不据此承诺泛化业务收益。',
      '', '五、工程与计算代价','BGE 512维文本与VisRAG 2304维图文共享空间分别检索，不跨模型相乘；按同一个page_id做RRF。向量有限/L2，页码/图片/文件SHA可核对。',
      '交叉重排器是独立query+候选文字联合编码，CPU运行，最多6候选页，文字块最大384token。当前并未训练reranker，也不把RRF叫重排。',
      'VisRAG原始3.11环境编码，Qwen在现有3.14环境；8GB显卡分时使用。实时视觉新问题需释放Reader后启动子进程，再加载问答模型，冷启动耗时不能忽略。',
      '检索计划预计算：component_compute_seconds_estimate包含首次文本/视觉问题编码和重排时间，但不含模型冷加载；warm_lookup_wall_seconds是缓存/排序墙钟。二者均不能直接当完整线上P95。',
      '同输入生成缓存只节省离线重复实验，generation_p50/p95为首次输入实测，不是命中缓存的零耗时；本机运行有负载/内核回退，不宣称严格加速倍数。',
      '注意：静态六组使用stage_a.Reader固定JSON输出；交互B复用A的原文摘录/冲突校验Reader与语义分流。因此本表不是完整聊天准确率。',
      'BGE增量缓存在独立探针中验证：292块不变全部复用，改1块只编码1块；其余291块缓存向量逐字节相同。不是生产向量库并发增量或局部视觉patch增量。',
      '', '六、真实改善与退化案例（完整在各运行TXT）']
    a={r['question_id']:r for r in rows['test']['bm25']};b={r['question_id']:r for r in rows['test'][selected]}
    for label,keys in [('改善',contrasts['selected_vs_bm25']['gains'][:5]),('退化',contrasts['selected_vs_bm25']['regressions'][:5])]:
        for key in keys:
            lines.extend([label+' '+key,'问题：'+a[key]['question'],'BM25：'+a[key]['answer']+' | '+str(a[key]['citations']),selected+'：'+b[key]['answer']+' | '+str(b[key]['citations'])])
    lines+=['重排/融合无提升甚至变差也如实保留，不采用“组件越多必定越好”的结论。下列是默认模式剩余失败：']
    for row in rows['test'][selected]:
        if row['failure_category']!='none':lines.extend([row['question_id']+' | '+row['failure_category'],'参考：'+row['reference_answer'],'模型：'+row['answer'],'原始输出：'+row['raw_output'],'实际候选：'+str([p['page_id'] for p in row['retrieved_pages']])])
    lines+=['自动标签匹配和参考页不是完整语义蕴含：来源同义表述、标题/字段冲突、照片细节、合法但不支持答案的引用仍需人工核验。',
      '单商品资料只有3至4页，任务含属性模板，不能泛化到工业长PDF/真实售后全业务。',
      '', '七、面试与C衔接','B重点：检索方案选择、稀疏/稠密/跨模态空间、融合与重排区别、候选预算、改写收益、坏例定位、版本/缓存/资源和统计。',
      '逐图取舍见记录/B阶段面试覆盖与取舍_20261104.txt。LoRA/SFT留C，工具/长期记忆/MCP留Agent，OPD/GRPO/MoE/视频不是B必做。',
      '开发正确证据诊断：7个BGE失败案例提供正确页后6个满足答案及证据；B07GNF6Q9L仍把无扶手照片回答为Yes，人工查看实际图片确认无独立扶手。不能只用自动评分确定训练数据。',
      'C先使用开发集正确证据诊断；检索错误与原始数据缺失不能靠微调补造事实；固定候选比较提示与LoRA。',
      '', '八、启动与记录','工作区PowerShell：.\\scripts\\run_stage_b.ps1 -Command chat；-Command check只检查；-Mode可选六种检索；-Retry启用至多一次检索扩展。',
      'A入口仍可独立运行；本轮不接订单/库存/交易，不云端上线，不推送用户尚未提交的A改动。',
      '每运行：reports/ecommerce_rag/b-<split>-<mode>-v1/逐题结果.txt、results.jsonl、运行报告.txt、config.json、source_snapshot。',
      '全局：记录/逐题结果汇总.txt；连续操作/问题日志仍在upstream/VisRAG/personal/a_baseline/docs。']
    lines+=['','九、真实应用验收']
    for name in ('b-app-acceptance-v1','b-visual-live-acceptance-v1'):
        checks=json.loads((REPORTS/name/'checks.json').read_text('utf-8'))
        lines.append(name+f"：{sum(c['passed'] for c in checks)}/{len(checks)}；详见该目录的应用验收记录.txt。")
    lines+=['连续聊天包含商品选择、参数单位、缺资料拒答、业务边界、重置；HTTP验证真实socket和会话隔离。视觉新查询为真实子进程编码，Reader在之后重新加载。',
      '已有A/common共163项检查通过，随后B重试计时补充测试及完整B定向9项检查通过；当前已验证总用例为164项。不是每项都对模型语义给出保证。']
    for split in ('dev','test'):
        for mode in modes:
            run=REPORTS/f'b-{split}-{mode}-v1'
            details=['B缓存与有限重试明细；生成时间保留该输入首次实测时间，不等于本次缓存命中墙钟。',
                     '旧attempts表示Reader轮次；retrieval尝试还需看rewritten_query/second_ranking，候选不变也消耗一次检索。']
            for row in rows[split][mode]:
                details.append(json.dumps({k:row.get(k) for k in ('question_id','prediction_cache_hit','prediction_cache_key','measured_prediction_wall_seconds','generation_seconds','retry_trace')},ensure_ascii=False))
            (run/'缓存与有限重试记录.txt').write_text('\n'.join(details)+'\n','utf-8')
    text='\n'.join(lines)+'\n'
    (WORKSPACE/'记录/B阶段完成情况与验收报告.txt').write_text(text,'utf-8',newline='\n');(out/'实验汇总与统计.txt').write_text(text,'utf-8',newline='\n')
    write_json(out/'failure-cases.json',[{'mode':m,**r} for m,rs in rows['test'].items() for r in rs if r['failure_category']!='none'])
    operation('汇总B阶段受控实验与统计','读取14组不可改写预测；核对逐题字段/配置/图片/引用；按商品连接组配对统计，分别报告事实/引用/召回/拒答与缓存成本。',
      '1400条管线对照记录完整；默认模式由开发选择为'+selected+'。',[WORKSPACE/'记录/B阶段完成情况与验收报告.txt',out/'实验汇总与统计.txt',out/'verification.json'],'失败原文及退化保留；没有将回归集包装成全新独立benchmark。')

if __name__=='__main__':main()

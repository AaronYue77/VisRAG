"""Development-only oracle evidence and retrieval-budget diagnostics."""
import json
import statistics
from b_protocol import ROOT,INDEX,MODES
from b_evaluate import REPORTS,CachedReader
from b_retrieval import BRetriever
from stage_a import score
from common import WORKSPACE,read_jsonl,write_json,write_jsonl,operation,file_hash,timestamp
from audit import append_question
from core import retrieval_metrics


def budget():
    qs={q['question_id']:q for q in read_jsonl(ROOT/'questions.jsonl') if q['split']=='dev' and q['reference_pages']}
    retriever=BRetriever();values=[]
    for mode in MODES:
        plans=read_jsonl(INDEX/'retrieval-plans'/(mode+'.jsonl'))
        for k in (1,2,4):
            metrics=[]
            for r in plans:
                if r['question_id'] not in qs:continue
                q=qs[r['question_id']]
                order=r['trace']['page_ranking']
                selected=retriever.reserve(order,retriever.by_id,r['plan']['skus'],k)
                metrics.append(retrieval_metrics(selected,q['reference_pages']))
            values.append({'mode':mode,'top_k':k,'questions':len(metrics),**{name:statistics.mean(v[name] for v in metrics) for name in ('hit_at_k','recall_at_k','reciprocal_rank')}})
    out=REPORTS/'b-dev-budget-v1';out.mkdir(exist_ok=False)
    write_json(out/'budget.json',values)
    (out/'Top-k检索对照.txt').write_text('只做开发集检索预算诊断，未运行额外生成；主六组仍固定单商品1页、双商品2页，不据此改测试预算。\n'+json.dumps(values,ensure_ascii=False,indent=2),'utf-8')
    operation('B开发集Top-k检索预算对照','复用真实候选顺序，对K=1/2/4计算Recall与MRR；双商品在预算够时各留一候选。','18组开发检索诊断已保存；未改变主对照配置。',[out/'Top-k检索对照.txt'],'增加K会扩大图片预算，高召回不能替代回答/延迟实测；不把检索表称生成质量对照。')


def oracle():
    selected=json.loads((ROOT/'selection.json').read_text('utf-8'))['selected_mode']
    failures=[r for r in read_jsonl(REPORTS/f'b-dev-{selected}-v1/results.jsonl') if r['reference_pages'] and r['failure_category']!='none'][:8]
    retriever=BRetriever();reader=CachedReader();name='b-dev-oracle-v1';out=REPORTS/name;out.mkdir(exist_ok=False)
    from stage_a import snapshot
    config={'diagnostic':'dev-only oracle','selected_mode':selected,'source_sha256':snapshot(out),'question_ids':[r['question_id'] for r in failures],
      'note':'Selection follows observed dev failures; not independent test or normal retrieval. Gold intentionally supplies evidence ONLY here.'}
    write_json(out/'config.json',config);rows=[]
    for q in failures:
        pages=[{**retriever.by_id[p],'rank':i,'score':0} for i,p in enumerate(q['reference_pages'],1)]
        predicted=reader.predict(q,pages)
        row={**q,**predicted,'run_name':name,'run_directory':str(out.relative_to(WORKSPACE)).replace('\\','/'),'timestamp':timestamp(),
             'config_sha256':file_hash(out/'config.json'),'source_sha256':config['source_sha256'],'diagnostic_oracle':True,
             'retrieval_seconds':0.0,'retrieval_trace':{'mode':'oracle_dev_only'},'routing':{'action':'retrieve','reason':'manual_reference_evidence_diagnosis'},
             'retrieved_pages':[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source')} for p in pages]}
        row['metrics']=score(row,q)
        row['failure_category']='none' if row['metrics']['answer_correct'] and row['metrics']['evidence_aligned_fact'] else 'reading_with_reference_evidence'
        rows.append(row);append_question(out,row)
        print(q['question_id'],q['answer'],'->',row['answer'],flush=True)
    write_jsonl(out/'results.jsonl',rows)
    text='开发集正确证据诊断：选择默认模式的开发失败，显式提供人工参考页。不能当普通检索成绩或独立泛化。\n'
    for before,after in zip(failures,rows):text+=before['question_id']+'\n原输出：'+before['answer']+'\n正确证据输出：'+after['answer']+'\n原始：'+after['raw_output']+'\n'
    (out/'正确证据诊断.txt').write_text(text,'utf-8')
    operation('B到C的正确证据诊断','只在开发失败题提供人工参考页，其他模型/提示/像素不变；保留原输出和诊断输出。',f'{len(rows)}道开发诊断，实际新增Reader调用{reader.calls}。',[out/'正确证据诊断.txt',out/'逐题结果.txt'],'原主对照预测/测试不改；正确页仍错才构成C训练候选，源数据缺失不靠训练补事实。')

if __name__=='__main__':budget();oracle()

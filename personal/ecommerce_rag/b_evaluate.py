"""Paired B matrix with frozen reader, gold-free retrieval and explicit cache costs."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
from b_protocol import ROOT,CORPUS,INDEX,MODES,SETTINGS,rewrite
from b_retrieval import BRetriever
from stage_a import Reader,score,snapshot,MODEL_REVISION,percentile
from common import WORKSPACE,read_jsonl,write_json,write_jsonl,file_hash,timestamp,operation,issue
from audit import append_question

REPORTS=WORKSPACE/'reports/ecommerce_rag'


def protocol():
    value=json.loads((ROOT/'protocol.json').read_text('utf-8'))
    assert value['questions_sha256']==file_hash(ROOT/'questions.jsonl')
    for name,digest in value['corpus_sha256'].items():assert file_hash(CORPUS/name)==digest
    return value


def prepare():
    protocol();retriever=BRetriever();folder=INDEX/'retrieval-plans';folder.mkdir(exist_ok=True)
    tasks=read_jsonl(ROOT/'questions.jsonl')
    for mode in MODES:
        path=folder/(mode+'.jsonl')
        if path.exists():
            assert len(read_jsonl(path))==len(tasks)
            continue
        rows=[]
        for i,q in enumerate(tasks):
            # The retriever receives only public inputs, never any reference labels.
            result=retriever.search(q['question'],q['request'],mode)
            rows.append({'question_id':q['question_id'],'query':q['question'],'request':q['request'],**result})
            if i%25==0:print('B plan',mode,i+1,'/',len(tasks),flush=True)
        write_jsonl(path,rows)
    operation('完成六组B候选计划','同一73页/商品过滤/1页或双商品2页预算；BM25、BGE、文本RRF、视觉、三路RRF、独立交叉重排分别生成候选。',
      f'{len(tasks)}题×6模式；候选先产生，gold不参与。',[folder],'重排仅排序融合候选；页面和查询向量分别保留，真实视觉编码非OCR文字向量。')


class CachedReader:
    def __init__(self):self.reader=None;self.calls=0
    def predict(self,q,pages):
        key=hashlib.sha256(json.dumps({'question':q['question'],'request':q['request'],
            'pages':[(p['page_id'],p['pdf_sha256']) for p in pages],'settings':{k:SETTINGS[k] for k in ('max_pixels','reader_revision','reader_prompt','max_new_tokens')},
            'reader_source_sha':file_hash(Path(__file__).parent/'stage_a.py')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        folder=INDEX/'reader-cache';folder.mkdir(exist_ok=True);file=folder/(key+'.json')
        if file.exists():return {**json.loads(file.read_text('utf-8')),'prediction_cache_hit':True,'prediction_cache_key':key,'measured_prediction_wall_seconds':0.0}
        if self.reader is None:self.reader=Reader()
        started=time.perf_counter();value=self.reader.read(q['question'],q['request'],pages,folder/'images',SETTINGS['max_pixels'])
        value.update(prediction_cache_key=key,prediction_cache_hit=False,measured_prediction_wall_seconds=time.perf_counter()-started)
        write_json(file,value);self.calls+=1
        return value


def run(split,modes=None):
    proto=protocol();modes=modes or MODES
    frozen=None
    if split=='test':
        frozen=json.loads((ROOT/'comparison-freeze.json').read_text('utf-8'))
        if frozen['settings']!=SETTINGS:raise ValueError('B settings changed after dev selection')
        if frozen['questions_sha256']!=proto['questions_sha256']:raise ValueError('B labels changed')
        if frozen['reader_source_sha256']!=file_hash(Path(__file__).parent/'stage_a.py'):raise ValueError('Reader changed after freeze')
    questions=[q for q in read_jsonl(ROOT/'questions.jsonl') if q['split']==split]
    reader=CachedReader();retriever=None
    for mode in modes:
        name=f'b-{split}-{mode}-v1';output=REPORTS/name
        if output.exists():
            if (output/'summary.json').exists():continue
            raise FileExistsError('Incomplete run preserved; repair/resume explicitly')
        output.mkdir(parents=True)
        plan_mode=mode
        if mode=='retry':
            selected=json.loads((ROOT/'selection.json').read_text('utf-8'));plan_mode=selected['selected_mode']
            if retriever is None:retriever=BRetriever()
        plans={r['question_id']:r for r in read_jsonl(INDEX/'retrieval-plans'/(plan_mode+'.jsonl'))}
        config={'schema_version':'b1','run_name':name,'split':split,'retrieval_mode':mode,'base_mode':plan_mode,'settings':SETTINGS,
            'questions_sha256':proto['questions_sha256'],'protocol_sha256':file_hash(ROOT/'protocol.json'),'source_sha256':snapshot(output),
            'reader_cache_policy':'Reuse only exact question/request/page-order/model/prompt/source/pixel inputs; predictions contain no gold. Report saved computation separately.',
            'ranking_cache_policy':'Precomputed independent retrieval plans include original component encoding/rerank times; wall lookup not called cold latency.',
            'model':'Qwen/Qwen3.5-2B','model_revision':MODEL_REVISION}
        write_json(output/'config.json',config);csha=file_hash(output/'config.json')
        shutil.copyfile(ROOT/'protocol.json',output/'input-protocol.json');shutil.copyfile(ROOT/'questions.jsonl',output/'input-questions.jsonl')
        rows=[];calls_before=reader.calls
        with (output/'results.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for i,q in enumerate(questions):
                start=time.perf_counter();found=plans[q['question_id']];pages=found['pages']
                row={**q,'run_name':name,'run_directory':str(output.relative_to(WORKSPACE)).replace('\\','/'),'timestamp':timestamp(),
                    'status':'ok','config_sha256':csha,'source_sha256':config['source_sha256'],'model':None,'model_revision':None,
                    'generation_seconds':0.0,'citations':[],'raw_output':'','retrieval_seconds':found['seconds'],'routing':found['plan'],
                    'retrieval_trace':found['trace'],'retrieved_pages':[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source')} for p in pages]}
                if found['plan']['action']=='retrieve':
                    row.update(model=config['model'],model_revision=MODEL_REVISION)
                    try:
                        first=reader.predict(q,pages);row.update(first)
                        if mode=='retry':
                            row['retry_trace']={'attempts':1,'max_retrieval_attempts':2,'stop_reason':'answer_supported' if row['action']=='answer' else 'no_safe_rewrite'}
                            new_query=rewrite(q['question'])
                            if first['action']=='not_found' and first.get('parse_ok') and new_query!=q['question']:
                                revised=retriever.search(new_query,q['request'],plan_mode)
                                revised_pages=revised['pages'];old_ids=[p['page_id'] for p in pages];new_ids=[p['page_id'] for p in revised_pages]
                                row['retry_trace'].update(rewritten_query=new_query,rewrite_method='bounded bilingual lexical expansion from current question',second_ranking=revised['trace'],first_output=first)
                                if new_ids!=old_ids:
                                    second=reader.predict(q,revised_pages);row.update(second)
                                    row['generation_seconds']=first['generation_seconds']+second['generation_seconds']
                                    row['retrieved_pages']=[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source')} for p in revised_pages]
                                    row['retry_trace'].update(attempts=2,stop_reason='answer_supported' if second['action']=='answer' else 'retry_limit_abstain')
                                else:row['retry_trace']['stop_reason']='unchanged_candidates_no_second_generation'
                                row['retrieval_seconds']+=revised['seconds']
                    except Exception as error:
                        row.update(status='error',answer='NOT_FOUND',action='not_found',answer_source='runtime_error',error=f'{type(error).__name__}: {error}')
                        row['issue_id']=issue('B题目运行失败：'+row['error'],'保留当前候选及配置。','修复后新建运行或明确恢复，不删除失败。')
                else:row.update(action=found['plan']['action'],answer=found['plan']['answer'],answer_source='rule_router')
                row['total_seconds']=time.perf_counter()-start
                # Gold fields are first read here, after the prediction has been produced.
                row['metrics']=score(row,q)
                if q['reference_pages']:
                    row['metrics']['evidence_aligned_fact']=float(bool(row['metrics']['answer_correct']) and set(q['reference_pages'])<=set(row['citations']))
                    row['metrics']['evaluation_note']='B: field/unit matching and all annotated evidence pages required for comparisons; not full free-form entailment.'
                if row['status']!='ok':failure='runtime'
                elif q['reference_pages'] and not row['metrics'].get('hit_at_k'):failure='retrieval'
                elif not row['metrics']['answer_correct']:failure='protocol' if row.get('output_guard') else 'reading_or_route'
                elif q['reference_pages'] and not row['metrics']['evidence_aligned_fact']:failure='citation'
                else:failure='none'
                row['failure_category']=failure
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();append_question(output,row);rows.append(row)
                if i%20==0 or failure!='none':print(name,i+1,'/',len(questions),q['question_id'],row['answer'][:70],failure,flush=True)
        summary=summary_for(rows);summary.update(actual_new_reader_calls=reader.calls-calls_before,mode=mode,split=split)
        write_json(output/'summary.json',summary)
        lines=['B阶段同题对照 '+name,json.dumps(summary,ensure_ascii=False,indent=2),'缓存说明：完全相同的模型输入复用结果；generation_seconds为该输入首次实测，非本次缓存命中耗时。','失败记录：']
        for r in rows:
            if r['failure_category']!='none':lines.extend([r['question_id']+' | '+r['failure_category'],'参考：'+r['reference_answer'],'输出：'+r['answer'],'引用：'+str(r['citations'])])
        (output/'运行报告.txt').write_text('\n'.join(lines),'utf-8',newline='\n')
        operation('完成B阶段'+mode+' '+split+'对照','同一73页资料、问答模型/提示/像素、单商品1页/双商品2页；只改变检索组件；相同VLM输入复用而非虚报新调用。',
            json.dumps(summary,ensure_ascii=False),[output/'逐题结果.txt',output/'运行报告.txt'],'逐题原始输出、路由、各路排序、重排耗时和缓存标记保留。')
        print(json.dumps(summary,ensure_ascii=False),flush=True)


def summary_for(rows):
    answerable=[r for r in rows if r['reference_pages']];unknown=[r for r in rows if r['field']=='unknown'];routes=[r for r in rows if r['field']=='route']
    mean=lambda rs,key:statistics.mean(r['metrics'][key] for r in rs) if rs else None
    return {'questions':len(rows),'answerable':len(answerable),'runtime_errors':sum(r['status']!='ok' for r in rows),
       'answer_accuracy':mean(answerable,'answer_correct'),'evidence_aligned_accuracy':mean(answerable,'evidence_aligned_fact'),
       'recall_at_k':mean(answerable,'recall_at_k'),'mrr':mean(answerable,'reciprocal_rank'),'action_accuracy':mean(rows,'action_correct'),
       'unknown_refusal':mean(unknown,'action_correct'),'route_accuracy':mean(routes,'action_correct'),
       'generation_p50_seconds_first_compute':percentile([r['generation_seconds'] for r in rows if r['model']],.5),
       'generation_p95_seconds_first_compute':percentile([r['generation_seconds'] for r in rows if r['model']],.95),
       'retrieval_component_compute_p50_estimate':percentile([r['retrieval_trace'].get('component_compute_seconds_estimate',0) for r in rows if r['model']],.5),
       'peak_cuda_gib_reader':max((r.get('peak_cuda_gib',0) for r in rows),default=0),'prediction_cache_hits':sum(r.get('prediction_cache_hit',False) for r in rows),
       'failures':dict(Counter(r['failure_category'] for r in rows)),'scope':'Small/custom same-product regression and paraphrase comparison, not new independent industry benchmark. Page annotations not full entailment.'}


def freeze():
    if (ROOT/'comparison-freeze.json').exists():raise FileExistsError('Do not overwrite B freeze')
    choices=[]
    for mode in MODES:
        value=json.loads((REPORTS/f'b-dev-{mode}-v1/summary.json').read_text('utf-8'));choices.append(value)
    # Predeclared tie-break: evidence, answer accuracy, then measured compute cost.
    winner=sorted(choices,key=lambda s:(-s['evidence_aligned_accuracy'],-s['answer_accuracy'],s['retrieval_component_compute_p50_estimate']))[0]
    write_json(ROOT/'selection.json',{'selected_mode':winner['mode'],'reason':'Dev evidence-aligned accuracy, answer accuracy, then lower retrieval component compute. Test not consulted.','dev_summaries':choices})
    write_json(ROOT/'comparison-freeze.json',{'settings':SETTINGS,'questions_sha256':file_hash(ROOT/'questions.jsonl'),'selection_sha256':file_hash(ROOT/'selection.json'),
              'reader_source_sha256':file_hash(Path(__file__).parent/'stage_a.py'),'frozen_at':timestamp()})
    operation('按开发集选择B交互模式并锁定对照','优先证据对齐事实，再答案，再检索组件实测成本；六模式全部保留，测试只报告不选参。',winner['mode'],[ROOT/'selection.json',ROOT/'comparison-freeze.json'],'旧A gold与代码不改；随后完整运行各100题回归测试和受限重试。')
    print('SELECTED',winner['mode'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','freeze']);parser.add_argument('--split',choices=['dev','test'],default='dev');parser.add_argument('--mode',choices=MODES+['retry'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='freeze':freeze()
    else:run(args.split,[args.mode] if args.mode else None)

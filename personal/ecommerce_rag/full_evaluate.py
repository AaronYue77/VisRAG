"""All-corpus retrieval comparisons and separate image-reader QA evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
from full_catalog import ROOT
from full_protocol import EVAL,SETTINGS,compact
from full_retrieval import FullRetriever,MODES
from full_app import FullApplication,EvidenceReader
from full_audit import append_record
from common import WORKSPACE,read_jsonl,write_json,file_hash,operation,timestamp
from stage_a import snapshot,MODEL_REVISION,percentile
from audit import append_question

REPORTS=WORKSPACE/'reports/ecommerce_rag'
PREFIX='abo-full-v2'


def verify():
    protocol=json.loads((EVAL/'protocol.json').read_text('utf-8'))
    assert file_hash(EVAL/'questions.jsonl')==protocol['questions_sha256']
    assert file_hash(EVAL/'retrieval-questions.jsonl')==protocol['retrieval_questions_sha256']
    assert file_hash(EVAL/'documents.json')==protocol['documents_sha256']
    return protocol


def retrieve(split):
    proto=verify();retriever=FullRetriever()
    if split=='test':
        freeze=json.loads((EVAL/'selection.json').read_text('utf-8'))
        assert freeze['questions_sha256']==proto['retrieval_questions_sha256'] and freeze['settings']==SETTINGS
    questions=[q for q in read_jsonl(EVAL/'retrieval-questions.jsonl') if q['split']==split]
    for mode in MODES:
        name=f'{PREFIX}-ret-{split}-{mode}';out=REPORTS/name
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True)
        config_path=out/'config.json'
        if not config_path.exists():
            write_json(config_path,{'mode':mode,'split':split,'settings':SETTINGS,'protocol':proto,'source_sha256':snapshot(out),
             'note':'All147702 searchable, NO SKU/type/domain narrowing. Name templates not an independent human relevance benchmark.'})
        config=json.loads(config_path.read_text('utf-8'));rows=read_jsonl(out/'results.jsonl') if (out/'results.jsonl').exists() else []
        done={r['question_id'] for r in rows}
        with (out/'results.jsonl').open('a',encoding='utf-8') as f:
            for i,q in enumerate(questions):
                if q['question_id'] in done:continue
                found=retriever.search(q['question'],mode,5,q['filters'])
                items=found['items'];family_ranks=[r['rank'] for r in items if r['family_key']==q['reference_family']]
                exact_ranks=[r['rank'] for r in items if r['listing_key']==q['reference_listing_key']]
                row={**q,'reference_pages':[q['reference_listing_key']+':metadata'],'evidence_kind':'metadata_record',
                  'run_name':name,'timestamp':timestamp(),'status':'ok','config_sha256':file_hash(config_path),
                  'source_sha256':config['source_sha256'],'model':None,'model_revision':None,'generation_seconds':0.0,
                  'answer':'仅全量检索评测，未调用回答模型','citations':[],'retrieval_seconds':found['seconds'],'retrieval_trace':found['trace'],
                  'retrieved_pages':[{**{k:r[k] for k in ('listing_key','sku','domain','product_type','rank','score','family_key')},
                       'page_id':r['listing_key']+':metadata','text_source':'original_metadata_record','source_member':r['source_member'],'source_line':r['source_line']} for r in items],
                  'metrics':{'family_hit_at_1':float(bool(family_ranks) and min(family_ranks)==1),'family_hit_at_5':float(bool(family_ranks)),
                             'family_mrr_at_5':1/min(family_ranks) if family_ranks else 0.0,'exact_hit_at_5':float(bool(exact_ranks)),
                             'evaluation':'Same source-family is relevant; full/short names are source-derived templates, not natural user labels.'}}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();append_record(out,row);rows.append(row)
                if i%20==0:print(name,i+1,'/',len(questions),flush=True)
        summary={'questions':len(rows),'global_records':147702,'mode':mode,'split':split,
          **{k:statistics.mean(r['metrics'][k] for r in rows) for k in ('family_hit_at_1','family_hit_at_5','family_mrr_at_5','exact_hit_at_5')},
          'retrieval_p50_seconds':percentile([r['retrieval_seconds'] for r in rows],.5),
          'retrieval_p95_seconds':percentile([r['retrieval_seconds'] for r in rows],.95),
          'macro_type_hit_at_5':statistics.mean(statistics.mean(r['metrics']['family_hit_at_5'] for r in rows if r['product_type']==t) for t in {r['product_type'] for r in rows})}
        write_json(out/'summary.json',summary)
        (out/'运行报告.txt').write_text('全147702条商品元数据全局检索；80条/划分为抽评问题，不是只索引80条。\n'+json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
        operation('全量全局检索对照完成','查询只含源商品名完整或截短文本，无SKU/类目/站点过滤；预测后按原始关联家族和具体listing分别计分。',
           json.dumps(summary,ensure_ascii=False),[out/'逐题结果.txt',out/'运行报告.txt'],'各模式同一全量库；模板与共享家族局限保留，不把仅检索当VLM答对。')
        print(json.dumps(summary),flush=True)
    retriever.close()


def freeze():
    verify();path=EVAL/'selection.json'
    if path.exists():return
    summaries=[json.loads((REPORTS/f'{PREFIX}-ret-dev-{m}/summary.json').read_text('utf-8')) for m in MODES]
    selected=max(summaries,key=lambda s:(s['family_hit_at_5'],s['family_mrr_at_5'],-s['retrieval_p50_seconds']))
    write_json(path,{'selected_mode':selected['mode'],'settings':SETTINGS,'dev_results':summaries,
      'questions_sha256':file_hash(EVAL/'retrieval-questions.jsonl'),'reason':'DEV familyHit@5, MRR@5, lower median time; test not consulted.',
      'frozen_at':timestamp()})
    operation('冻结全量B检索选择','仅读取四组开发检索结果，不读取测试；允许简单BM25胜出，避免增加无收益组件。',
      selected['mode'],[path],'测试之后不回改模式、预算和题集。')
    print('Selected full B:',selected['mode'],flush=True)


class CachedReader(EvidenceReader):
    def predict(self,question,request,pages,image_dir):
        settings={'question':question,'request':request,'pages':[(p['page_id'],p['pdf_sha256']) for p in pages],
          'reader_source':file_hash(Path(__file__).parent/'stage_a.py'),'revision':MODEL_REVISION,'pixels':786432,'output_tokens':160}
        key=hashlib.sha256(json.dumps(settings,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        folder=ROOT/'reader-cache';folder.mkdir(exist_ok=True);p=folder/(key+'.json')
        if p.exists():return {**json.loads(p.read_text('utf-8')),'prediction_cache_hit':True,'prediction_cache_key':key,'actual_generation_wall_seconds':0.0}
        start=time.perf_counter();value=super().predict(question,request,pages,image_dir)
        value.update(prediction_cache_hit=False,prediction_cache_key=key,actual_generation_wall_seconds=time.perf_counter()-start)
        write_json(p,value);return value


def qa(only_split=None,only_stage=None):
    proto=verify();cache=CachedReader()
    for split in ('dev','test'):
        if only_split and split!=only_split:continue
        for stage in ('A','B'):
            if only_stage and stage!=only_stage:continue
            name=f'{PREFIX}-qa-{split}-{stage.lower()}-r2';out=REPORTS/name
            if (out/'summary.json').exists():continue
            if out.exists():raise FileExistsError('Incomplete QA preserved; inspect before resuming')
            out.mkdir();app=FullApplication(stage,report_root=out);app.reader=cache;session=app.session()
            config={'protocol':proto,'stage':stage,'mode':app.mode,'source_sha256':snapshot(out),
               'note':'Explicit-SKU fields use exact identity + within-product page BM25 for both stages. Reader cache reuses identical evidence only; not a second independent prediction or global retrieval gain.'}
            write_json(out/'config.json',config);rows=[]
            with (out/'results.jsonl').open('w',encoding='utf-8') as f:
                for q in read_jsonl(EVAL/'questions.jsonl'):
                    if q['split']!=split:continue
                    response=session.respond(q['question'],q['request'])
                    prediction=read_jsonl(session.out/'results.jsonl')[-1]
                    row={**prediction,**q,'run_name':name,'run_directory':str(out.relative_to(WORKSPACE)),'source_sha256':config['source_sha256'],
                      'config_sha256':file_hash(out/'config.json'),'origin_session_record':str(session.out/'results.jsonl')}
                    correct=float(row['action']==q['expected_action'])
                    if q['expected_action']=='answer':correct*=float(compact(q['reference_answer']) in compact(row['answer']))
                    evidence=float(bool(correct) and set(q['reference_pages'])<=set(row['citations'])) if q['reference_pages'] else None
                    row['metrics']={'action_correct':float(row['action']==q['expected_action']),'answer_correct':correct,'evidence_aligned_fact':evidence,
                         'evaluation':'Source value substring/number-unit and all annotated pages; not full semantic entailment or general visual accuracy.'}
                    row['failure_category']='none' if correct and (not q['reference_pages'] or evidence) else ('runtime' if row['status']=='error' else 'reading_or_protocol')
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();append_question(out,row);rows.append(row)
                    if len(rows)>=3 and all(r['status']=='error' for r in rows[-3:]) and len({r.get('error') for r in rows[-3:]})==1:
                        raise RuntimeError('Repeated Reader failure; stopped early and preserved partial run: '+str(row.get('error')))
                    if len(rows)%20==0:print(name,len(rows),row['answer'][:90],flush=True)
            facts=[r for r in rows if r['reference_pages']];unknown=[r for r in rows if r['field']=='unknown'];routes=[r for r in rows if r['field']=='route']
            summary={'questions':len(rows),'answerable':len(facts),'stage':stage,'split':split,
              'answer_accuracy':statistics.mean(r['metrics']['answer_correct'] for r in facts),
              'evidence_aligned_accuracy':statistics.mean(r['metrics']['evidence_aligned_fact'] for r in facts),
              'unknown_refusal':statistics.mean(r['metrics']['answer_correct'] for r in unknown),
              'boundary_accuracy':statistics.mean(r['metrics']['answer_correct'] for r in routes),
              'runtime_errors':sum(r['status']=='error' for r in rows),'generation_cache_hits':sum(r.get('prediction_cache_hit',False) for r in rows),
              'actual_new_predictions':sum(r.get('model') is not None and not r.get('prediction_cache_hit',False) for r in rows)}
            write_json(out/'summary.json',summary)
            (out/'运行报告.txt').write_text('全库身份查找后的PDF图片问答；字段题两阶段证据相同，明确复用同输入预测。\n'+json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            operation('全量A/B跨类目图片问答验收','公共商品编号/站点查全量库，生成实际PDF并选择证据页，固定Qwen读图；预测后计算源字段/引用/拒答和边界。',
              json.dumps(summary,ensure_ascii=False),[out/'逐题结果.txt',out/'运行报告.txt'],'未使用参考答案选择证据；相同预测缓存标记；不把字段查询当无编号全局召回收益。')
            print(json.dumps(summary),flush=True)
            app.close(release_model=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['retrieval','freeze','qa']);p.add_argument('--split',choices=['dev','test']);p.add_argument('--stage',choices=['A','B']);a=p.parse_args()
    if a.command=='retrieval':retrieve(a.split or 'dev')
    elif a.command=='freeze':freeze()
    else:qa(a.split,a.stage)

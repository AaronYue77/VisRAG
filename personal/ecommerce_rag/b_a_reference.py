"""Auxiliary reference: current A's topic-normalized BM25, with the same B reader."""
import json
from pathlib import Path
import re
import time
from b_protocol import ROOT,CORPUS
from b_evaluate import CachedReader,summary_for,REPORTS
from dialogue_retrieval import DialogueRetriever
from dialogue_app import literal_topic
from stage_a_index import route
from stage_a import snapshot,score,MODEL_REVISION
from common import WORKSPACE,read_jsonl,write_json,file_hash,timestamp,operation
from audit import append_question


def run():
    reader=CachedReader();retriever=DialogueRetriever(CORPUS)
    for split in ('dev','test'):
        name='b-'+split+'-a-topic-reference-v1';out=REPORTS/name;out.mkdir(exist_ok=False)
        config={'source_sha256':snapshot(out),'reference':'Existing A DialogueRetriever with public literal topic detection, fixed B Reader',
           'note':'Auxiliary baseline added after main B dev selection to avoid confusing raw-query BM25 with current A normalized-topic retrieval. Does NOT change chosen B mode or predeclared six-arm protocol. Not the full A semantic-routing dialogue benchmark.'}
        write_json(out/'config.json',config);rows=[];before=reader.calls
        with (out/'results.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for q in read_jsonl(ROOT/'questions.jsonl'):
                if q['split']!=split:continue
                decision=route(q['question'],q['request'],retriever.documents);started=time.perf_counter()
                topic=literal_topic(q['question']) or ('appearance' if re.search('扶手|照片|图片',q['question']) else 'unknown')
                pages,trace,seconds=retriever.search(q['question'],decision['skus'],topic,{},2) if decision['action']=='retrieve' else ([],{},0.0)
                row={**q,'run_name':name,'run_directory':str(out.relative_to(WORKSPACE)).replace('\\','/'),'timestamp':timestamp(),'status':'ok',
                    'source_sha256':config['source_sha256'],'config_sha256':file_hash(out/'config.json'),'routing':decision,'retrieval_trace':trace,
                    'retrieval_seconds':seconds,'generation_seconds':0.0,'model':None,'model_revision':None,'citations':[],'raw_output':'',
                    'retrieved_pages':[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source')} for p in pages]}
                if decision['action']=='retrieve':row.update(reader.predict(q,pages),model='Qwen/Qwen3.5-2B',model_revision=MODEL_REVISION)
                else:row.update(action=decision['action'],answer=decision['answer'],answer_source='rule_router')
                row['metrics']=score(row,q)
                if q['reference_pages']:row['metrics']['evidence_aligned_fact']=float(bool(row['metrics']['answer_correct']) and set(q['reference_pages'])<=set(row['citations']))
                row['failure_category']='none' if row['metrics']['answer_correct'] and (not q['reference_pages'] or row['metrics']['evidence_aligned_fact']) else 'reference_failure'
                row['total_seconds']=time.perf_counter()-started;rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();append_question(out,row)
        summary=summary_for(rows);summary.update(mode='a_topic_reference',split=split,actual_new_reader_calls=reader.calls-before)
        write_json(out/'summary.json',summary)
        (out/'辅助基线说明.txt').write_text(config['note']+'\n'+json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
        operation('补充当前A主题规范化检索参考组','保留现有A DialogueRetriever，不改代码；从当前问题字面取topic，统一使用B固定Reader与像素预算。',json.dumps(summary,ensure_ascii=False),[out/'逐题结果.txt',out/'辅助基线说明.txt'],'这是后补参考分析，不拿它回改六组主实验参数或默认选择；避免把规范化query变化误归于向量算法。')
        print(name,summary,flush=True)

if __name__=='__main__':run()

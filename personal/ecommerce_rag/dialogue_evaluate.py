"""Run PREDECLARED scenarios, keep all turns, evaluate only after prediction."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from dialogue_app import DialogueApplication,SETTINGS
from dialogue_protocol import ROOT
from common import WORKSPACE,file_hash,write_json,operation


def run(split,name,ids=None,protocol_root=ROOT):
    output=WORKSPACE/'reports/ecommerce_rag'/name
    if output.exists():raise FileExistsError('Use a new evaluation name')
    output.mkdir(parents=True)
    protocol_root=Path(protocol_root)
    protocol=json.loads((protocol_root/'protocol.json').read_text('utf-8'))
    scenario_file=protocol_root/(split+'.json')
    if file_hash(scenario_file)!=protocol['sha256'][split]:raise ValueError('Predeclared scenarios changed')
    if split=='acceptance':
        freeze=protocol_root/'acceptance-freeze.json'
        if not freeze.exists():raise ValueError('Freeze code and settings before acceptance')
        frozen=json.loads(freeze.read_text('utf-8'))
        for path,digest in frozen['source_sha256'].items():
            if file_hash(Path(path))!=digest:raise ValueError('Code differs from acceptance freeze')
        for name,digest in frozen['corpus_sha256'].items():
            if file_hash(WORKSPACE/'data/ecommerce_rag/abo-dialogue-v2'/name)!=digest:raise ValueError('Corpus differs from acceptance freeze')
        if ids:raise ValueError('Acceptance must include complete predeclared suite')
    cases=json.loads(scenario_file.read_text('utf-8'))
    if ids:cases=[c for c in cases if c['id'] in ids.split(',')]
    write_json(output/'evaluation-config.json',{'split':split,'scenarios_sha256':file_hash(scenario_file),'settings':SETTINGS,
                'note':'Conversation evaluation is separate from original 76 static questions. Not a claim of generalization to unseen products.'})
    app=DialogueApplication()
    rows=[]
    for case in cases:
        session=app.session()
        for i,turn in enumerate(case['turns'],1):
            result=session.respond(turn['question'])
            raw_rows=[json.loads(line) for line in (session.output/'results.jsonl').open(encoding='utf-8') if line.strip()]
            actual=raw_rows[-1]
            expected=turn['expected']
            checks={'action':result['action'] in expected['actions'],
                    'context':expected['skus'] is None or result['current_skus']==expected['skus'],
                    'expected_content':all(v.lower() in result['answer'].lower() for v in expected['contains']),
                    'pending':expected['pending'] is None or bool(session.state['pending'])==expected['pending'],
                    'citation_scope':set(result['citations'])<=set(p['page_id'] for p in result['retrieved_pages']),
                    'answer_has_evidence':result['action']!='answer' or bool(actual.get('claims') and result['citations']),
                    'no_false_999_fact':not (result['action']=='answer' and any('999' in c['quote'] for c in actual.get('claims',[])))}
            if 'active_topic' in expected:checks['active_topic']=session.state['active_topic']==expected['active_topic']
            if 'needs_contains' in expected:checks['needs_slots']=all(v in session.state['needs'][k] for k,v in expected['needs_contains'].items())
            if 'conflict' in expected:checks['source_conflict']=bool(actual.get('source_conflicts'))==expected['conflict']
            if 'catalog_count' in expected:checks['catalogue_count']=len(result['catalog_items'])==expected['catalog_count']
            if result['action']=='catalog':
                source_items=[{'sku':d['sku'],'title':d['title']} for d in sorted(app.documents,key=lambda d:d['sku'])]
                checks['catalogue_metadata_truth']=result['catalog_items']==source_items and all(x['sku']+' | '+x['title'] in result['answer'] for x in source_items) and actual['answer_source']=='current_catalog_metadata'
                checks['catalogue_no_fake_pages']=not result['citations'] and not result['retrieved_pages']
            if 'lane' in expected:checks['conversation_lane']=actual['conversation_lane']==expected['lane']
            if actual['conversation_lane']=='general':checks['general_without_fake_evidence']=not actual['citations'] and not actual['retrieved_pages']
            row={'case':case['id'],'turn':i,'question':turn['question'],'expected':expected,'actual':result,'checks':checks,
                 'model_calls':actual['model_calls'],'context_after':actual['context_after'],'claims':actual.get('claims',[]),'source_conflicts':actual.get('source_conflicts',[])}
            rows.append(row)
            with (output/'evaluation.jsonl').open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(case['id'],i,result['action'],'PASS' if all(checks.values()) else 'FAIL',f"{result['total_seconds']:.2f}s",flush=True)
        session.close('evaluation_end')
    summary={'split':split,'sessions':len(cases),'turns':len(rows),'all_checks_passed':sum(all(r['checks'].values()) for r in rows),
       'by_check':{k:{'passed':sum(r['checks'].get(k,False) for r in rows if k in r['checks']),'applicable':sum(k in r['checks'] for r in rows)} for k in sorted({k for r in rows for k in r['checks']})},
       'total_model_calls':sum(c.get('model_executed',True) for r in rows for c in r['model_calls']),'model_load_count':app.model.load_count,
       'turn_latency_p50':statistics.median(r['actual']['total_seconds'] for r in rows),
       'turn_latency_p95':sorted(r['actual']['total_seconds'] for r in rows)[min(len(rows)-1,int(.95*len(rows)))],
       'peak_cuda_gib':max((c.get('peak_cuda_gib',0) for r in rows for c in r['model_calls']),default=0),
       'failures':[{k:r[k] for k in ('case','turn','question','expected','actual','checks')} for r in rows if not all(r['checks'].values())],
       'scope':'Predeclared new dialogue scenarios on existing 24 products. Automated checks are NOT full semantic/hallucination evaluation; manual quote/relevance review required.'}
    write_json(output/'summary.json',summary)
    lines=['A阶段独立对话评测：'+split,'方法：预测只接收问题和本会话上下文；预测之后才读取本轮预期评分；原静态76题未改、未混算。',json.dumps(summary,ensure_ascii=False,indent=2)]
    for r in rows:lines += ['='*65,f"{r['case']} 第{r['turn']}轮",'问题：'+r['question'],'预期：'+json.dumps(r['expected'],ensure_ascii=False),'实际：'+r['actual']['answer'],'检查：'+json.dumps(r['checks'],ensure_ascii=False),'本会话逐题TXT：'+r['actual']['record_txt']]
    (output/'对话评测结果.txt').write_text('\n'.join(lines)+'\n','utf-8')
    if split=='acceptance':
        from dialogue_verify import extra_checks
        extra_checks(app,output)
    operation('执行独立对话'+split+'评测','按预先固定的完整会话执行真实本地模型，分别检查动作、上下文、内容关键字和引用范围；保留所有失败。',
       f"{len(rows)}轮，{summary['all_checks_passed']}轮通过自动检查；模型加载{app.model.load_count}次，仍需人工核验语义与证据。",[output/'对话评测结果.txt',output/'evaluation.jsonl'],summary['scope'])
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','acceptance'],required=True)
    parser.add_argument('--run-name',required=True);parser.add_argument('--ids');parser.add_argument('--protocol-root',default=str(ROOT))
    args=parser.parse_args();print(json.dumps(run(args.split,args.run_name,args.ids,args.protocol_root),ensure_ascii=False,indent=2))

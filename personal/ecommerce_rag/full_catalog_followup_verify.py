"""Real HTTP reproduction of the user's whole catalog/search/overview conversation."""
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from full_app import FullApplication,handler_for
from common import WORKSPACE,read_jsonl,write_json,operation


def run():
    out=WORKSPACE/'reports/ecommerce_rag'/('catalog-followup-'+time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir();checks=[];exchanges=[]
    app=FullApplication('B',report_root=out)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    sid=None
    def ask(q,action,extra=None,fresh=False):
        nonlocal sid
        body={'question':q}
        if sid and not fresh:body['session_id']=sid
        req=Request('http://127.0.0.1:'+str(server.server_port)+'/ask',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=180) as r:v=json.load(r)
        if not fresh:sid=v['session_id']
        raw=read_jsonl(Path(v['record_txt']).with_name('results.jsonl'))[-1]
        ok=v['status']=='ok' and v['action']==action
        if extra:ok&=extra(v,raw)
        checks.append({'question':q,'expected':action,'passed':bool(ok),'action':v['action'],'answer':v['answer'],'citations':v['citations'],
            'answer_source':raw.get('answer_source'),'routing':raw['routing'],'trace':raw['retrieval_trace'],'record_txt':v['record_txt']})
        exchanges.append({'request':body,'response':v});write_json(out/'checks.json',checks);write_json(out/'http-exchanges.json',exchanges)
        print(q,action,v['action'],bool(ok),v['answer'][:110].replace('\n',' / '),flush=True)
        return v,raw
    try:
        ask('hello！！','chat')
        ask('你包含哪些商品？','catalog',lambda v,r:'147,702' in v['answer'] and 'SHOES' in v['answer'] and r['answer_source']=='verified_full_catalog_statistics')
        ask('我想查询鞋子类的商品','search_results',lambda v,r:bool(v['candidates']) and all(p['product_type'] in {'SHOES','BOOT','SANDAL','TECHNICAL_SPORT_SHOE'} for p in v['candidates']) and r['retrieval_trace']['public_search_plan']['query']=='shoes')
        ask('B005FIG90C@amazon.co.jp','selected')
        _,first=ask('给我这个商品的相关信息','answer',lambda v,r:bool(r.get('claims')) and 'B005FIG90C' in v['answer'] and 'B005F1G90C' not in v['answer'] and bool(v['citations']))
        _,second=ask('还有呢','answer',lambda v,r:bool(r.get('claims')) and all(c['item_id'] not in {x['item_id'] for x in first['claims']} for c in r['claims']) and '写代码' not in v['answer'])
        ask('我说的还有什么相关信息','answer',lambda v,r:r['retrieval_trace']['overview_exhausted'] and '展示完' in v['answer'])
        ask('换个话题，帮我写一首关于晚风的小诗','chat',lambda v,r:not v['citations'] and not v['retrieved_pages'])
        ask('还有呢','chat',lambda v,r:not v['citations'] and not v['retrieved_pages'])
        ask('这个商品的材质是什么？','answer',lambda v,r:bool(v['citations']) and ('天然石' in v['answer'] or 'natural' in v['answer'].casefold()))
        ask('你们资料库还收录哪些类目？','catalog')
        ask('还有呢','catalog')
        ask('我想找运动鞋类的产品','search_results',lambda v,r:bool(v['candidates']) and all(p['product_type'] in {'SHOES','TECHNICAL_SPORT_SHOE'} for p in v['candidates']))
        ask('查找凉鞋','search_results',lambda v,r:bool(v['candidates']) and all(p['product_type']=='SANDAL' for p in v['candidates']))
        ask('帮我找耳机类的商品','search_results',lambda v,r:bool(v['candidates']) and all(p['product_type']=='HEADPHONES' for p in v['candidates']))
        ask('这件事让我很纠结，怎么和朋友沟通？','chat',lambda v,r:not v['citations'] and not v['retrieved_pages'])
        ask('给我这个商品的相关信息','clarify',fresh=True)
        summary={'checks':len(checks),'passed':sum(c['passed'] for c in checks),'run':str(out),
          'note':'Real HTTP/local Qwen for dialogue/router/field VLM. Catalog and overview are explicitly source-extractive, not model-generated facts. Sample dialogs are development regressions, not the old frozen test set.'}
        write_json(out/'summary.json',summary)
        (out/'完整会话验收.txt').write_text('目录、类目搜索与商品接续完整会话实测\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+'\n\n'.join(json.dumps(c,ensure_ascii=False,indent=2) for c in checks),'utf-8')
        operation('目录、类目搜索与商品接续真实HTTP完整会话验收','按用户实际问题顺序执行，再覆盖话题切换、概念对话、鞋类子类、缺商品上下文；检查真实来源字段与引用。',
          json.dumps(summary,ensure_ascii=False),[out/'完整会话验收.txt'],'目录与概览明确按源数据投影，普通/字段问题实际调用现有模型；不混写旧冻结成绩。',status='完成' if summary['checks']==summary['passed'] else '存在失败待处理')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:server.shutdown();server.server_close();thread.join(5);app.close()
    assert summary['checks']==summary['passed']
    return out


if __name__=='__main__':run()

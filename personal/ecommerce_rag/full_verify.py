"""Real HTTP and model/visual integration acceptance for the full catalog."""
import argparse
import json
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from full_app import FullApplication,handler_for
from full_cards import display_values
from full_protocol import compact
from common import WORKSPACE,write_json,operation,issue,read_jsonl


def run(model=False):
    name='abo-full-model-acceptance-v1' if model else 'abo-full-api-acceptance-v1'
    out=WORKSPACE/'reports/ecommerce_rag'/name;out.mkdir(exist_ok=False)
    app=FullApplication('B' if model else 'A',report_root=out)
    write_json(out/'scenario-before.json',{'mode':'model_and_conditional_visual' if model else 'metadata_HTTP',
       'checks':['147702 real rows and576 classes','global non-chair search','market identity','session scope','schema rejection'],
       'model_checks':['public source brand via actualVLM','outside original80 AUTO_OIL','followup evidence','visual worker GPU release/reload'] if model else [],
       'note':'Integration checks, not an independent human accuracy benchmark. Expected source fields only checked after prediction.'})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1]);checks=[];exchanges=[]
    def check(name,passed,detail):
        checks.append({'name':name,'passed':bool(passed),'detail':detail});print(name,bool(passed),str(detail)[:100],flush=True)
    def call(path,body=None):
        req=Request(base+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=240) as response:result=(response.status,json.load(response))
        except HTTPError as error:result=(error.code,json.load(error))
        exchanges.append({'path':path,'request':body,'status':result[0],'response':result[1]});return result
    try:
        status,health=call('/health');check('real HTTP full catalog health',status==200 and health['records']==147702 and health['types']==576,health)
        if not model:
            status,kinds=call('/categories');check('all source categories returned',status==200 and len(kinds)==576 and kinds['SHOES']==12965,kinds.get('SHOES'))
            status,found=call('/search',{'query':'backpack black','filters':{'product_type':'BACKPACK'},'top_k':5})
            check('global non-chair filtered search',status==200 and found['items'] and all(r['product_type']=='BACKPACK' for r in found['items']),found)
            status,oil=call('/search',{'query':'motor oil','filters':{'product_type':'AUTO_OIL'},'top_k':5})
            check('category outside evaluation20 is searchable',status==200 and oil['items'] and all(r['product_type']=='AUTO_OIL' for r in oil['items']),oil)
            status,selection=call('/ask',{'question':'选择 B01HTRD79M@primenow.amazon.co.uk'})
            check('composite identity selection',status==200 and selection['action']=='selected' and selection['current_listing_keys']==['B01HTRD79M@primenow.amazon.co.uk'],selection)
            status,guard=call('/ask',{'question':'这款的品牌是什么','session_id':selection['session_id'],'request':{'domain':'amazon.com'}})
            check('marketplace followup conflict stopped',status==200 and guard['action']=='clarify',guard)
            status,other=call('/ask',{'question':'这款的品牌是什么'})
            check('separate session has no product context',status==200 and other['action']=='clarify' and not other['current_listing_keys'],other)
            for question,action in [('查询今天库存','requires_tool'),('当前店铺退货政策是什么','not_found')]:
                status,result=call('/ask',{'question':question});check('boundary '+question,status==200 and result['action']==action,result)
            for body in [{'question':['bad']},{'question':'商品列表','reference_answer':'injected'}]:
                status,result=call('/ask',body);check('invalid schema rejected',status==400,result)
            status,result=call('/search',{'query':'coffee','filters':{'domain':"amazon.com' OR1=1 --"}})
            check('filter injection not executed',status==200 and result['items']==[],result)
            check('metadata checks do not load GPU reader',app.reader.reader is None,app.reader.load_count)
        else:
            # At least one product not used in the80-source evaluation or old24.
            row=dict(app.retriever.db.execute("SELECT * FROM products WHERE product_type='AUTO_OIL' AND main_image_id IS NOT NULL ORDER BY id LIMIT1".replace('LIMIT1','LIMIT 1')).fetchone())
            old=[r['listing_key'] for r in json.loads((WORKSPACE/'data/ecommerce_rag/abo-full-evaluation-v2/selected.json').read_text('utf-8'))]
            check('new category/product outside selected80',row['listing_key'] not in old,row['listing_key'])
            status,first=call('/ask',{'question':row['sku']+' 的品牌 brand 字段原文是什么？','request':{'skus':[row['sku']],'domain':row['domain']}})
            expected=display_values(json.loads(row['raw_json']),'brand')[0]
            check('actual VLM reads new-category source',status==200 and first['action']=='answer' and compact(expected) in compact(first['answer']),first)
            status,next_turn=call('/ask',{'question':'它的品牌原文再说一次','session_id':first['session_id']})
            check('same-session followup retrieves evidence again',status==200 and next_turn['action']=='answer' and compact(expected) in compact(next_turn['answer']) and next_turn['citations'],next_turn)
            app.visual=True
            status,search=call('/ask',{'question':'搜索 black backpack','session_id':first['session_id']})
            check('conditional visual search returns candidates',status==200 and search['action']=='search_results' and search['candidates'],search)
            session=app.sessions[first['session_id']];raw=read_jsonl(session.out/'results.jsonl')[-1]
            visual=raw['retrieval_trace'].get('conditional_visual',{})
            check('actual separate VisRAG candidate runtime',visual.get('visual_candidates',0)>0 and 'candidate' in visual.get('scope',''),visual)
            check('Qwen released before VisRAG',app.reader.reader is None,app.reader.load_count)
            status,choice=call('/ask',{'question':'第一个','session_id':first['session_id']})
            check('choose visual search candidate',status==200 and choice['action']=='selected',choice)
            status,result=call('/ask',{'question':'品牌 brand 字段原文是什么？','session_id':first['session_id']})
            chosen=dict(app.retriever.db.execute('SELECT * FROM products WHERE listing_key=?',(choice['current_listing_keys'][0],)).fetchone())
            expected=display_values(json.loads(chosen['raw_json']),'brand')[0]
            check('Qwen reloaded and answers selected visual candidate',status==200 and result['action']=='answer' and compact(expected) in compact(result['answer']) and app.reader.load_count>=2,result)
    finally:server.shutdown();server.server_close();thread.join(timeout=5);app.close()
    write_json(out/'checks.json',checks);write_json(out/'http-exchanges.json',exchanges)
    lines=[name+f'：{sum(r["passed"] for r in checks)}/{len(checks)}',
       '真实HTTP socket；模型测试用本地Qwen，视觉为独立VisRAG进程。不是生产线上压测或独立总体准确率。']
    lines += [r['name']+'：'+('通过' if r['passed'] else '未通过')+'；'+json.dumps(r['detail'],ensure_ascii=False) for r in checks]
    (out/'应用验收记录.txt').write_text('\n'.join(lines)+'\n','utf-8')
    related=issue('全量应用集成存在未通过检查','见实际检查原文。','保留输出，区分工程错误和模型质量，不据测试调整检索默认。',str(out/'应用验收记录.txt'),status='待排查') if not all(r['passed'] for r in checks) else '无'
    operation('全量模型/条件视觉集成验收' if model else '全量真实HTTP元数据接口验收',
       '启动实际本机HTTP服务，调用健康/类目/检索/会话接口，按预列场景检查返回；结束后关闭socket与数据库。',
       f'{sum(r["passed"] for r in checks)}/{len(checks)}项通过。',[out/'应用验收记录.txt'],'全元数据可检索不等于全原图已下载；条件视觉只重排候选。',related_issues=related)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',action='store_true');run(parser.parse_args().model)

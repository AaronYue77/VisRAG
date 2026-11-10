"""Real local-model B conversation/API checks; expectations never enter predictions."""
import argparse
import json
import re
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from b_app import BApplication,b_handler_for
from common import WORKSPACE,read_jsonl,write_json,operation,issue


def main(visual=False):
    out=WORKSPACE/'reports/ecommerce_rag'/('b-visual-live-acceptance-v1' if visual else 'b-app-acceptance-v1')
    out.mkdir(exist_ok=False)
    checks=[]
    def check(name,ok,detail):
        checks.append({'check':name,'passed':bool(ok),'detail':detail})
        print(name,bool(ok),str(detail)[:180],flush=True)
    if visual:
        question='请查看资料，B07124WN69 的宽度到底是多少？请保留规格表里的原单位。'
        write_json(out/'scenario-before.json',{'question':question,'request':{'skus':['B07124WN69']},
            'checks':['uncached visual worker','separate GPU models','candidate SKU isolation','quotes bound to candidates'],
            'note':'Engineering smoke check, not an independent accuracy benchmark. No parameters are selected from it.'})
        app=BApplication('visual',report_root=out)
        result=app.respond(question,{'skus':['B07124WN69']})
        session=app.sessions[result['session_id']];rows=read_jsonl(session.output/'results.jsonl');row=rows[-1]
        trace=row['retrieval_trace']
        check('new visual query used real child worker',trace.get('visual_query_cache_hit') is False and trace.get('visual_process_seconds',0)>0,trace)
        check('reader reloaded after visual worker',row['model_load_count']>=2,row['model_load_count'])
        check('no runtime error',row['status']!='error',row['status'])
        check('all candidates match requested product',bool(row['retrieved_pages']) and all(p['sku']=='B07124WN69' for p in row['retrieved_pages']),row['retrieved_pages'])
        check('citations bounded',set(row['citations'])<=set(p['page_id'] for p in row['retrieved_pages']),row['citations'])
        check('width quality check',row['action']=='answer' and bool(re.search(r'25\s*(?:inches|inch|英寸)',row['answer'],re.I)),row['answer'])
        write_json(out/'actual.json',result)
    else:
        scenarios=[
          ('商品列表','catalog',None),
          ('选择 B07124WN69','selected',None),
          ('它的宽度是多少？请保留原单位。','answer','25 inches'),
          ('它的重量是多少？请保留原单位。','answer','33 pounds'),
          ('这把椅子的电池容量是多少毫安？','not_found',None),
          ('帮我查询今天这把椅子的库存','requires_tool',None),
          ('重置','reset',None),
          ('这把椅子宽度是多少？','clarify',None),
          ('取消刚才的问题，列出商品列表','catalog',None),
          ('退出','exit',None)]
        write_json(out/'scenario-before.json',{'turns':scenarios,
           'note':'Known development-product integration regression; separate from the frozen static reader matrix. Gold checks are applied after responses only.',
           'http':['B health','new session with SKU','same-session follow-up','different-session isolation','invalid schema 400']})
        app=BApplication(report_root=out);session=app.session();results=[]
        for question,expected,fact in scenarios:
            result=session.respond(question);results.append(result)
            check(question+' action',result['action']==expected,{'expected':expected,'actual':result['action'],'answer':result['answer']})
            if fact:check(question+' fact',fact in result['answer'],result['answer'])
            check(question+' candidate/citation scope',set(result['citations'])<=set(p['page_id'] for p in result['retrieved_pages']),result['citations'])
        session.close('B_integration_regression_finished')
        server=ThreadingHTTPServer(('127.0.0.1',0),b_handler_for(app))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_address[1]);http=[]
        def request(path,body=None):
            req=Request(base+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            try:
                with urlopen(req,timeout=180) as response:value=(response.status,json.load(response))
            except HTTPError as error:value=(error.code,json.load(error))
            http.append({'path':path,'request':body,'status':value[0],'response':value[1]});return value
        try:
            status,value=request('/health');check('HTTP B health',status==200 and value['stage']=='B' and value['mode']=='bge',value)
            status,first=request('/ask',{'question':'规格表的宽度是多少？','request':{'skus':['B07124WN69']}})
            check('HTTP product query',status==200 and first['action']=='answer' and '25 inches' in first['answer'],first)
            status,next_turn=request('/ask',{'question':'重量呢？','session_id':first['session_id']})
            check('HTTP session follow-up',status==200 and next_turn['session_id']==first['session_id'] and '33 pounds' in next_turn['answer'],next_turn)
            status,other=request('/ask',{'question':'这款的宽度是多少？'})
            check('HTTP session isolation',status==200 and other['action']=='clarify' and not other['current_skus'] and other['session_id']!=first['session_id'],other)
            status,invalid=request('/ask',{'question':['bad'],'request':{}});check('HTTP invalid question rejected',status==400,invalid)
            status,invalid=request('/ask',{'question':'查询商品','untrusted_gold':'25 inches'});check('HTTP extra fields rejected',status==400,invalid)
        finally:server.shutdown();server.server_close();thread.join(timeout=5)
        write_json(out/'http-exchanges.json',http);write_json(out/'actual.json',results)
    write_json(out/'checks.json',checks)
    lines=['B真实应用集成验收：'+str(sum(v['passed'] for v in checks))+'/'+str(len(checks)),
           '聊天和API使用A原文摘录/校验Reader，静态六组采用另一个固定JSON Reader；两者成绩不能混用。',
           '检查前的场景保存在scenario-before.json，输出全部保留，未据此更改冻结数据或主对照参数。']
    lines.extend(f"{v['check']}：{'通过' if v['passed'] else '未通过'}；{json.dumps(v['detail'],ensure_ascii=False)}" for v in checks)
    lines.append('每个会话的对话记录.txt、逐题结果.txt和B阶段检索与重试轨迹.txt位于本目录的abo-dialogue-*子目录。')
    (out/'应用验收记录.txt').write_text('\n'.join(lines)+'\n','utf-8')
    failures=[v for v in checks if not v['passed']]
    related=issue('B应用集成检查存在未通过项',json.dumps(failures,ensure_ascii=False),
        '保留实际输出，区分工程错误与模型质量；不使用测试集修改主参数。',str(out/'应用验收记录.txt'),status='未解决的集成或质量短板') if failures else '无'
    operation('B真实视觉启动验收' if visual else 'B真实连续聊天与HTTP验收',
        '使用本地Qwen与实际检索，预测后检查动作、字段/单位、会话隔离、引用边界；HTTP通过真实本机socket调用。',
        f'{len(checks)-len(failures)}/{len(checks)}项通过。',[out/'应用验收记录.txt'],
        '未通过项原文保留；不以端点能访问等同生产部署。',related_issues=related)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--visual',action='store_true');main(parser.parse_args().visual)

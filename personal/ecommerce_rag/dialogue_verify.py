"""Local in-process HTTP and explicitly marked fault-recovery verification."""
import json
from http.server import ThreadingHTTPServer
import threading
import urllib.request
import urllib.error
from dialogue_app import handler_for
from common import write_json


def extra_checks(app,output):
    checks=[]
    def expect(name,value):
        checks.append({'check':name,'passed':bool(value)})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    def post(payload):
        request=urllib.request.Request(base+'/ask',data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=120) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:return error.code,json.loads(error.read())
    responses=[]
    try:
        status,r=post({'question':'你好，请问你能帮助我做什么？'});responses.append(r)
        expect('real_HTTP_identity',status==200 and r['action']=='chat')
        key=r['session_id']
        status,r=post({'question':'请解释递归，用一个小例子','session_id':key});responses.append(r)
        expect('real_HTTP_open_question',status==200 and r['action']=='chat' and r['conversation_lane']=='general' and not r['citations'])
        status,r=post({'question':'我没听懂，换成更简单的例子','session_id':key});responses.append(r)
        expect('real_HTTP_general_followup',status==200 and r['action']=='chat' and r['session_id']==key and not r['retrieved_pages'])
        status,r=post({'question':'把刚才的解释再说一次'});responses.append(r)
        expect('HTTP_new_session_has_no_general_history',status==200 and r['action']=='clarify' and '没有可引用' in r['answer'])
        status,r=post({'question':'选择 B07124WN69','session_id':key});responses.append(r)
        expect('HTTP_product_selection',status==200 and r['current_skus']==['B07124WN69'])
        status,r=post({'question':'它的重量呢？','session_id':key});responses.append(r)
        expect('real_HTTP_continuation',status==200 and r['action']=='answer' and '33' in r['answer'] and bool(r['citations']))
        status,r=post({'question':'它的重量呢？'});responses.append(r)
        expect('real_HTTP_session_isolation',status==200 and r['action']=='clarify' and r['current_skus']==[])
        status,r=post({'question':'问题','reference_answer':'not public'});expect('HTTP_reject_gold_keys',status==400)
        status,r=post({'question':None});expect('HTTP_invalid_input',status==400)
        status,r=post({'question':'重置','session_id':key});responses.append(r)
        expect('HTTP_reset',status==200 and r['current_skus']==[])
        status,r=post({'question':'退出','session_id':key});responses.append(r)
        expect('HTTP_exit',status==200 and r['action']=='exit')
    finally:
        server.shutdown();server.server_close();worker.join(3)
    expect('HTTP_listener_stopped',not worker.is_alive())
    # The fault is controlled and NOT passed off as a real-model output.
    session=app.session(initial_skus=['B07124WN69'])
    original_route=app.model.route
    def malformed(question,context,trace):
        trace.append({'stage':'semantic_route','status':'controlled_fault','raw_output':'{broken JSON',
                      'controlled_fault':True,'model_executed':False,'generation_seconds':0})
        raise ValueError('controlled malformed semantic output')
    try:
        app.model.route=malformed
        failed=session.respond('查询重量')
    finally:app.model.route=original_route
    recovered=session.respond('它的重量是多少？')
    expect('malformed_route_is_conservative',failed['action']=='clarify' and not failed['citations'] and failed['model_call_count']<=1)
    expect('real_model_recovers_after_route_fault',recovered['action']=='answer' and '33' in recovered['answer'])
    original_read=app.model.read
    def unavailable(*args,**kwargs):
        trace=args[-1]
        trace.append({'stage':'evidence_read','status':'controlled_fault','raw_output':'','controlled_fault':True,'model_executed':False,'generation_seconds':0})
        raise RuntimeError('controlled reader unavailable')
    try:
        app.model.read=unavailable
        failed_read=session.respond('这把有多重？')
    finally:app.model.read=original_read
    recovered_read=session.respond('再按资料给出重量')
    expect('reader_error_preserves_session',failed_read['status']=='error' and failed_read['current_skus']==['B07124WN69'])
    expect('real_model_recovers_after_reader_fault',recovered_read['action']=='answer' and '33' in recovered_read['answer'])
    original_scope=app.model.classify_scope
    def broken_scope(question,context,trace):
        trace.append({'stage':'scope_route','status':'controlled_fault','raw_output':'unknown-scope',
                      'controlled_fault':True,'model_executed':False,'generation_seconds':0})
        raise ValueError('controlled malformed scope output')
    try:
        app.model.classify_scope=broken_scope
        failed_scope=session.respond('解释一下太阳为什么发光')
    finally:app.model.classify_scope=original_scope
    recovered_scope=session.respond('为什么树叶通常是绿色的？')
    expect('open_scope_fault_preserves_raw_and_is_conservative',failed_scope['action']=='clarify' and not failed_scope['citations'] and failed_scope['model_call_count']==0)
    expect('real_open_reply_recovers_after_scope_fault',recovered_scope['action']=='chat' and recovered_scope['conversation_lane']=='general')
    session.close('controlled_fault_verification_complete')
    payload={'checks':checks,'HTTP_responses':responses,'fault_recovery_records':str(session.output/'逐题结果.txt'),
             'all_passed':all(c['passed'] for c in checks),'model_load_count':app.model.load_count,
             'scope':'Temporary loopback listener closed after verification. Faults explicitly injected; recovery turns use the actual already-loaded local model.'}
    write_json(output/'extra-verification.json',payload)
    (output/'接口与异常恢复验证.txt').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n','utf-8')
    return payload

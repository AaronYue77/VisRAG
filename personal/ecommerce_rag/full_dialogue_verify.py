"""Real HTTP, shared local Qwen and cross-topic full-catalog conversation checks."""
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from full_app import FullApplication,handler_for
from common import WORKSPACE,write_json,read_jsonl,operation,timestamp


def run():
    root=WORKSPACE/'reports/ecommerce_rag'/('full-conversation-verify-'+time.strftime('%Y%m%d-%H%M%S'))
    root.mkdir();all_checks=[];exchanges=[]
    for stage in ['B','A']:
        app=FullApplication(stage,report_root=root)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        sid=None
        def ask(question,expected,extra=None,new_session=False):
            nonlocal sid
            body={'question':question}
            if sid and not new_session:body['session_id']=sid
            req=Request('http://127.0.0.1:'+str(server.server_port)+'/ask',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=180) as response:value=json.load(response)
            if not new_session:sid=value['session_id']
            record=read_jsonl(Path(value['record_txt']).with_name('results.jsonl'))[-1]
            passed=value['status']=='ok' and value['action']==expected
            if expected=='chat':passed&=not value['citations'] and not value['retrieved_pages'] and not record.get('catalog_items') and record['retrieval_seconds']==0
            if extra:passed&=extra(value,record)
            check={'stage':stage,'question':question,'expected_action':expected,'passed':bool(passed),'actual_action':value['action'],'answer':value['answer'],
                'record_txt':value['record_txt'],'model_load_count':record['model_load_count'],'routing':record['routing']}
            all_checks.append(check);exchanges.append({'request':body,'response':value})
            write_json(root/'checks.json',all_checks);write_json(root/'http-exchanges.json',exchanges)
            print(stage,question,expected,value['action'],bool(passed),value['answer'][:130].replace('\n',' / '),flush=True)
            return value,record
        try:
            ask('你好','chat')
            ask('你是谁','chat',lambda v,r:'助手' in v['answer'] and '下列候选' not in v['answer'])
            ask('你会什么','chat',lambda v,r:'24' not in v['answer'] and '椅子商品' not in v['answer'])
            ask('什么是RAG？用两句话解释','chat')
            ask('没听懂，换个生活中的例子','chat')
            ask('搜索 headphones','search_results')
            ask('第一个','selected')
            selected=list(app.session(sid).selected)
            ask('品牌是什么？','answer',lambda v,r:bool(v['citations']) and v['current_listing_keys']==selected)
            ask('现在换个话题，帮我写一个Python函数计算列表平均值','chat',lambda v,r:'def ' in v['answer'] and v['current_listing_keys']==selected)
            ask('这个函数遇到空列表怎么办？','chat')
            ask('这款的品牌呢？','answer',lambda v,r:bool(v['citations']) and v['current_listing_keys']==selected)
            ask('什么是库存？','chat')
            ask('帮我查一下这款今天的库存','requires_tool')
            ask('谢谢，刚才你是谁那个问题，再介绍一下你','chat')
            ask('这款颜色是什么？','clarify',new_session=True)
            ask('重置','reset',lambda v,r:not v['current_listing_keys'])
            ask('你好，可以陪我随便聊聊吗？','chat')
            ask('我有点紧张，明天第一次面试，你有什么准备建议？','chat')
            ask('你会什么','chat')
            assert app.reader.load_count==1,'Qwen loaded twice instead of shared'
            all_checks.append({'stage':stage,'question':'shared_model_and_history','passed':True,'model_load_count':app.reader.load_count,
                  'general_history_turns':len(app.session(sid).general_history)})
        finally:
            server.shutdown();server.server_close();thread.join(5);app.close()
    write_json(root/'checks.json',all_checks)
    summary={'checks':len(all_checks),'passed':sum(c['passed'] for c in all_checks),'run':str(root),'timestamp':timestamp(),
        'note':'Real local models and HTTP, not mock answers. Content reviewed separately; routing success is not general-knowledge truth guarantee.'}
    write_json(root/'summary.json',summary)
    report='全量版自由对话真实模型验收\n方法：A/B共用现有本地Qwen执行普通聊天、解释与代码追问；仅商品事实进入资料检索，并检查切换话题后可回来。真实HTTP /ask保留session_id。\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'
    report+='\n\n'.join(json.dumps(c,ensure_ascii=False,indent=2) for c in all_checks)
    (root/'自由对话验收记录.txt').write_text(report,'utf-8')
    operation('全量自由对话真实HTTP模型验收','A/B真实本地Qwen普通聊天/跨题追问/商品取证/切换话题/会话隔离，检查Qwen仅加载一次。',json.dumps(summary,ensure_ascii=False),[root/'自由对话验收记录.txt'],'逐条真实输出与失败均保存；路由通过不代表模型知识全正确。',status='完成' if summary['checks']==summary['passed'] else '存在失败，待处理')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return root


if __name__=='__main__':run()

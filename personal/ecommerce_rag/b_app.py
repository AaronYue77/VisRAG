"""B-stage retrieval plugged into the existing continuous e-commerce application."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from b_protocol import ROOT,MODES,rewrite,query_key,SETTINGS as B_SETTINGS
from b_retrieval import BRetriever
from dialogue_app import DialogueApplication,DialogueSession,SETTINGS as A_SETTINGS,chat,handler_for
from dialogue_retrieval import KEYWORDS
from common import WORKSPACE,write_json,operation,file_hash


def selected_mode():
    return json.loads((ROOT/'selection.json').read_text('utf-8'))['selected_mode']


class BDialogueRetriever:
    def __init__(self,app,mode):
        self.app=app;self.mode=mode;self.engine=BRetriever(online_visual=True)
        self.base=self.engine.base;self.documents=self.engine.documents;self.forced=None

    def search(self,question,skus,topic,needs,top_k=2):
        if self.forced is not None:
            value=self.forced;self.forced=None
            return value
        # Keep the user's full question. A topic is a public routing hint, never gold.
        effective=question
        if topic=='fit':topic=needs.get('_active_topic') or ('width' if needs.get('space') else 'features')
        if self.mode in {'visual','multimodal_rrf','rerank'}:
            key=query_key(effective)
            if not (self.engine.visual_queries/(key+'.json')).exists() and self.app.model.reader is not None:
                # The 8GB GPU cannot hold VisRAG and Qwen at the same time.
                self.app.model.reader=None;gc.collect()
                import torch
                torch.cuda.empty_cache()
        result=self.engine.search(effective,{'skus':skus,'store_id':'amazon.com','as_of':'2026-11-01'},self.mode)
        trace={**result['trace'],'normalized_topic':topic,'query':effective,'keyword_query':effective,
               'stage':'B','reader_follows_original_dialogue_quotes':True}
        return result['pages'],trace,result['seconds']


class BSession(DialogueSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.config.update(stage='B',b_mode=self.app.b_mode,b_settings=B_SETTINGS,b_retry_enabled=self.app.b_retry,
                           baseline_source='A a-dialogue-v3.3 retained; B replaces retrieval only')
        write_json(self.output/'config.json',self.config);self.config_sha=file_hash(self.output/'config.json')
        with self.transcript.open('a',encoding='utf-8') as f:f.write('当前为B阶段：'+self.app.b_mode+'；使用原A会话规则，当前检索模式与重试信息见逐题轨迹。\n')

    def _save(self,row,before):
        result=super()._save(row,before)
        detail={'question_id':row['question_id'],'mode':self.app.b_mode,
                'retry':row.get('b_retry'),'retrieval_trace':row.get('retrieval_trace'),
                'retrieval_seconds':row.get('retrieval_seconds',0)}
        with (self.output/'B阶段检索与重试轨迹.txt').open('a',encoding='utf-8') as f:
            f.write(json.dumps(detail,ensure_ascii=False,indent=2)+'\n')
        return {**result,'stage':'B','retrieval_mode':self.app.b_mode,'bounded_retry':row.get('b_retry')}

    def _answer_product(self,row,effective,skus,topic):
        super()._answer_product(row,effective,skus,topic)
        first={k:deepcopy(row.get(k)) for k in ('answer','action','citations','retrieved_pages','retrieval_trace','retrieval_seconds')}
        row['b_retry']={'enabled':self.app.b_retry,'attempts':1,'retrieval_attempts':1,
                        'stop_reason':'evidence_found' if row['action']=='answer' else 'not_enabled_or_no_rewrite'}
        if not self.app.b_retry or row['action']!='not_found':return
        query=rewrite(effective)
        if query==effective:return
        pages,trace,seconds=self.app.retriever.search(query,skus,topic,{**self.state['needs'],'_active_topic':self.state.get('active_topic')})
        if [p['page_id'] for p in pages]==[p['page_id'] for p in first['retrieved_pages']]:
            row['retrieval_seconds']+=seconds
            row['b_retry'].update(retrieval_attempts=2,rewritten_query=query,stop_reason='unchanged_candidates_no_second_read',rewrite_retrieval_seconds=seconds)
            return
        self.app.retriever.forced=(pages,trace,seconds)
        super()._answer_product(row,effective,skus,topic)
        row['b_retry']={'enabled':True,'attempts':2,'retrieval_attempts':2,'first':first,'rewritten_query':query,
                        'stop_reason':'evidence_found' if row['action']=='answer' else 'retry_limit_abstain'}
        row['retrieval_seconds']+=first['retrieval_seconds']


class BApplication(DialogueApplication):
    def __init__(self,mode=None,retry=False,**kwargs):
        super().__init__(**kwargs)
        self.b_mode=mode or selected_mode();self.b_retry=retry
        self.retriever=BDialogueRetriever(self,self.b_mode)

    def session(self,initial_skus=None,session_id=None):
        import re
        if session_id is not None and (not isinstance(session_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',session_id)):raise ValueError('Invalid session_id')
        with self.lock:
            key=session_id or uuid.uuid4().hex;now=time.monotonic()
            for old,s in list(self.sessions.items()):
                if now-s.last_used>A_SETTINGS['session_idle_seconds']:s.close('idle_timeout');self.sessions.pop(old,None)
            if key in self.sessions:return self.sessions[key]
            if len(self.sessions)>=A_SETTINGS['max_sessions']:raise ValueError('Too many active sessions')
            session=BSession(self,key,initial_skus);self.sessions[key]=session
            return session


def b_handler_for(app):
    BaseHandler=handler_for(app)
    class BHandler(BaseHandler):
        def do_GET(self):
            if self.path=='/health':self.reply(200,{'status':'ready','stage':'B','mode':app.b_mode,'retry':app.b_retry,'agent':False,'model_loaded':app.model.reader is not None})
            else:super().do_GET()
    return BHandler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['chat','ask','serve','check'],default='chat',nargs='?')
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--retry',action='store_true')
    parser.add_argument('--question',default='');parser.add_argument('--sku',action='append',default=[]);parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args();app=BApplication(args.mode,args.retry)
    operation('启动B阶段应用','复用现有A连续对话/目录/路由/证据校验，替换为开发集选择的检索器；可显式指定六种模式。',
        f'{args.command} mode={app.b_mode} retry={app.b_retry}',verification='check只检查完整性不加载GPU；chat/API按需加载，视觉模式分时使用GPU。',status='启动')
    if args.command=='check':print(json.dumps({'status':'ready','stage':'B','mode':app.b_mode,'pages':len(app.retriever.base.pages),'retry':app.b_retry,'model_loaded':False}))
    elif args.command=='chat':chat(app,args.sku)
    elif args.command=='ask':print(json.dumps(app.session(initial_skus=args.sku).respond(args.question),ensure_ascii=False,indent=2))
    else:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),b_handler_for(app));print(f'B local API http://127.0.0.1:{args.port}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()

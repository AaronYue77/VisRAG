"""A/B full ABO catalog application: browse/search/select/evidence answer."""
import argparse
import gc
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
import uuid
from full_catalog import ROOT,field
from full_cards import ensure_card,ensure_photo_card,choose_pages,evidence_fields
from full_retrieval import FullRetriever,MODES,exact_skus,expand
from full_protocol import EVAL,SETTINGS,compact
from common import WORKSPACE,write_json,file_hash,timestamp,operation,issue
from audit import append_question,format_question
from stage_a import Reader,MODEL_REVISION,snapshot
from full_conversation import FullConversation,VERSION as CONVERSATION_VERSION,bound_history
from full_catalogue import catalogue_request,category_search_request,continuation_request,overview_request,catalog_page,normalize_search,overview_chunk
from full_reference import bare_category_choice,product_task,ordinal_choice
from full_evidence import VERSION as EVIDENCE_VERSION,PHOTO_REQUEST,alternate_pages,photo_pages,read_fields,requested_fields,split_photo_request
from full_policy import POLICY_FILE,policy_answer
from qwen8_runtime import Qwen8Runtime, MODEL_ID as QWEN8, REVISION as QWEN8_REVISION


def selected_mode():
    path=EVAL/'selection.json'
    return json.loads(path.read_text('utf-8'))['selected_mode'] if path.exists() else 'bm25'


def public_item(row):
    return {k:row[k] for k in ('listing_key','sku','domain','product_type','title','brand')}


def boundary(question):
    if re.search(r'订单|物流|退款|下单|付款|库存|实时|今天.*(?:价格|售价)|现在.*(?:有货|多少钱)',question,re.I):
        return {'action':'requires_tool','answer':'当前只查询ABO历史商品资料，未接入实时价格、库存、订单或交易工具。'}
    if re.search(r'(?:当前|现行|现在|店铺).*(?:退货|保修|售后)|退货政策',question):
        return {'action':'not_found','answer':'ABO历史文案不能确认当前店铺政策，当前没有可核实的现行政策来源。'}
    return None


class EvidenceReader:
    def __init__(self, model='8b', model_dir=None):
        self.reader=None;self.load_count=0
        self.backend=model;self.model_dir=model_dir
        self.model_name,self.revision=(QWEN8,QWEN8_REVISION) if model=='8b' else ('Qwen/Qwen3.5-2B',MODEL_REVISION)
    def ensure_reader(self):
        if self.reader is None:
            runtime=Qwen8Runtime(self.model_dir) if self.backend=='8b' else None
            self.reader=Reader(vlm=runtime);self.load_count+=1
        return self.reader
    def unload(self):
        self.reader=None;gc.collect()
        import torch
        torch.cuda.empty_cache()
    def predict(self,question,request,pages,image_dir,prompt_addition=""):
        for page in pages:
            if not {'page_id','doc_id','sku','pdf','page','version'}<=page.keys():raise ValueError('Incomplete page metadata for the fixed Reader')
        self.ensure_reader()
        result=self.reader.read(question,{'skus':request.get('skus',[]),'store_id':request.get('domain','all_ABO_domains')},
                                pages,image_dir,786432,prompt_addition)
        if result['action']=='answer' and (not result.get('parse_ok') or not result.get('citations')):
            result.update(action='not_found',answer='NOT_FOUND',citations=[],guard='invalid citation/output protocol')
        return result


class FullApplication:
    def __init__(self,stage='B',mode=None,report_root=None,visual=False,retry=False,audit=True,legacy_evidence=False,model='8b',model_dir=None):
        self.stage=stage;self.mode='bm25' if stage=='A' else (mode or selected_mode())
        self.visual=visual;self.retry=retry;self.legacy_evidence=legacy_evidence
        self.audit=audit
        self.retriever=FullRetriever();self.reader=EvidenceReader(model,model_dir);self.sessions={};self.lock=threading.RLock()
        self.conversation=FullConversation(self.reader)
        self.report_root=report_root or WORKSPACE/'reports/ecommerce_rag'
    def close(self,release_model=True):
        self.sessions.clear();self.retriever.close()
        if release_model and self.reader.reader is not None:self.reader.unload()
    def session(self,key=None):
        if key is not None and (not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',key)):raise ValueError('Invalid session_id')
        key=key or uuid.uuid4().hex
        with self.lock:
            now=time.monotonic()
            for old,s in list(self.sessions.items()):
                if now-s.used>1800:self.sessions.pop(old)
            if key not in self.sessions:
                if len(self.sessions)>=64:raise ValueError('Session capacity exceeded')
                self.sessions[key]=FullSession(self,key)
            return self.sessions[key]
    def respond(self,question,request=None,session_id=None):return self.session(session_id).respond(question,request)
    def search_catalog(self,query,mode=None,top_k=5,filters=None,trace=None):
        if not isinstance(query,str) or not 0<len(query.strip())<=2000:raise ValueError('Query must contain1..2000 characters')
        if not isinstance(filters or {},dict) or set(filters or {})-{'product_type','product_types','domain'}:raise ValueError('Invalid public category filters')
        if mode is not None and mode not in MODES:raise ValueError('Unknown retrieval mode')
        if not isinstance(top_k,int) or not 1<=top_k<=50:raise ValueError('Invalid top_k')
        plan=normalize_search(query,self.retriever.manifest,filters,separate_terms=not self.legacy_evidence)
        if plan['needs_semantic_translation']:
            try:
                proposal=self.conversation.plan_search(query,self.retriever.manifest['product_types'],trace if trace is not None else [])
                plan['semantic_proposal']=proposal;plan['query']=proposal['query']
                if proposal['product_types'] and not any(k in plan['filters'] for k in ('product_type','product_types')):
                    plan['filters']['product_types']=proposal['product_types']
            except (ValueError,TypeError) as error:plan['semantic_plan_fallback']=str(error)
        if not plan['query'].strip():raise ValueError('Please specify a product name or category')
        result=self.retriever.search(plan['query'],mode or self.mode,top_k,plan['filters'],expand_query=False)
        result['trace']['public_search_plan']=plan
        return result


class FullSession:
    def __init__(self,app,key):
        self.app=app;self.key=key;self.selected=[];self.candidates=[];self.turn=0;self.used=time.monotonic();self.ended=False
        self.general_history=[];self.last_lane=None;self.last_product_query=None
        self.last_task=None;self.recent_tasks=[];self.overview_shown={};self.catalog_offset=0
        self.selected_entities=[];self.candidate_entities=[]
        name='abo-full-'+time.strftime('%Y%m%d-%H%M%S')+'-'+key[:8]
        self.out=app.report_root/name;self.out.mkdir(parents=True,exist_ok=False)
        config={'version':CONVERSATION_VERSION,'stage':app.stage,'mode':app.mode,'settings':SETTINGS,
          'active_model':app.reader.model_name,'active_model_revision':app.reader.revision,
          'quantization':'nf4' if app.reader.backend=='8b' else None,
          'evidence_version':'legacy' if app.legacy_evidence else EVIDENCE_VERSION,
          'evidence_source_sha256':file_hash(Path(__file__).with_name('full_evidence.py')),
          'policy_sha256':file_hash(POLICY_FILE) if not app.legacy_evidence and app.stage=='B' and POLICY_FILE.is_file() else None,
          'conditional_visual':app.visual,'bounded_retry':app.retry,
          'catalog_sha256':app.retriever.manifest['database_sha256'],'records':147702,'source_sha256':snapshot(self.out),
          'scope':'Open local-model conversation with bounded context; only product knowledge uses full metadata and page evidence. No business tools or long-term memory.'}
        write_json(self.out/'config.json',config);self.csha=file_hash(self.out/'config.json');self.source=config['source_sha256']
        self.transcript=self.out/'对话记录.txt';self.transcript.write_text('全量ABO商品资料检索会话：历史回答不作为事实证据。\n','utf-8')
    def respond(self,question,request=None):
        if not isinstance(question,str) or not 0<len(question.strip())<=2000:raise ValueError('question must contain 1..2000 characters')
        if request is not None and (not isinstance(request,dict) or set(request)-{'skus','domain','product_type'}):raise ValueError('Only skus/domain/product_type accepted')
        request=request or {}
        if 'skus' in request and (not isinstance(request['skus'],list) or any(not isinstance(s,str) for s in request['skus']) or len(request['skus'])>2):raise ValueError('skus must be an array of at most2 strings')
        if any(k in request and not isinstance(request[k],str) for k in ('domain','product_type')):
            raise ValueError('domain and product_type must be strings')
        with self.app.lock:
            if self.ended:raise ValueError('Session has ended')
            self.used=time.monotonic();self.turn+=1;started=time.perf_counter()
            row={'question_id':self.key+f'-{self.turn:03d}','question':question,'reference_answer':'交互未标注，参考标签不进入预测','reference_pages':[],
              'request':request,'run_name':self.out.name,'run_directory':str(self.out.relative_to(WORKSPACE)),
              'timestamp':timestamp(),'config_sha256':self.csha,'source_sha256':self.source,'status':'ok',
              'model':None,'model_revision':None,'generation_seconds':0.0,'retrieval_seconds':0.0,'retrieved_pages':[],
              'citations':[],'raw_output':'','metrics':{'evaluation':'unlabelled_interaction'},'routing':{},'retrieval_trace':{},
              'stage':self.app.stage,'global_catalog_records':147702,'current_listing_keys_before':list(self.selected),
              'model_calls':[],'conversation_lane':None,'conversation_version':CONVERSATION_VERSION}
            try:self.dispatch(question.strip(),request,row)
            except Exception as error:
                row.update(status='error',action='not_found',answer='本次检索或模型执行失败，未提供事实答案。',error=type(error).__name__+': '+str(error))
                row['issue_id']=issue('全量应用请求执行失败：'+str(error),'检索、资料生成或模型阶段异常。','保留本次失败；修复后新建运行，不覆盖原结果。',str(self.out/'逐题结果.txt'))
            calls=row['model_calls']
            if calls:
                row['model']=self.app.reader.model_name;row['model_revision']=self.app.reader.revision
                row['generation_seconds']+=sum(c.get('generation_seconds',0) for c in calls)
            if row['status']=='ok':
                lane=row.get('conversation_lane')
                if lane=='general' and row.get('action')=='chat':
                    self.general_history=bound_history(self.general_history+[{'user':question,'assistant':row['answer']}])
                if lane in {'general','product','catalog'}:self.last_lane=lane
                if lane in {'general','product','catalog'}:
                    self.last_task={'lane':lane,'kind':row.get('task_kind') or ('general' if lane=='general' else row['action']),
                        'question':question[:500],'listing_keys':list(self.selected)}
                    self.recent_tasks=(self.recent_tasks+[self.last_task])[-6:]
            row.update(total_seconds=time.perf_counter()-started,current_listing_keys_after=list(self.selected),model_load_count=self.app.reader.load_count)
            with (self.out/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if self.app.audit:append_question(self.out,row)
            else:
                with (self.out/'逐题结果.txt').open('a',encoding='utf-8') as f:f.write(format_question(row))
            with self.transcript.open('a',encoding='utf-8') as f:f.write(f'\n第{self.turn}轮\n用户：{question}\n助手：{row["answer"]}\n动作：{row["action"]}\n引用：{row["citations"]}\n')
            write_json(self.out/'session.json',{'session_id':self.key,'selected':self.selected,'candidates':self.candidates,'ended':self.ended,'turns':self.turn,
                'last_lane':self.last_lane,'general_history':self.general_history,'last_product_query':self.last_product_query,
                'last_task':self.last_task,'recent_tasks':self.recent_tasks,'overview_shown':self.overview_shown,'catalog_offset':self.catalog_offset,
                'selected_entities':self.selected_entities})
            return {k:row.get(k) for k in ('question_id','action','answer','citations','retrieved_pages','status','stage','total_seconds','retrieval_seconds','generation_seconds','global_catalog_records')}|{
                'session_id':self.key,'current_listing_keys':list(self.selected),'candidates':row.get('catalog_items',[]),
                'strategy':row.get('routing',{}).get('strategy') or row.get('answer_source') or row.get('conversation_lane') or row['action'],
                'record_txt':str(self.out/'逐题结果.txt'),'conversation_txt':str(self.transcript)}
    def dispatch(self,text,request,row):
        def finish(action,answer,reason):row.update(action=action,answer=answer,routing={'reason':reason})
        if text.lower() in {'退出','exit','quit'}:self.ended=True;return finish('exit','对话已结束。','exit')
        if text.lower() in {'重置','reset','清空'}:
            self.selected=[];self.candidates=[];self.general_history=[];self.last_lane=None;self.last_product_query=None
            self.last_task=None;self.recent_tasks=[];self.overview_shown={};self.catalog_offset=0
            self.selected_entities=[];self.candidate_entities=[]
            return finish('reset','已清空本会话的聊天上下文和商品选择。','reset')
        category_search=category_search_request(text,self.app.retriever.manifest) or bare_category_choice(text,self.last_task,self.app.retriever.manifest)
        ids=exact_skus(text)
        if text in {'商品列表','类目列表','类别','类目','categories','list'} or (
                catalogue_request(text) and not category_search and (self.app.legacy_evidence or not ids)):
            return self.show_catalog(row)
        question_has_ids=bool(ids);supplied=[s.upper() for s in request.get('skus',[])]
        inline_refs=[(m[1].upper(),m[2].lower().rstrip('.')) for m in re.finditer(r'((?:B[0-9][A-Z0-9]{8}|[0-9]{9}[0-9X]))@([a-z0-9./-]+)',text,re.I)]
        inline_domains=dict(inline_refs)
        if not self.app.legacy_evidence and len(set(inline_refs))>len(inline_domains):
            return finish('clarify','同一编号出现于多个站点，请先指定本轮要核对的站点。','multiple_sites_for_one_sku')
        if request.get('domain') and any(v!=request['domain'] for v in inline_domains.values()):
            return finish('clarify','文本中的站点与请求指定站点不同，请确认。','conflicting_marketplace_context')
        if ids and supplied and set(ids)!=set(supplied):return finish('clarify','问题中的编号与请求指定编号不一致，请确认。','conflicting_product_context')
        ids=ids or supplied
        if len(ids)>2:return finish('clarify','一次最多查询两个商品，请缩小范围。','bounded_product_count')
        ordered_task=None
        if not ids:
            ordered=ordinal_choice(text,self.last_task)
            if ordered and self.candidates:
                if ordered.get('ambiguous'):return finish('clarify','这次提到了多个候选，请明确先看哪一款，或提供商品编号进行比较。','multiple_ordered_references')
                i=ordered['index']
                if not 0<=i<len(self.candidates):return finish('clarify','该序号不在上一轮候选列表中。','invalid_candidate_number')
                self.selected=[self.candidates[i]];self.selected_entities=[self.candidate_entities[i]]
                row.update(conversation_lane='product',task_kind='selected')
                if ordered['task']=='select':return finish('selected','已选择 '+self.selected[0]+'，可以继续问参数或原始描述。','candidate_selection')
                ordered_task={'intent':'fact','reason':'ordered_candidate_task','product_task':ordered['task']}
        from full_conversation import clear_route,direct_business_request,programming_task
        certain=clear_route(text,bool(self.selected or supplied),self.last_lane)
        if not self.app.legacy_evidence and self.app.stage=='B' and (not certain or certain['intent']!='general'):
            policy_started=time.perf_counter()
            policy=policy_answer(text,request.get('domain'))
            if policy:
                row.update(policy,conversation_lane='policy',routing={'strategy':'policy_snapshot','reason':'explicit_policy_scope'},
                           retrieval_seconds=time.perf_counter()-policy_started)
                return
        if not certain and not self.app.legacy_evidence and (self.selected or supplied) and requested_fields(text):
            certain={'intent':'fact','reason':'explicit_source_field_lookup'}
        if category_search and (not certain or certain['intent']!='general'):
            certain={'intent':'search','reason':'explicit_named_category_search'}
        # Apply the business boundary to resolved references as well as model routes.
        decision=boundary(text)
        if decision and direct_business_request(text) and (certain is None or certain['intent']!='general'):
            row.update(decision,routing={'reason':'business_or_current_policy_boundary'},conversation_lane='product');return
        followup=continuation_request(text)
        previous=self.last_task or {}
        resolved=ordered_task or product_task(text,self.selected_entities,previous,has_explicit_id=question_has_ids)
        if followup and previous.get('lane')=='catalog':return self.show_catalog(row,more=True)
        product_more=bool(followup and self.selected and previous.get('lane')=='product' and previous.get('kind')!='search_results')
        if product_more:
            route={'intent':'fact','reason':'continue_latest_product_task'};row['overview_more']=True
        elif resolved:
            route=resolved
            if resolved.get('product_task') in {'overview','features'}:row['product_task']=resolved['product_task']
        elif overview_request(text) and ((self.selected and previous.get('lane')=='product') or re.search(r'这款|这个商品|该商品|所选商品',text)):
            route={'intent':'fact','reason':'explicit_product_overview'}
        elif question_has_ids and not programming_task(text):
            route={'intent':'fact','reason':'explicit_product_identity'}
        else:
            route=certain or self.app.conversation.route(text,self.selected or ids,self.last_lane,self.general_history,row['model_calls'],
                                                     recent_tasks=self.recent_tasks,last_task=self.last_task,selected_entities=self.selected_entities)
        row['routing']={**route,'last_lane':self.last_lane}
        if route['intent']=='general':
            row.update(action='chat',answer=self.app.conversation.reply(text,self.general_history,row['model_calls']),
                conversation_lane='general',answer_source='local_model_general_conversation')
            row['answer_truncated']=any(c['stage']=='full_open_conversation' and c['status']=='partial' for c in row['model_calls'])
            return
        if route['intent']=='catalog':return self.show_catalog(row)
        if route['intent'] in {'business','policy'}:
            decision=boundary(text) or ({'action':'requires_tool','answer':'当前没有接入实时业务查询或交易工具，无法替你查询或执行。'} if route['intent']=='business' else
                {'action':'not_found','answer':'当前没有可核实的现行店铺政策来源。'})
            row.update(decision,conversation_lane='product');return
        if route['intent']=='clarify':return finish('clarify',route.get('clarification','我还没确定你的意思，可以补充一下想聊的内容或要查的商品吗？'),route['reason'])
        search_intent=route['intent']=='search'
        row['conversation_lane']='product'
        if ids:
            decision=boundary(text)
            if decision:row.update(decision,routing={'reason':'business_or_current_policy_boundary'});return
        if not ids and search_intent:
            if re.search(r'这款|它的|这件|这把',text) and not search_intent:return finish('clarify','请先搜索并选择商品，或提供明确编号。','missing_product_context')
            query=re.sub(r'^(?:搜索|查找|帮我找|请帮我找|找|search|find)[:：\s]*','',text,flags=re.I)
            filters={k:request[k] for k in ('domain','product_type') if request.get(k)}
            result=self.app.search_catalog(query,self.app.mode,6 if self.app.visual else 5,filters,row['model_calls'])
            if self.app.visual:
                # Candidate-conditional only; it cannot recover products missed globally.
                from full_visual import rerank
                ordering,visual_trace=rerank(query,result['items'],self.app.reader)
                result['items']=ordering[:5];result['trace']['conditional_visual']=visual_trace
                result['seconds']+=visual_trace.get('seconds',0)
            self.selected=[]
            self.selected_entities=[]
            self.last_product_query=None
            row.update(retrieval_seconds=result['seconds'],retrieval_trace=result['trace'],catalog_items=[public_item(r) for r in result['items']])
            self.candidates=[r['listing_key'] for r in result['items']]
            self.candidate_entities=[public_item(r) for r in result['items']]
            if not result['items']:return finish('not_found','没有检索到匹配资料，可以换商品名称、品牌或类目。','empty_global_search')
            lines=['从全部147702条历史商品元数据中检索到以下候选，请选择后再查具体事实：']
            lines += [f'{i}. {r["listing_key"]} | {r["product_type"]} | {r["title"]}' for i,r in enumerate(result['items'],1)]
            return finish('search_results','\n'.join(lines),'global_retrieval_select_before_fact')
        if not ids and not self.selected:
            return finish('clarify','这次需要查具体商品资料，请先告诉我商品名称/编号，或搜索后选择商品。','missing_product_context')
        products=[]
        if ids:
            for sku in ids:
                selected_domain=request.get('domain') or inline_domains.get(sku)
                rows=self.app.retriever.get(sku,selected_domain)
                if not rows:
                    self.selected=[]
                    self.selected_entities=[]
                    return finish('not_found','全量知识库也没有该编号或站点对应的资料：'+sku,'unknown_product_or_domain')
                if len(rows)>1 and not selected_domain:return finish('clarify','同一编号在多个站点有记录，请指定domain：'+', '.join(r['domain'] for r in rows),'ambiguous_marketplace')
                products.append(rows[0])
            if not self.app.legacy_evidence and request.get('product_type') and any(
                    p['product_type']!=request['product_type'] for p in products):
                return finish('clarify','指定类目与商品来源不一致，请核对商品编号或类目。','product_type_conflict')
            self.selected=[r['listing_key'] for r in products]
            self.selected_entities=[public_item(r) for r in products]
            if re.fullmatch(r'(?:选择|选中|select)?\s*'+ids[0]+r'(?:@[a-z0-9./-]+)?',text,re.I):
                row['task_kind']='selected';return finish('selected','已选择 '+self.selected[0]+'。','explicit_product_selection')
        else:
            products=[dict(self.app.retriever.db.execute('SELECT * FROM products WHERE listing_key=?',(key,)).fetchone()) for key in self.selected]
            if request.get('domain') and any(p['domain']!=request['domain'] for p in products):
                return finish('clarify','指定站点与当前商品来源不一致，请重新选择对应站点的记录。','followup_marketplace_conflict')
            if not self.app.legacy_evidence and request.get('product_type') and any(
                    p['product_type']!=request['product_type'] for p in products):
                return finish('clarify','指定类目与商品来源不一致，请核对商品编号或类目。','product_type_conflict')
        if row.get('overview_more') or row.get('product_task') in {'overview','features'} or overview_request(text):
            more=row.get('overview_more') or bool(re.search(r'还有|更多|其他|其它|继续',text))
            return self.answer_overview(row,products,more,kind=row.get('product_task','overview'))
        # Source-key availability is a public schema check, not a gold-answer lookup.
        named=re.search(r'\b(manufacturing_timestamp_utc|material|color|width|height|length|weight|brand)\b',text,re.I)
        if named and re.search(r'字段',text):
            key=named[1].lower()
            if all(key not in {k for k,v in evidence_fields(json.loads(p['raw_json']))} for p in products):
                return finish('not_found','原始商品记录没有报告该字段，不能根据常识补充。','source_field_absent')
        # A request to explain the preceding product answer retrieves it again.
        query=text
        if self.last_lane=='product' and self.last_product_query and re.search(r'再说|没听懂|解释一下|用中文|重说|刚才.*(?:回答|参数)',text) and not re.search(r'品牌|颜色|材质|尺寸|重量|多重|宽|高|长',text):
            previous=self.last_product_query
            if previous['listing_keys']==self.selected:query=previous['question']+'\n本轮追问：'+text
        self.last_product_query={'question':query,'listing_keys':list(self.selected)}
        fields=requested_fields(query) if not self.app.legacy_evidence else ()
        if fields and len(products)==1 and re.search(r'这(?:两款|两个|两件)',query):
            return finish('clarify','当前只确认了一件商品，请告诉我另一件的编号。','missing_second_product')
        if fields:
            stamp=time.perf_counter()
            row.update(read_fields(products,fields))
            row['retrieval_seconds']=time.perf_counter()-stamp
            row['routing']['strategy']='source_fields'
            row['retrieval_trace']['selected_listing_keys']=list(self.selected)
            return
        photo_only=not self.app.legacy_evidence and self.app.stage=='B' and bool(PHOTO_REQUEST.search(query))
        mixed_fields,query=split_photo_request(query) if photo_only and len(products)==1 else ((),query)
        if photo_only and not mixed_fields and re.search(r'重量|多重|材质|品牌|宽度|高度|长度',query) and not re.search(r'(?:图片|照片|图)[上中里]',query):
            photo_only=False
        photo_hint=("只根据实际可见的商品图回答。相关位置未展示或看不清时返回 not_found，"
                    "不能据此断言商品没有该部件。") if photo_only else ""
        stamp=time.perf_counter();pages=[];assets=[];documents=[]
        metadata=read_fields(products,mixed_fields) if mixed_fields else None
        for product in products:
            doc=ensure_card(product)
            if photo_only:doc=ensure_photo_card(product,doc)
            assets.append(doc['image_asset'])
            documents.append(doc)
            pages.extend(photo_pages(doc,query) if photo_only else choose_pages(doc,query,1 if len(products)>1 else 2))
        for i,page in enumerate(pages,1):page['rank']=i
        row.update(routing={'reason':'exact_product_then_document_retrieval'},retrieval_seconds=time.perf_counter()-stamp,
            retrieval_trace={'scope':'full catalog exact identity + within-product page retrieval','global_index_records':147702,'selected_listing_keys':self.selected,
                             'image_assets':assets,'page_budget':2},
            retrieved_pages=[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source','listing_key')} for p in pages])
        row['routing']['strategy']='fields_and_photo' if metadata else 'source_photo' if photo_only else 'page_reader'
        if photo_only and len(pages)!=len(products):
            result={'action':'not_found','answer':'至少有一件商品缺少可用原图，无法核实这次的图片问题。','citations':[]}
        else:
            result=self.app.reader.predict(query,request,pages,self.out/'images',photo_hint) if photo_only else self.app.reader.predict(query,request,pages,self.out/'images')
            row.update(model=self.app.reader.model_name,model_revision=self.app.reader.revision)
        if self.app.retry and result['action']=='not_found' and not photo_only and not self.app.legacy_evidence:
            retry_start=time.perf_counter()
            rewritten=expand(query)
            alternative=alternate_pages(documents,pages,rewritten,2)
            row['retrieval_seconds']+=time.perf_counter()-retry_start
            retry={'max_extra_retrievals':1,'query':rewritten,'attempted':bool(alternative),
                   'first_pages':[p['page_id'] for p in pages],'first_raw_output':result['raw_output'],
                   'stop_reason':'no_new_matching_page'}
            if alternative:
                for i,page in enumerate(alternative,1):
                    alternative[i-1]={**page,'rank':i}
                second=self.app.reader.predict(query,request,alternative,self.out/'images')
                retry['stop_reason']='one_retry_completed'
                second['generation_seconds']+=result['generation_seconds']
                result=second
                row['retrieved_pages']=[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source','listing_key')} for p in alternative]
            row['retrieval_trace']['bounded_retry']=retry
        if self.app.retry and result['action']=='not_found' and self.app.legacy_evidence:
            retry_start=time.perf_counter();rewritten=expand(text);alternative=[]
            for doc in documents:alternative.extend(choose_pages(doc,rewritten,1 if len(products)>1 else 2))
            row['retrieval_seconds']+=time.perf_counter()-retry_start
            retry={'max_extra_retrievals':1,'query':rewritten,'attempted':rewritten!=text,'first_answer':result['answer'],
                   'first_pages':[p['page_id'] for p in pages]}
            if rewritten!=text and [p['page_id'] for p in alternative]!=[p['page_id'] for p in pages]:
                second=self.app.reader.predict(text,request,alternative,self.out/'images')
                retry['first_raw_output']=result['raw_output'];retry['stop_reason']='one_retry_completed'
                second['generation_seconds']+=result['generation_seconds'];result=second;pages=alternative
                row['retrieved_pages']=[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source','listing_key')} for p in pages]
            else:retry['stop_reason']='unchanged_candidates_or_query_no_second_read'
            row['retrieval_trace']['bounded_retry']=retry
        row.update(result)
        row['answer_source']='original_photo_page' if photo_only and pages else 'photo_unavailable' if photo_only else 'current_derived_page_images'
        if metadata:
            row['claims']=metadata['claims']
            row['answer_source']='metadata_fields_and_photo_page'
            if result['action']=='not_found' and result['answer']=='NOT_FOUND':
                row['answer']='图片证据不足，无法核实图片部分。'
            row['answer']=metadata['answer']+'\n'+row['answer']
            row['citations']=metadata['citations']+row['citations']
            row['retrieved_pages']=metadata['retrieved_pages']+row['retrieved_pages']
            for i,page in enumerate(row['retrieved_pages'],1):page['rank']=i
            if metadata['action']=='answer':row['action']='answer'

    def show_catalog(self,row,more=False):
        answer,self.catalog_offset,items=catalog_page(self.app.retriever.manifest,self.catalog_offset if more else 0)
        row.update(action='catalog',answer=answer,conversation_lane='catalog',task_kind='catalog',
            answer_source='verified_full_catalog_statistics',routing={'reason':'catalog_metadata_query'},catalog_categories=items,
            retrieval_trace={'catalog_records':147702,'database_sha256':self.app.retriever.manifest['database_sha256'],'next_category_offset':self.catalog_offset})

    def answer_overview(self,row,products,more=False,kind='overview'):
        stamp=time.perf_counter();answers=[];claims=[];docs=[]
        for product in products:
            key=product['listing_key'];previous_shown=self.overview_shown.get(key,[]);shown=previous_shown if more else []
            doc=ensure_card(product);docs.append(doc)
            answer,chunk=overview_chunk(product,doc,shown,kind=kind)
            keep=previous_shown if kind=='features' else shown
            self.overview_shown[key]=list(dict.fromkeys(keep+[i['item_id'] for i in chunk]))
            answers.append(answer);claims.extend(chunk)
        refs=list(dict.fromkeys(i['citation'] for i in claims))
        if not refs:refs=[p['listing_key']+':metadata' for p in products]
        pages=[]
        for doc in docs:
            pages.extend({k:p[k] for k in ('page_id','doc_id','page','sku','score','text_source','listing_key')} for p in doc['pages'] if p['page_id'] in refs)
        for ref in refs:
            if ref.endswith(':metadata'):
                pages.append({'page_id':ref,'doc_id':ref.rsplit(':',1)[0],'page':None,'sku':ref.split('@')[0],'score':1.0,
                              'text_source':'original_metadata_record','listing_key':ref.rsplit(':',1)[0]})
        for i,p in enumerate(pages,1):p['rank']=i
        row.update(action='answer',answer='\n\n'.join(answers),citations=refs,claims=claims,retrieved_pages=pages,
          retrieval_seconds=time.perf_counter()-stamp,conversation_lane='product',task_kind='overview',
          answer_source='verified_source_fields_and_notes',routing={'reason':'continue_verified_product_overview' if more else 'verified_product_overview'},
          retrieval_trace={'scope':'selected listing original metadata + derived PDF evidence','selected_listing_keys':list(self.selected),
                           'more':bool(more),'requested_product_task':kind,'new_source_items':[i['item_id'] for i in claims],'overview_exhausted':not bool(claims)})
        self.last_product_query={'question':row['question'],'listing_keys':list(self.selected)}


def handler_for(app):
    slots=threading.BoundedSemaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':return self.reply(200,{'status':'ready','version':'abo-full-v1','stage':app.stage,'mode':app.mode,
                'active_model':app.reader.model_name,'active_model_revision':app.reader.revision,
                'records':147702,'types':len(app.retriever.manifest['product_types']),'model_loaded':app.reader.reader is not None,'agent':False})
            if self.path=='/categories':return self.reply(200,app.retriever.manifest['product_types'])
            self.reply(404,{'error':'not_found'})
        def do_POST(self):
            if self.path not in {'/ask','/search'}:return self.reply(404,{'error':'not_found'})
            if not slots.acquire(False):return self.reply(429,{'error':'busy'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:raise ValueError('Body size must be1..16384')
                body=json.loads(self.rfile.read(size))
                if self.path=='/ask':
                    if not isinstance(body,dict) or set(body)-{'question','request','session_id'}:raise ValueError('Unexpected request fields')
                    result=app.respond(body.get('question'),body.get('request'),body.get('session_id'))
                else:
                    if not isinstance(body,dict) or set(body)-{'query','mode','top_k','filters'}:raise ValueError('Unexpected search fields')
                    with app.lock:
                        found=app.search_catalog(body.get('query'),body.get('mode',app.mode),body.get('top_k',5),body.get('filters'))
                    result={'items':[{**public_item(r),'rank':r['rank'],'score':r['score']} for r in found['items']],'seconds':found['seconds'],'trace':found['trace']}
                self.reply(500 if result.get('status')=='error' else 200,result)
            except (ValueError,TypeError) as error:self.reply(400,{'error':str(error)})
            except Exception as error:self.reply(500,{'error':type(error).__name__})
            finally:slots.release()
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['chat','ask','serve','check','search'],nargs='?',default='chat')
    parser.add_argument('--stage',choices=['A','B'],default='B');parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--question',default='');parser.add_argument('--sku',action='append',default=[]);parser.add_argument('--domain');parser.add_argument('--type')
    parser.add_argument('--port',type=int,default=8768);parser.add_argument('--visual',action='store_true');parser.add_argument('--retry',action='store_true')
    parser.add_argument('--legacy-evidence',action='store_true')
    parser.add_argument('--model',choices=['8b','2b'],default='8b');parser.add_argument('--model-dir',type=Path)
    args=parser.parse_args()
    app=FullApplication(args.stage,args.mode,visual=args.visual,retry=args.retry,legacy_evidence=args.legacy_evidence,
                        model=args.model,model_dir=args.model_dir)
    operation('启动全量ABO应用','A使用全库FTS5 BM25；B可使用全量语义/融合/重排；商品证据PDF/原图按需，业务边界先判断。',
       f'{args.command}: stage={args.stage}, mode={app.mode}, catalog=147702',verification='首次模型调用才加载GPU；不代表全量原图已下载。',status='启动')
    if args.command=='check':print(json.dumps({'status':'ready','records':147702,'types':len(app.retriever.manifest['product_types']),'stage':args.stage,'mode':app.mode,'model_loaded':False,'active_model':app.reader.model_name}))
    elif args.command=='search':
        result=app.search_catalog(args.question,app.mode,5,{k:v for k,v in [('domain',args.domain),('product_type',args.type)] if v})
        print(json.dumps({'items':[public_item(r) for r in result['items']],'seconds':result['seconds']},ensure_ascii=False,indent=2))
    elif args.command=='ask':
        request={'skus':args.sku,**{k:v for k,v in [('domain',args.domain),('product_type',args.type)] if v}}
        print(json.dumps(app.respond(args.question,request),ensure_ascii=False,indent=2))
    elif args.command=='chat':
        session=app.session()
        print('助手已启动，可以自由聊天、解释知识或帮写代码；具体商品资料才会检索ABO全量147702条、'+str(len(app.retriever.manifest['product_types']))+'类。可输入“商品列表”或“搜索 黑色双肩包”，输入“退出”结束。',flush=True)
        print('记录：'+str(session.transcript),flush=True)
        while not session.ended:
            try:text=input('\n你：')
            except (EOFError,KeyboardInterrupt):break
            if not text.strip():continue
            value=session.respond(text);print('助手：'+value['answer'],flush=True)
            if value['citations']:print('来源：'+'；'.join(value['citations']),flush=True)
            print('逐题记录：'+value['record_txt'],flush=True)
    else:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(app));print('Full ABO API http://127.0.0.1:'+str(args.port),flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
    app.close()


if __name__=='__main__':main()

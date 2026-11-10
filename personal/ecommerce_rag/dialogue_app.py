"""Versioned, fixed-flow dialogue RAG. Baseline modules are not modified."""
import argparse
from collections import deque
from copy import deepcopy
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time
import uuid

HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'rag_common'))
from common import WORKSPACE, file_hash, timestamp, write_json, operation, issue
from audit import format_question, ALL_RESULTS_TXT
from stage_a import snapshot
from dialogue_corpus import CORPUS
from dialogue_model import LocalDialogueModel, VERSION, MODEL_REVISION,is_short_thanks
from dialogue_retrieval import DialogueRetriever
from dialogue_evidence import field_excerpt,current_page_conflicts,conflict_notice
from dialogue_open import product_guard,scope_context,quoted_glosses,missing_history_reference,invented_fact_request,explicit_task_change
from dialogue_catalog import Catalogue,catalogue_request,ordinal_selection

SETTINGS={"version":VERSION,"corpus":"abo-dialogue-v2","model":"Qwen/Qwen3.5-2B","model_revision":MODEL_REVISION,
 "retrieval":"bm25","top_k":2,"single_product_top_k":1,"max_pixels":786432,"history_turns":6,"history_characters":6000,
 "router_max_new_tokens":128,"conversation_max_new_tokens":128,"reader_max_new_tokens":320,"max_input_tokens":7000,"generation_max_time_seconds":35,
 "conversation_permission_max_new_tokens":8,"requirement_topic_max_new_tokens":8,"max_semantic_route_passes":2,"max_retrieval_passes":1,
 "turn_facets_max_new_tokens":48,"needs_independent_of_action":True,"conflict_scope":"provided_page_explicit_inch_axis_vs_attribute_only",
 "ordinary_reply_input":"General knowledge uses current question and typed general history; product facts use current RAG evidence. Courtesy controls use a capability manifest.",
 "open_conversation":True,"general_max_new_tokens":384,"general_history":"general turns only; product answers excluded","scope_max_new_tokens":12,
 "catalogue_source":"current_corpus_metadata","pending_max_age_turns":6,"pending_relation_max_new_tokens":8,
 "max_sessions":32,"session_idle_seconds":1800,"evidence_policy":"Only current retrieved image-page quotes are product facts; past assistant replies and user claims are not evidence."}


def explicit_ids(text):
    return list(dict.fromkeys(re.findall(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])",text.upper())))


def redact_input(text):
    """Discard obvious unrelated secrets/contact details before either inference or logging."""
    original=text
    for pattern in (r'\bsk-[A-Za-z0-9_-]{12,}',r'\bAKIA[A-Z0-9]{16}\b',
                    r'(?i)Bearer\s+[A-Za-z0-9._-]{12,}',r'(?i)(?:api[_-]?key|token|password)\s*[:=]\s*[^\s,;，；]{8,}'):
        text=re.sub(pattern,'[敏感凭据已隐藏]',text)
    text=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[邮箱已隐藏]',text)
    text=re.sub(r'(?<![A-Za-z0-9])1[3-9][0-9]{9}(?![A-Za-z0-9])','[手机号已隐藏]',text)
    return text,text!=original


def width_limit(text):
    if not re.search(r"宽|width",text,re.I) or not re.search(r"空间|可用|限制|预留|最多|只有|只能|不超过",text):return None
    match=re.search(r"(\d+(?:\.\d+)?)\s*(厘米|公分|cm|英寸|inches|inch)",text,re.I)
    if not match:return None
    return float(match[1])*(2.54 if match[2].lower() in {"英寸","inch","inches"} else 1)


def literal_topic(text):
    fields={"width":r'宽(?:度)?|(?<![a-z])width(?![a-z])',"height":r'高度|(?<![a-z])height(?![a-z])',"length":r'长度|(?<![a-z])length(?![a-z])',
      "weight":r'重量|多重|(?<![a-z])weight(?![a-z])',"cleaning":r'清洁|擦洗|打理|维护|(?<![a-z])clean',"assembly":r'组装|安装|(?<![a-z])assembl',
      "age":r'年龄|几岁|(?<![a-z])age(?![a-z])',"capacity":r'承重',"material":r'材质|材料|(?<![a-z])material(?![a-z])',"color":r'颜色|(?<![a-z])color(?![a-z])'}
    matches=[key for key,pattern in fields.items() if re.search(pattern,text,re.I)]
    return matches[0] if len(matches)==1 else None


class DialogueApplication:
    def __init__(self,corpus=CORPUS,model=None,report_root=None,audit=True):
        self.corpus=Path(corpus)
        required=[self.corpus/'documents.json',self.corpus/'pages.jsonl',self.corpus/'index-manifest.json',WORKSPACE/'models/Qwen3.5-2B/config.json']
        missing=[str(p) for p in required if not p.is_file()]
        if missing:raise FileNotFoundError("Startup check: missing files: "+", ".join(missing))
        self.retriever=DialogueRetriever(self.corpus)
        self.documents=self.retriever.documents
        self.by_sku={d['sku']:d for d in self.documents}
        self.model=model or LocalDialogueModel()
        self.catalogue=Catalogue(self.corpus,self.documents)
        self.model.catalog_manifest=deepcopy(self.catalogue.manifest)
        self.report_root=report_root or WORKSPACE/'reports/ecommerce_rag'
        self.audit=audit
        self.lock=threading.RLock()
        self.sessions={}

    def session(self,initial_skus=None,session_id=None):
        if session_id is not None and (not isinstance(session_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",session_id)):
            raise ValueError("Invalid session_id")
        with self.lock:
            key=session_id or uuid.uuid4().hex
            now=time.monotonic()
            for old,s in list(self.sessions.items()):
                if now-s.last_used>SETTINGS['session_idle_seconds']:
                    s.close('idle_timeout');self.sessions.pop(old,None)
            if key in self.sessions:return self.sessions[key]
            if len(self.sessions)>=SETTINGS['max_sessions']:raise ValueError("Too many active sessions")
            session=DialogueSession(self,key,initial_skus)
            self.sessions[key]=session
            return session

    def respond(self,question,request=None,session_id=None):
        session=self.session(session_id=session_id)
        result=session.respond(question,request)
        if session.ended:
            with self.lock:self.sessions.pop(session.session_id,None)
        return result


class DialogueSession:
    def __init__(self,app,session_id,initial_skus=None):
        self.app,self.session_id=app,session_id
        skus=list(dict.fromkeys(s.upper() for s in initial_skus or []))
        if len(skus)>2 or not set(skus)<=set(app.by_sku):raise ValueError("Choose at most two known products")
        self.state={"skus":skus,"previous_skus":[],"candidates":[],"ordered_reference":None,"needs":{"purpose":"","space":"","focus":""},"active_topic":None,"pending":None}
        self.history=[]
        self.turn=0;self.ended=False;self.last_used=time.monotonic()
        name='abo-dialogue-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        self.output=app.report_root/name;self.output.mkdir(parents=True)
        config={**SETTINGS,"source_sha256":snapshot(self.output),"documents_sha256":file_hash(app.corpus/'documents.json'),
                "catalogue_manifest":app.catalogue.manifest,
                "pages_sha256":file_hash(app.corpus/'pages.jsonl'),"evaluation":"separate unlabelled dialogue; not original static benchmark"}
        write_json(self.output/'config.json',config)
        self.config=config;self.config_sha=file_hash(self.output/'config.json')
        self.transcript=self.output/'对话记录.txt'
        self.transcript.write_text('A阶段对话增强版本 '+VERSION+'\n普通历史用于交流衔接，用户需求是上下文；商品事实只取本轮证据，历史回答不是商品证据。\n','utf-8')

    def context(self):
        return {**deepcopy(self.state),"recent_dialogue_untrusted":deepcopy(self.history)}

    def _catalog(self,exclude=()):
        items=self.app.catalogue.items(exclude)
        self.state['candidates']=[item['sku'] for item in items]
        self.state['ordered_reference']='catalog'
        return self.app.catalogue.render(items)

    def _retire_pending(self,row,reason):
        if self.state['pending']:
            row.setdefault('pending_transitions',[]).append({'from':deepcopy(self.state['pending']),'to':None,'reason':reason})
            self.state['pending']=None

    def _show_catalogue(self,row,kind):
        request=row.get('request',{})
        if request.get('store_id','amazon.com')!='amazon.com' or request.get('as_of','2026-11-01')!='2026-11-01':
            row.update(conversation_lane='catalog',action='not_found',answer='当前目录仅对应amazon.com的ABO登记快照，没有所请求店铺或日期的目录。',routing={'reason':'catalogue_scope_guard'})
            return
        self._retire_pending(row,'superseded_by_'+kind)
        row.update(conversation_lane='catalog',action=kind,answer_source='current_catalog_metadata',
                   metadata_sources=deepcopy(self.app.catalogue.sources),routing={'reason':'local_catalogue_'+kind})
        if kind=='catalog':
            row['catalog_items']=self.app.catalogue.items()
            row['answer']=self._catalog()
        else:row['answer']=self.app.catalogue.coverage()

    def _select(self,skus):
        if len(skus)>1:self.state['ordered_reference']='selection'
        if skus!=self.state['skus']:
            self.state['previous_skus']=list(self.state['skus'])
            self.state['skus']=list(skus)

    def _resolve(self,decision,ids,supplied,question=''):
        if ids and supplied and not set(ids)<=set(supplied):return [],'问题编号与接口指定商品冲突，请确认以哪个为准。'
        chosen=ids or supplied
        if chosen:
            if not set(chosen)<=set(self.app.by_sku):return [],'知识库没有该商品编号，请从商品列表选择。'
            if len(chosen)>2:return [],'一次请明确选择最多两个商品。'
            return chosen,None
        ref=decision['reference']
        # Closed references are resolved only against an actually saved ordered list, never guessed IDs.
        if re.search(r'第[二2]个|第[二2]款|后一个|后一款',question):ref='second'
        elif re.search(r'第[一1]个|第[一1]款|前一个|前一款',question):ref='first'
        elif re.search(r'刚才那款|上一款',question):ref='previous'
        if ref=='other':return [],'请明确想换成哪款，或从新列表中选择。'
        if ref in {'first','second'}:
            candidates=self.state['candidates'] if self.state.get('ordered_reference')=='catalog' else self.state['skus'] if len(self.state['skus'])>1 else self.state['candidates']
            index=0 if ref=='first' else 1
            if len(candidates)<=index:return [],'当前没有明确的有序商品列表，请提供商品编号。'
            return [candidates[index]],None
        if ref=='previous':
            if len(self.state['previous_skus'])!=1:return [],'之前涉及的商品不唯一，请明确商品编号。'
            return list(self.state['previous_skus']),None
        if len(self.state['skus'])>1 and ref not in {'both'}:
            return [],'当前有两个商品，“它”指哪一个？请提供编号或明确说分别比较。'
        return list(self.state['skus']),None

    def _needs_update(self,decision):
        if decision['intent']!='requirements' and not decision.get('need_confirmed'):return
        for key,value in decision['needs'].items():
            if value:
                # An explicit, unit-checked personal space constraint has one canonical slot.
                # This is not a product parameter, and must replace the old space constraint.
                if width_limit(value) is not None:
                    self.state['needs']['space']=value;self.state['active_topic']='width'
                else:
                    self.state['needs'][key]=value
                    topic=decision['topic'] if decision['topic'] not in {'unknown','comparison'} else literal_topic(value)
                    self.state['active_topic']=topic or ('fit' if key=='purpose' else None)

    def _answer_product(self,row,effective,skus,topic):
        row.update(conversation_lane='product',product_query=effective[:900],product_topic=topic,product_skus=list(skus))
        pages,retrieval,seconds=self.app.retriever.search(effective,skus,topic,{**self.state['needs'],'_active_topic':self.state.get('active_topic')})
        row.update(retrieval_seconds=seconds,retrieval_trace=retrieval,
            retrieved_pages=[{k:p[k] for k in ('page_id','doc_id','page','sku','rank','score','text_source')} for p in pages],
            request={"skus":skus,"store_id":"amazon.com","as_of":"2026-11-01"})
        actual_topic=retrieval['normalized_topic'] if topic=='fit' else topic
        if re.search(r'电池|毫安|mAh|battery',effective,re.I) and not any(re.search(r'电池|毫安|mAh|battery',p['text'],re.I) for p in pages):
            claims,images=[],[]
            row['retrieval_trace']['missing_requested_subject']='battery field absent in this turn\'s retrieved text; conservative abstention'
        else:
            claims,images=self.app.model.read(effective,pages,actual_topic,self.state['needs'],self.output/'images',row['model_calls'])
        for c in claims:c['quote']=field_excerpt(c['quote'],actual_topic)
        conflicts=current_page_conflicts(pages,actual_topic) if claims else []
        row.update(claims=claims,source_conflicts=conflicts,image_paths=images,answer_source='current_image_quotes',
            citations=list(dict.fromkeys([c['page_id'] for c in claims]+[c['page_id'] for c in conflicts])),action='answer' if claims else 'not_found',
            answer=self._render(claims,topic,conflicts) if claims else '本轮提供的资料没有足够证据回答这项信息，暂不作推断。')
        if claims and row.get('product_followup_task'):row['answer']+=quoted_glosses(claims)
        row['routing']['reason']='grounded_evidence' if claims else 'missing_evidence'
        self.state['pending']=None

    def _open_dispatch(self,row,question,ids,request):
        text=question.strip()
        controls={'退出','结束','exit','quit','q','重置','清空','清空上下文','reset','商品列表','列表','list'}
        pending=self.state['pending']
        if not 0<len(text)<=3000 or text.casefold() in controls or is_short_thanks(text):return False
        if pending and pending.get('missing')=='intent_confirmation' and text.casefold() in {'确认','是','对','yes','不是','否','取消确认','no'}:return False
        if pending and explicit_task_change(question):
            self._retire_pending(row,'explicit_stop_or_topic_change');pending=None
        if missing_history_reference(question,self.history,self.state['skus']):
            row.update(conversation_lane='general',action='clarify',answer='当前会话没有可引用的先前内容，可能已新建或重置。请把要继续解释的问题或主题再告诉我。',
                       scope_route={'domain':'unclear','method':'missing_history_check'},routing={'reason':'no_previous_context'})
            return True
        metadata=catalogue_request(question) if not ids else None
        if metadata=='mixed_catalog':
            self._retire_pending(row,'new_batch_catalogue_request')
            row.update(conversation_lane='catalog',action='clarify',answer='可以先列出商品目录；参数需要逐款检索证据。请明确最多两个商品和要查的字段，或先让我展示目录。',routing={'reason':'catalogue_fields_need_product_scope'})
            return True
        if invented_fact_request(question) and (ids or self.state['skus']):
            # This is a request about product facts, never a personal preference.
            # Handle it before any model-generated need slots can mutate the session.
            self._retire_pending(row,'invention_request_cannot_be_user_need')
            decision={'reference':'current'}
            skus,problem=self._resolve(decision,ids,[s.upper() for s in request.get('skus',[])],text)
            row.update(conversation_lane='product',scope_route={'domain':'product','method':'invention_fact_guard'},routing={'reason':'invention_requires_current_evidence'})
            if problem or not skus:
                row.update(action='clarify',answer=problem or '不能编造商品参数；请明确要核对的商品。')
            elif request.get('store_id','amazon.com')!='amazon.com' or request.get('as_of','2026-11-01')!='2026-11-01':
                row.update(action='not_found',answer='不能编造参数；当前没有所请求店铺或日期的商品资料。')
            else:
                self._select(skus)
                self._answer_product(row,question,skus,literal_topic(question) or 'unknown')
                row['answer']='不能编造商品参数。'+row['answer']
            return True
        relative=ordinal_selection(text) is not None and bool(self.state['candidates'] or self.state['skus'])
        field=literal_topic(question)
        general_explanation=bool(re.search(r'什么是|什么叫|是什么意思|概念|一般|通常|原理|为什么|定义|编程|代码|程序',question)) and not bool(re.search(r'这把|这款|所选|当前商品|资料|规格表',question))
        field_request=bool(field) and not general_explanation
        forced=metadata or ('product' if relative or field_request else product_guard(question,self.state['skus'],self.history,ids))
        domain=forced or self.app.model.classify_scope(question,scope_context(self.history,self.state['skus'],self.state['needs'],pending,self.app.catalogue.manifest),row['model_calls'])
        row['scope_route']={'domain':domain,'method':'explicit_scope_guard' if forced else 'local_model_scope'}
        if domain in {'catalog','coverage'}:
            self._show_catalogue(row,domain);return True
        if domain in {'general','unclear'}:
            if domain=='unclear' and self.state['skus'] and any(self.state['needs'].values()):
                row['conversation_lane']='product';return False
            if pending:
                relation=self.app.model.pending_relation(question,pending,row['model_calls'])
                row['pending_relation']=relation
                if relation=='replace':self._retire_pending(row,'new_independent_question')
                elif relation=='resume':
                    row['conversation_lane']='product';return False
            row['conversation_lane']='general'
            message=self.app.model.open_converse(question,self.history,row['model_calls'],clarify_only=domain=='unclear')
            row.update(action='chat' if domain=='general' else 'clarify',answer=message,
                       answer_source='local_model_general_knowledge',routing={'reason':'open_general_conversation' if domain=='general' else 'neutral_topic_clarification'})
            if row['model_calls'] and row['model_calls'][-1].get('stop_reason')=='token_or_time_budget':
                row['status']='partial';row['answer']+='\n（本轮达到输出预算，可让我继续。）'
            return True
        if domain=='business':
            self._retire_pending(row,'new_business_request')
            row.update(conversation_lane='business',action='requires_tool',answer='我目前没有联网或实时业务工具，无法核实今天的天气、价格、库存等实时信息，也不能代办订单或退款。可以解释一般原理，或查询现有商品资料。',routing={'reason':'scope_business_boundary'})
            return True
        if domain=='mixed':
            row.update(conversation_lane='control',action='clarify',answer='这句话包含一般知识和具体商品查询，你希望先处理哪部分？商品事实需要单独检索证据。',routing={'reason':'mixed_scope_order_clarification'})
            return True
        row['conversation_lane']='product'
        if domain=='product_followup':
            previous=next((h for h in reversed(self.history) if h.get('product_query')),None)
            if previous:
                row['product_followup_task']={k:previous[k] for k in ('product_query','product_topic','product_skus')}
            else:
                row.update(action='clarify',answer='当前短期历史中没有可重新取证的商品问题，请说明要查哪项资料。',routing={'reason':'missing_product_followup_context'})
                return True
        return False

    def _render(self,claims,topic,conflicts=()):
        text='资料事实（来自本轮页面，原文描述不等于实测结论）：\n'+'\n'.join(f"{c['sku']}：{c['quote']}【第{c['page']}页】" for c in claims)
        needs=self.state['needs']
        if topic=='policy_history':text+='\n这是历史商品描述，不能用它确认当前店铺的退货或保修承诺。'
        limit=width_limit(needs['space'])
        if limit is not None and topic in {'fit','width'}:
            comparisons=[]
            for c in claims:
                match=re.search(r"width\s*[:：/]?\s*(\d+(?:\.\d+)?)\s*(inches|inch|厘米|cm)",c['quote'],re.I)
                if not match:continue
                width=float(match[1])*(2.54 if match[2].lower() in {'inch','inches'} else 1)
                comparisons.append(f"{c['sku']}的资料宽度换算为{width:g}厘米；你的可用宽度为{limit:g}厘米，"+('仅按宽度比较超过限制。' if width>limit else '仅按宽度比较未超过限制。'))
            if comparisons:text+='\n条件性分析：'+' '.join(comparisons)+'其他尺寸和实际摆放余量仍需核对。'
            else:text+='\n条件性分析：未取得可核对的宽度数值，暂不判断是否放得下。'
        elif any(needs.values()):
            text+='\n需求上下文：'+'；'.join(v for v in needs.values() if v)+'。这里只结合资料已报告的信息，不能保证实际舒适度或耐用性。'
        if conflicts:
            text+='\n'+conflict_notice(conflicts)
            if limit is not None and topic in {'fit','width'}:text+='\n以上换算仅针对规格属性，来源冲突未消解，不能确认一定放得下。'
        return text

    def _save(self,row,before):
        if 'image_paths' not in row:
            row['image_paths']=[p for call in row['model_calls'] for p in call.get('image_paths',[])]
        row.update(context_before=before,context_after=self.context(),model_load_count=self.app.model.load_count,
                   model_call_count=sum(x.get('model_executed',True) for x in row['model_calls']),model_attempt_count=len(row['model_calls']),generation_seconds=sum(x.get('generation_seconds',0) for x in row['model_calls']),
                   peak_cuda_gib=max((x.get('peak_cuda_gib',0) for x in row['model_calls']),default=0),
                   raw_output='\n'.join(x.get('raw_output','') for x in row['model_calls']))
        row['model']='Qwen/Qwen3.5-2B' if row['model_call_count'] else None
        row['model_revision']=MODEL_REVISION if row['model_call_count'] else None
        with (self.output/'results.jsonl').open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        content=format_question(row)+'对话范围/来源：'+json.dumps({'lane':row.get('conversation_lane'),'answer_source':row['answer_source'],'scope_route':row.get('scope_route'),'metadata_sources':row.get('metadata_sources'),'pending_transitions':row.get('pending_transitions'),'pending_relation':row.get('pending_relation')},ensure_ascii=False)+'\n增强上下文（本轮后）：'+json.dumps(row['context_after'],ensure_ascii=False)+'\n模型分阶段调用：'+json.dumps(row['model_calls'],ensure_ascii=False)+'\n证据摘录：'+json.dumps(row.get('claims',[]),ensure_ascii=False)+'\n本轮页面来源冲突：'+json.dumps(row.get('source_conflicts',[]),ensure_ascii=False)+'\n整轮耗时（秒）：'+str(row['total_seconds'])+'\n\n'
        paths=[self.output/'逐题结果.txt']+([ALL_RESULTS_TXT] if self.app.audit else [])
        for path in paths:
            with path.open('a',encoding='utf-8',newline='\n') as stream:stream.write(content)
        with self.transcript.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(f"\n[{timestamp()}] 第{self.turn}轮\n你：{row['question']}\n助手：{row['answer']}\n动作：{row['action']}\n引用：{row['citations']}\n上下文：{json.dumps(row['context_after'],ensure_ascii=False)}\n实际模型调用{row['model_call_count']}次；阶段尝试{row['model_attempt_count']}次；整轮{row['total_seconds']:.3f}秒\n")
        write_json(self.output/'session.json',{"session_id":self.session_id,"ended":self.ended,"turns":self.turn,"context":self.context(),"updated_at":timestamp()})
        return {k:row[k] for k in ('question_id','action','answer','citations','retrieved_pages','status','model_call_count','model_load_count','total_seconds','routing','conversation_lane','answer_source')} | {
            "session_id":self.session_id,"current_skus":list(self.state['skus']),"record_txt":str(self.output/'逐题结果.txt'),"conversation_txt":str(self.transcript),
            'catalog_items':row.get('catalog_items',[]),'metadata_sources':row.get('metadata_sources',{})}

    def respond(self,question,request=None):
        if not isinstance(question,str):raise ValueError('question must be a string')
        question,redacted=redact_input(question)
        if request is not None and (not isinstance(request,dict) or set(request)-{'skus','store_id','as_of'}):raise ValueError('Only public request context is accepted')
        request=dict(request or {})
        if 'skus' in request and (not isinstance(request['skus'],list) or any(not isinstance(s,str) for s in request['skus'])):raise ValueError('skus must be string array')
        if 'as_of' in request:date.fromisoformat(request['as_of'])
        with self.app.lock:
            if self.ended:raise ValueError('Session already ended')
            started=time.perf_counter();before=self.context();self.turn+=1;self.last_used=time.monotonic()
            row={"question_id":self.session_id+f'-{self.turn:03d}',"question":question,"reference_answer":"独立对话请求，参考标签不进入预测", "reference_pages":[],
                "run_name":self.output.name,"run_directory":str(self.output.relative_to(WORKSPACE)) if self.output.is_relative_to(WORKSPACE) else str(self.output),
                "session_id":self.session_id,"turn_number":self.turn,"timestamp":timestamp(),"status":"ok","schema_version":VERSION,
                "config_sha256":self.config_sha,"source_sha256":self.config['source_sha256'],"model_calls":[],"retrieved_pages":[],"citations":[],
                "request":request,"privacy_redacted":redacted,"retrieval_seconds":0.0,"routing":{},"retrieval_trace":{},"metrics":{"evaluation":"independent_dialogue_unlabelled"},"answer_source":"dialogue_control"}
            def reply(action,answer,reason):row.update(action=action,answer=answer);row['routing'].update(reason=reason)
            text=question.strip();ids=explicit_ids(text)
            try:
                if self.state['pending'] and self.turn-self.state['pending'].get('created_turn',self.turn)>SETTINGS['pending_max_age_turns']:
                    self._retire_pending(row,'clarification_expired')
                if self._open_dispatch(row,question,ids,request):pass
                elif not 0<len(text)<=3000:reply('clarify','请输入1到3000个字符的问题。','input_length')
                elif text.casefold() in {'退出','结束','exit','quit','q'}:self.ended=True;reply('exit','对话已结束。','user_exit')
                elif text.casefold() in {'重置','清空','清空上下文','reset'}:
                    self.state={"skus":[],"previous_skus":[],"candidates":[],"ordered_reference":None,"needs":{"purpose":"","space":"","focus":""},"active_topic":None,"pending":None};self.history=[]
                    reply('reset','已清空本会话的商品、需求和历史；模型仍可复用。','reset_all_context')
                elif is_short_thanks(text):
                    row['routing']={'mode':'finite_courtesy_grammar'}
                    reply('chat',self.app.model.converse(question,row['model_calls']),'short_courtesy')
                elif request.get('store_id','amazon.com')!='amazon.com' or request.get('as_of','2026-11-01')!='2026-11-01':
                    reply('not_found','当前只有amazon.com的ABO登记快照，没有所请求站点或日期的资料。','scope_guard')
                elif self.state['pending'] and self.state['pending'].get('missing')=='intent_confirmation' and text.casefold() in {'确认','是','对','yes'}:
                    pending=self.state['pending'];row['routing']={'mode':'explicit_confirmation_control','confirmed_topic':pending['topic']}
                    if request.get('skus') and [s.upper() for s in request['skus']]!=self.state['skus']:
                        reply('clarify','请求指定商品与本次待确认商品不一致，请重新选择。','confirmation_scope_conflict')
                    else:self._answer_product(row,pending['question'],list(self.state['skus']),pending['topic'])
                elif self.state['pending'] and self.state['pending'].get('missing')=='intent_confirmation' and text.casefold() in {'不是','否','取消确认','no'}:
                    self.state['pending']=None;reply('clarify','已取消本次意图确认，请重新说明问题。','confirmation_cancelled')
                elif row.get('product_followup_task'):
                    previous=row['product_followup_task']
                    if self.state['skus']!=previous['product_skus'] or request.get('skus') and [s.upper() for s in request['skus']]!=previous['product_skus']:
                        reply('clarify','商品选择已改变或与请求不一致，请确认要重述哪款商品的资料。','product_followup_scope_conflict')
                    else:
                        row['routing']={'reason':'retrieve_again_for_product_followup'}
                        self._answer_product(row,previous['product_query']+'\n本轮追问：'+question,list(self.state['skus']),previous['product_topic'])
                elif re.search(r'实时|现在.*(?:价格|多少钱|售价|库存|有货)|(?:查|跟踪).*订单|物流|快递|包裹|下单|退款|取消订单|有货|库存|多少钱|售价',text,re.I):
                    reply('requires_tool','当前能查静态商品资料，尚未连接实时价格、库存、订单或交易工具，不能替你查询或执行。','live_business_boundary')
                elif re.search(r'店铺.*(?:退|保修|售后)|(?:当前|现在|现行).*(?:退货|售后|保修)|(?:退货|售后)政策',text):
                    reply('not_found','当前没有可确认有效期与适用店铺的现行售后政策。原始商品历史文案不能代替今天的店铺承诺。','current_policy_boundary')
                elif text in {'商品列表','列表','list'}:self._show_catalogue(row,'catalog')
                else:
                    bare=bool(ids and len(ids)==1 and re.fullmatch(r'(?:选择|选中|select)?\s*'+ids[0],text,re.I))
                    relative=ordinal_selection(text)
                    if bare:
                        decision={"intent":"select","topic":"unknown","reference":"explicit","message":"","needs":{"purpose":"","space":"","focus":""}}
                        row['routing']={"mode":"explicit_selection_control",**decision}
                    elif relative is not None and (self.state['candidates'] or self.state['skus']):
                        decision={"intent":"select","topic":"unknown","reference":"first" if relative==0 else "second","message":"","needs":{"purpose":"","space":"","focus":""}}
                        row['routing']={"mode":"ordered_selection_control",**decision}
                    else:
                        decision=self.app.model.route(question,self.context(),row['model_calls'])
                        row['routing']={"mode":"local_model_semantic",**decision}
                        facets=self.app.model.analyze_turn(question,self.context(),row['model_calls'])
                        row['routing']['turn_facets']=facets
                        task=facets['task'];need=facets['need']
                        original_intent=decision['intent']
                        # Operation and personal need are independent: a fact request can also
                        # update focus, and selecting products cannot discard its requested field.
                        if task in {'live','execute','policy'}:decision['intent']='live' if task=='execute' else task
                        elif task=='fact':decision['intent']='product'
                        elif need!='none' and original_intent!='product':decision['intent']='requirements'
                        elif task=='meta' and not ids and literal_topic(text) is None:decision['intent']='chat'
                        concrete_space=width_limit(text) is not None
                        if task=='none' and original_intent=='requirements' and decision['needs'].get('purpose') and need in {'none','focus'}:
                            # Purpose is a more specific personal slot than a generic focus label.
                            need='purpose';decision['intent']='requirements'
                            row['routing']['specific_need_preserved']='purpose'
                        if concrete_space and task not in {'live','execute','policy'}:
                            need='space'
                            if task!='fact' and original_intent!='product':decision['intent']='requirements'
                        confirmed=need!='none' and (task=='fact' or original_intent!='product' or concrete_space)
                        if confirmed and task not in {'live','execute','policy'}:
                            decision['needs']={k:question[:180] if k==need else '' for k in ('purpose','space','focus')}
                            decision['need_confirmed']=True
                        elif original_intent!='requirements':
                            decision['needs']={k:'' for k in ('purpose','space','focus')}
                        if original_intent=='product' and task=='none' and need!='none':
                            row['routing']['facet_disagreement']='retain_explicit_fact_query; need-only classification cannot replace it'
                    intent=decision['intent'];topic=decision['topic']
                    if decision.get('topic_fallback'):
                        row['status']='guarded'
                    if (intent=='requirements' or decision.get('need_confirmed')) and any(decision['needs'].values()):
                        value=next(v for v in decision['needs'].values() if v)
                        # Need-only words can mention a negated former topic; classify the latest
                        # positive focus rather than choosing the first matched property word.
                        if decision['needs'].get('purpose'):
                            # A use such as office work is valid context, not necessarily one of
                            # the finite visual/property topics; do not reject the need for that.
                            if intent!='product' or topic=='unknown':topic='fit'
                        elif decision['needs'].get('space') and width_limit(value) is not None:
                            topic='width'
                        elif topic=='unknown' or decision.get('need_confirmed'):
                            topic=self.app.model.focus_topic(value,row['model_calls'])
                        decision['topic']=topic;row['routing']['requirement_topic_check']=topic
                        if topic in {'unknown','fit'} and decision.get('need_confirmed') and self.state.get('active_topic') and decision['needs'].get('focus'):
                            # A deictic continuation has no new focus content: retain the user's
                            # existing need instead of replacing it with an empty reference.
                            topic='fit' if topic=='fit' else self.state['active_topic'];intent='product'
                            decision['needs']={k:'' for k in ('purpose','space','focus')}
                            decision['need_confirmed']=False
                            row['routing']['retained_explicit_user_need']=True
                        self._needs_update(decision)
                    property_topic=literal_topic(text)
                    if intent=='select' and ids and not bare and property_topic:
                        intent='product';topic=property_topic
                        row['routing']['selection_query_preserved']=topic
                    if intent=='select' and ids and not bare and not property_topic:
                        # A bounded second semantic check of the residual request handles composite
                        # selection+question inputs; it is not a second retrieval or an Agent loop.
                        residual=text
                        for sku in ids:residual=re.sub(re.escape(sku),'',residual,flags=re.I)
                        context=self.context();context['skus']=ids
                        extra=self.app.model.route(residual,context,row['model_calls'])
                        row['routing']['selection_residual_check']=extra
                        if extra['intent']=='product':intent='product';topic=extra['topic']
                    # A safety boundary, not a replacement for model-based semantic understanding.
                    # In particular, a misclassified reference or user specification assertion cannot use free chat.
                    if intent=='chat' and (ids or decision['reference'] in {'previous','first','second','other','both'} or
                            re.search(r'忽略.*(?:资料|页面|证据)|(?:它|这款|这把).*(?:宽|重|高|材质)|(?:宽|重|高)\s*\d|怎么.*(?:清洁|打理|擦|组装)|如何.*(?:清洁|打理|组装)',text)):
                        row['routing']['application_guard']='product_reference_or_assertion_cannot_use_chat'
                        intent='select' if decision['reference']=='other' else 'product'
                    if intent in {'unsupported','clarify'} and (ids or self.state['skus']) and width_limit(text) is None and (invented_fact_request(text) or re.search(r'忽略.*(?:资料|页面|证据)|(?:宽|重|高)\s*\d',text)):
                        intent='product';row['routing']['application_guard']='specification_assertion_requires_current_evidence'
                    if intent=='product' and topic=='unknown':
                        # Canonical property vocabulary, used only after a product request is identified.
                        matched=literal_topic(text)
                        if matched:
                            topic=matched;row['routing']['canonical_property_guard']=topic
                    if topic=='unknown' and self.state.get('active_topic') and self.state['skus'] and (intent=='product' or
                         intent in {'clarify','chat'} and re.search(r'(?:按|这个|这项|那|继续|刚才|前面).*(?:关注|要求|说明|资料|回答|分析)',text)):
                        intent='product';topic=self.state['active_topic'];row['routing']['active_need_topic']=topic
                    if intent=='requirements' and self.state['pending'] and self.state['pending'].get('missing')=='needs':
                        self._needs_update(decision);intent='product';topic=self.state['pending']['topic']
                        row['routing']['resume_requirement_clarification']=True
                    row['routing'].update(effective_intent=intent,effective_topic=topic)
                    if intent=='live':reply('requires_tool','尚未连接实时业务工具，不能代办退货、退款、提交申请或下单，也不能查询实时价格、库存、订单或物流。','semantic_live_boundary')
                    elif intent=='policy':reply('not_found','当前没有经过有效期核验的店铺售后政策；历史产品描述不代表现行承诺。','semantic_policy_boundary')
                    elif intent=='chat' and not ids:
                        message=self.app.model.converse(question,row['model_calls'])
                        if message is None:
                            field=literal_topic(question)
                            skus,problem=self._resolve(decision,ids,[s.upper() for s in request.get('skus',[])],text)
                            if field and skus and not problem:
                                row['routing']['non_meta_fallback']='literal_property_resolved_against_current_product_then_image_evidence'
                                self._select(skus);self._answer_product(row,question,skus,field)
                            else:
                                if self.state['skus'] and self.state.get('active_topic') and any(self.state['needs'].values()):
                                    reply('clarify','我还不能确定本轮意图。是否继续按当前需求（'+'；'.join(v for v in self.state['needs'].values() if v)+'）查询所选商品？回复“确认”再取证，或重新说明。','confirm_existing_need_intent')
                                    self.state['pending']={'question':question[:900],'topic':self.state['active_topic'],'missing':'intent_confirmation'}
                                else:
                                    reply('clarify','这项问题需要查商品资料。请明确商品和关注字段；我会先取证，再回答。','non_meta_chat_blocked')
                                    if field:self.state['pending']={'question':question[:900],'topic':field,'missing':'product'}
                        else:
                            reply('chat',message,'semantic_conversation');row['answer_source']='local_model_conversation'
                    elif intent=='catalog':self._show_catalogue(row,'catalog')
                    elif intent=='requirements':
                        self._needs_update(decision)
                        if not any(decision['needs'].values()):reply('clarify','你希望更新用途、空间限制还是关注点？请补充具体要求。','missing_requirement_span')
                        else:reply('requirements','已记录你的需求：'+'；'.join(v for v in self.state['needs'].values() if v)+'。后续会优先核对资料中相关信息，无法据此保证实际体验。','requirements_updated')
                    elif intent in {'clarify','unsupported'} and not self.state['pending']:
                        if self.state['skus'] and self.state.get('active_topic') and any(self.state['needs'].values()):
                            reply('clarify','我还不能确定本轮意图。是否继续按当前需求（'+'；'.join(v for v in self.state['needs'].values() if v)+'）查询所选商品？回复“确认”再取证，或重新说明。','confirm_existing_need_intent')
                            self.state['pending']={'question':question[:900],'topic':self.state['active_topic'],'missing':'intent_confirmation'}
                        else:reply('clarify' if intent=='clarify' else 'unsupported','请说明想了解哪项商品信息；也可以询问我的能力或查看商品列表。','intent_not_actionable')
                    else:
                        skus,problem=self._resolve(decision,ids,[s.upper() for s in request.get('skus',[])],text)
                        if problem or not skus:
                            if ids and not set(ids)<=set(self.app.by_sku):self.state['skus']=[]
                            if decision['reference']=='other':
                                reply('clarify',problem+'\n'+self._catalog(exclude=self.state['skus']),'choose_alternative')
                            else:
                                reply('clarify',problem or '你想查询哪款商品？可以提供编号，或输入“商品列表”选择。','missing_or_ambiguous_product')
                            self.state['pending']={"question":question[:900],"topic":topic,"missing":"product"}
                        else:
                            self._select(skus)
                            pending=self.state['pending']
                            if pending and pending.get('missing')=='product' and literal_topic(text) and not bare and relative is None:
                                self._retire_pending(row,'explicit_current_query_replaces_missing_product_question');pending=None
                            if pending and pending.get('missing')=='intent_confirmation':
                                pending=None;self.state['pending']=None
                            if intent=='select' and not pending:
                                reply('selected','已选择 '+', '.join(skus)+'。可以问功能、使用维护、参数或图片信息。','selected_product')
                            else:
                                effective=question
                                if pending and ((intent=='select' or topic=='unknown') and (ids or decision['reference'] in {'first','second','previous','explicit'}) or pending.get('missing')=='needs'):
                                    effective=pending['question']+'\n用户补充选择：'+question
                                    if topic=='unknown':topic=pending['topic']
                                    row['resumed_question']=pending['question']
                                if topic=='fit' and not any(self.state['needs'].values()):
                                    reply('clarify','你希望判断哪方面适合：摆放空间、适用年龄，还是其他明确需求？','fit_needs_required')
                                    self.state['pending']={"question":effective[:900],"topic":"fit","missing":"needs"}
                                else:
                                    self._needs_update(decision)
                                    self._answer_product(row,effective,skus,topic)
            except KeyboardInterrupt:
                reply('cancelled','本轮已取消。','cancelled');row['status']='cancelled';self.ended=True
            except Exception as error:
                row.update(error=type(error).__name__+': '+str(error),status='guarded' if isinstance(error,ValueError) else 'error')
                if not row['routing']:self.state=deepcopy({k:v for k,v in before.items() if k!='recent_dialogue_untrusted'})
                reply('clarify' if isinstance(error,ValueError) else 'error','本轮理解或输出未通过检查，暂不能确认处理成功。请换个说法说明问题或目标；原始输出已保存。','conservative_failure')
                if self.app.audit:
                    row['issue_id']=issue('A对话增强输出/运行异常：'+row['error'],'检查本轮语义结构、证据摘录及模型轨迹。','保留原始输出，保守返回并允许后续继续；开发期按失败原因修正，验收期保留失败。',status='待复查')
            if self.state['pending']:self.state['pending'].setdefault('created_turn',self.turn)
            # Only this bounded history is available to routing. Reader never receives historical answers.
            row.setdefault('conversation_lane','product' if row['action'] in {'answer','not_found','requirements','selected','catalog','requires_tool'} else 'control')
            if row['action'] not in {'reset','exit'}:
                entry={"user":question[:600],"assistant_untrusted":row['answer'][:1600],"action":row['action'],'lane':row['conversation_lane']}
                if 'product_query' in row:entry.update({k:row[k] for k in ('product_query','product_topic','product_skus')})
                self.history.append(entry)
                self.history=self.history[-SETTINGS['history_turns']:]
                while sum(len(x['user'])+len(x['assistant_untrusted']) for x in self.history)>SETTINGS['history_characters']:self.history.pop(0)
            row['total_seconds']=time.perf_counter()-started
            result=self._save(row,before)
            if self.app.audit:operation('A增强对话本轮处理','确定性控制→本地模型语义分流→受限上下文→商品过滤/BM25→图片原文与引用校验。',row['action'],[self.output/'逐题结果.txt'],f"实际模型调用{row['model_call_count']}次，耗时{row['total_seconds']:.3f}秒；新对话不计入旧76题分数。",related_issues=row.get('issue_id','无'))
            return result

    def close(self,reason):
        self.ended=True
        write_json(self.output/'session.json',{"session_id":self.session_id,"ended":True,"reason":reason,"turns":self.turn,"context":self.context()})


def chat(app,skus):
    session=app.session(initial_skus=skus)
    print('智能助手已启动。可以直接聊一般知识、学习或写作，也可查看商品列表、查询商品资料；输入“退出”结束。',flush=True)
    print('记录：'+str(session.transcript),flush=True)
    try:
        while not session.ended:
            try:q=input('\n你：')
            except EOFError:break
            if not q.strip():continue
            result=session.respond(q)
            print('助手：'+result['answer'],flush=True)
            print('逐题记录：'+result['record_txt'],flush=True)
    except KeyboardInterrupt:print('\n对话已停止。',flush=True)
    finally:session.close('terminal_closed');print('对话记录：'+str(session.transcript),flush=True)


def handler_for(app,slots=None):
    slots=slots or threading.BoundedSemaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,{"status":"ready","version":VERSION,"agent":False,"model_loaded":app.model.reader is not None}) if self.path=='/health' else self.reply(404,{"error":"not_found"})
        def do_POST(self):
            if self.path!='/ask':return self.reply(404,{"error":"not_found"})
            if not slots.acquire(blocking=False):return self.reply(429,{"error":"busy"})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:raise ValueError('Request size must be 1..16384 bytes')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict) or set(body)-{'question','request','session_id'}:raise ValueError('Only question/request/session_id are allowed')
                if not isinstance(body.get('question'),str):raise ValueError('question must be a string')
                result=app.respond(body['question'],body.get('request'),body.get('session_id'))
                self.reply(500 if result['status']=='error' else 200,result)
            except (ValueError,TypeError) as error:self.reply(400,{"error":str(error)})
            except Exception as error:self.reply(500,{"error":type(error).__name__})
            finally:slots.release()
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['chat','ask','serve','check'])
    parser.add_argument('--question',default='');parser.add_argument('--sku',action='append',default=[])
    parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    app=DialogueApplication()
    if args.command=='chat':chat(app,args.sku)
    elif args.command=='ask':print(json.dumps(app.session(initial_skus=args.sku).respond(args.question),ensure_ascii=False,indent=2))
    elif args.command=='check':print(json.dumps({"status":"ready","version":VERSION,"products":len(app.documents),"pages":len(app.retriever.base.pages),"model_loaded":False},ensure_ascii=False))
    else:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(app))
        print(f'Local dialogue API: http://127.0.0.1:{args.port}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()


if __name__=='__main__':main()

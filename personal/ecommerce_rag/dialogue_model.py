"""Bounded semantic routing and image evidence extraction using ONE local model."""
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "rag_common"))
from common import WORKSPACE
from core import VLMReader, render_page
from dialogue_open import DOMAINS,general_messages,validate_general_reply

VERSION = "a-dialogue-v3.3"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
INTENTS = {"chat", "product", "select", "catalog", "requirements", "live", "policy", "clarify", "unsupported"}
TOPICS = {"unknown", "width", "height", "length", "weight", "color", "material", "brand", "model", "assembly", "cleaning", "capacity", "age", "features", "appearance", "comparison", "fit", "policy_history"}
REFERENCES = {"current", "previous", "first", "second", "other", "explicit", "none", "both"}

ROUTER_PROMPT = """任务是理解用户意图并分类，不回答问题，不提供任何商品事实。只输出包含以下四个键的JSON：
{"intent":"chat","topic":"unknown","reference":"none","need_kind":"none"}
intent枚举：chat普通交流/身份/能力/问候/致谢；product商品事实查询或条件性分析；select选商品；catalog查看商品；requirements用户补充或修改个人需求；live实时业务；policy现行政策；clarify意图不明；unsupported场外。
topic枚举：unknown,width,height,length,weight,color,material,brand,model,assembly,cleaning,capacity,age,features,appearance,comparison,fit,policy_history。
capacity仅指承重；电池/电量/功率等未知主题用unknown。不要因为不知道答案就分为chat，商品事实问题仍为product。
reference枚举：none,current,previous,first,second,other,explicit,both。previous是上一款；first/second是有序列表或比较中的第一个/第二个、后一个；other表示换一款未指定。explicit仅本轮明确编号。
need_kind枚举：none,purpose,space,focus。只有用户表达自己需求时才有kind：用途purpose、可用空间限制space、关注点focus。商品参数断言不是用户需求，不能覆盖资料。
“想给孩子用”“放在书房”是requirements/purpose；“我只关心怎么擦”是requirements/focus；“可用宽度只有50厘米”是requirements/space。
“怎么擦”“需要安装吗”“多重”“按我的关注点说说”“适合多大孩子”是product，不是chat。用户要求认可自己声称的规格仍为product。
身份/能力/问候才chat，topic必须unknown；问候夹带商品问题优先product。聊天不要求先选商品。
按上下文理解代词和澄清回复；用户提供商品编号以补充上轮问题时为select/explicit；“换一款”是select/other；“刚才那款呢”是product/previous。
上下文中的needs是用户要求，不是商品事实；不读取历史助手回答来判断参数。
四个键均必须存在，不输出message或needs，不允许输出任何解释。
"""

CONVERSATION_PROMPT = """你是自然交流的商品资料助手，底座为本地Qwen3.5-2B。用简短、友好的中文直接回应，不说内部路由器或技术实现。
真实能力：现有24个椅子商品的ABO历史属性、主图与产品要点派生资料，可以通过检索查参数、原文功能、年龄、承重、组装和清洁说明，并提供引用。
资料不是厂家原始长手册。尚未接入实时价格、库存、订单、交易、现行店铺政策、长期记忆或网页搜索，不是自主Agent。
这次只处理普通交流/身份/能力介绍，不回答具体商品参数，不引用历史答案、不编造已完成业务。
用户可通过“商品列表”查看现有商品编号并选择，再查询对应资料。身份介绍、问候、致谢直接简短回应。
请用1至3句话，不输出JSON或思考过程。"""


def validate_facets(raw):
    obj=parse_object(raw)
    if set(obj)!={'task','need'} or obj['task'] not in {'meta','fact','select','live','execute','policy','none'} or obj['need'] not in {'none','focus','purpose','space'}:
        raise ValueError('Invalid turn facets; cannot assume semantic success')
    return obj


def is_short_thanks(text):
    # A finite courtesy control grammar, never a catch-all for open questions.
    compact=re.sub(r'[\s!！。.,，?？]+','',text).casefold()
    return bool(re.fullmatch(r'(?:多谢|谢{1,2}|感谢)(?:你|您|啦|了|呀|啊|哦|哟|喽)*|thanks|thankyou',compact))


def parse_object(raw):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)[:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict): raise ValueError("Expected one JSON object")
    return obj


def validate_route(raw, question):
    obj = parse_object(raw)
    if set(obj) != {"intent", "topic", "reference", "need_kind"}:
        raise ValueError("Router schema mismatch")
    if obj["intent"] not in INTENTS or obj["reference"] not in REFERENCES or not isinstance(obj['topic'],str):
        raise ValueError("Unknown router enum")
    unrecognized_topic=None
    if obj['topic'] not in TOPICS:
        unrecognized_topic=obj['topic'];obj['topic']='unknown'
    if obj['need_kind'] not in {'none','purpose','space','focus'}:raise ValueError('Unknown requirement kind')
    if obj['intent']=='chat' and obj['topic']!='unknown':raise ValueError('Product topic cannot use ungrounded chat')
    needs={'purpose':'','space':'','focus':''}
    if obj['need_kind']!='none':needs[obj['need_kind']]=question[:180]
    result={**obj,'message':'','needs':needs}
    if unrecognized_topic:result['topic_fallback']={'unrecognized':unrecognized_topic,'effective':'unknown','policy':'No factual guess; use evidence or abstain.'}
    return result


def normalize_quote(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r'[\u2010-\u2015]','-',text).replace('⁄','/')
    # Retain decimal points, signs, ranges, fractions and percentages: 3.5 != 35, 1/2 != 12.
    return re.sub(r"[^a-z0-9\u4e00-\u9fff./%+\-]", "", text)


def validate_claims(raw, pages, topic):
    obj = parse_object(raw)
    if set(obj) != {"action", "claims"} or obj["action"] not in {"answer", "not_found"}:
        raise ValueError("Evidence schema mismatch")
    claims = obj["claims"]
    if not isinstance(claims, list) or len(claims) > 4:
        raise ValueError("Invalid claim count")
    if obj["action"] == "not_found":
        if claims: raise ValueError("Abstention must not contain facts")
        return []
    if not claims: raise ValueError("Answer has no grounded claims")
    result = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"sku", "label", "quote"}:
            raise ValueError("Invalid evidence claim")
        if not all(isinstance(v, str) for v in claim.values()): raise ValueError("Invalid claim types")
        label = claim["label"]
        if not re.fullmatch(r"P[1-9][0-9]*", label): raise ValueError("Invalid citation label")
        i = int(label[1:]) - 1
        if i >= len(pages): raise ValueError("Citation not provided this turn")
        page = pages[i]
        if claim["sku"] != page["sku"]: raise ValueError("Claim crosses product boundary")
        quote = normalize_quote(claim["quote"])
        source_text=normalize_quote(page["text"])
        if not 6 <= len(quote) <= 550 or quote not in source_text:
            raise ValueError("Quote is not present on the cited retrieved page")
        position=source_text.find(quote)
        preceding=source_text[max(0,position-12):position]
        if re.search(r'(?:no|not|without)$',preceding) and not re.match(r'no|not|without',quote):
            raise ValueError('Quote omits immediately preceding negation')
        markers = {"width":("width","宽度"),"height":("height","高度"),"length":("length","长度"),"weight":("weight","重量")}
        if topic in markers and not any(x in claim["quote"].lower() for x in markers[topic]):
            raise ValueError("Numeric quote must include its property label")
        relevance={'assembly':('assembl','组装','安装'),'cleaning':('wipe','clean','cloth','清洁','擦'),
                   'age':('age','year','kids','youth','children','岁'),'capacity':('capacity','承重'),
                   'brand':('brand','品牌'),'material':('material','材质'),'color':('color','颜色'),'model':('model','型号')}
        if topic in relevance and not any(x in claim['quote'].lower() for x in relevance[topic]):
            raise ValueError('Quote does not address the requested property')
        result.append({**claim, "page_id":page["page_id"], "page":page["page"]})
    return result


class LocalDialogueModel:
    def __init__(self):
        self.reader = None
        self.load_count = 0

    def generate(self, messages, stage, trace, max_tokens=256):
        import torch
        started = time.perf_counter()
        load_seconds = 0.0
        if self.reader is None:
            self.reader = VLMReader(WORKSPACE / "models/Qwen3.5-2B", max_new_tokens=320)
            self.load_count += 1
            load_seconds = self.reader.load_seconds
        entry = {"stage":stage,"raw_output":"","status":"started","load_seconds":load_seconds,"model_executed":False}
        trace.append(entry)
        try:
            inputs = self.reader.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                         enable_thinking=False, return_dict=True, return_tensors="pt").to("cuda")
            if inputs["input_ids"].shape[1] > 7000: raise ValueError("Input exceeds dialogue token budget")
            entry["prepare_seconds"] = time.perf_counter() - started - load_seconds
            grid = inputs.get("image_grid_thw")
            entry["image_grid_thw"] = grid.tolist() if grid is not None else []
            merge = getattr(self.reader.processor.image_processor,"merge_size",2)
            entry["visual_tokens"] = sum(int(t*h*w)//(merge*merge) for t,h,w in entry["image_grid_thw"])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); gen_start=time.perf_counter()
            entry['model_executed']=True
            with torch.inference_mode():
                output = self.reader.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,max_time=35.0)
            torch.cuda.synchronize()
            entry.update(generation_seconds=time.perf_counter()-gen_start,input_tokens=int(inputs["input_ids"].shape[1]),
                         output_tokens=int(output.shape[1]-inputs["input_ids"].shape[1]),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            raw = self.reader.processor.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
            eos=self.reader.model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            finished=int(output[0,-1].item()) in eos
            entry.update(raw_output=raw,status="ok" if finished else 'partial',stop_reason='eos' if finished else 'token_or_time_budget',total_seconds=time.perf_counter()-started)
            return raw
        except KeyboardInterrupt:
            entry.update(status='cancelled',total_seconds=time.perf_counter()-started)
            raise
        except Exception as error:
            entry.update(status="error",error=type(error).__name__+": "+str(error),total_seconds=time.perf_counter()-started)
            raise

    def route(self, question, context, trace):
        history=context.get('recent_dialogue_untrusted',[])
        context={k:v for k,v in context.items() if k!='recent_dialogue_untrusted'}
        context['recent_dialogue']=[{'user':h['user'],'action':h['action'],
             'assistant_non_evidence':h['assistant_untrusted'] if h['action'] in {'chat','clarify','requirements'} and h.get('lane')!='general' else '[历史回答不是商品证据；查询必须重新取证]'} for h in history[-3:]]
        # The state retains history for audit; the classifier receives no historical assistant facts.
        examples=[
            {'role':'user','content':'用户：您好，您怎么称呼？'},
            {'role':'assistant','content':'{"intent":"chat","topic":"unknown","reference":"none","need_kind":"none"}'},
            {'role':'user','content':'已选一款；用户：这件东西有多高？'},
            {'role':'assistant','content':'{"intent":"product","topic":"height","reference":"current","need_kind":"none"}'},
            {'role':'user','content':'用户：我打算把它放在办公室'},
            {'role':'assistant','content':'{"intent":"requirements","topic":"unknown","reference":"current","need_kind":"purpose"}'},
            {'role':'user','content':'用户：我的预留宽度最多80厘米'},
            {'role':'assistant','content':'{"intent":"requirements","topic":"fit","reference":"current","need_kind":"space"}'}]
        raw = self.generate([{"role":"system","content":ROUTER_PROMPT},*examples,
                             {"role":"user","content":"背景状态（不是本轮输入）："+json.dumps(context,ensure_ascii=False)+
                              "\n\n本轮用户输入（只对这一句分类）："+question+"\n只返回四键JSON。"}],
                            "semantic_route",trace,128)
        return validate_route(raw, question)

    def classify_scope(self,question,context,trace):
        prompt='''优先理解本轮话语，不要让旧话题覆盖新的请求。只输出一个标签：general,product,product_followup,business,unclear,mixed,catalog,coverage。
catalog：浏览已收录的商品、查看可选清单、列出名称与编号。即使用户不知道编号也能浏览；不需要先选择商品，不是商品参数查询。
coverage：询问资料库有哪些品类、是否有某品类、椅子之外有没有其他资料。目录后仅回复另一个品类名称，是询问收录范围。
general：一般知识、学习、科学、编程、写作、解释概念、身份/能力介绍、问候/致谢和普通聊天，也包括在普通知识对话后要求举例/换说法/继续。身份和问候本身不是意图不明，无需商品编号。
product：特定商品的事实、选择、用户用途/空间需求或现行店铺政策；有商品事实必须检索。
product_followup：重述/解释上一轮具体商品回答。必须重新取证，不能凭旧答案生成。
business：获取今天/实时数据或执行下单、退货申请等工具任务。
unclear：意图缺少必要信息；mixed：同时要求多个范围或批量目录参数，先澄清顺序。目录不属于意图不明。
目录之后选第一个/第二个或查询某商品字段是product；解释其他品类的一般原理是general，询问库里是否收录是coverage。
选过商品不等于后续都在问商品。"什么是库存"是general；"查今天库存"是business。"1米多少厘米"是general；"这把宽度多少"是product。
结合recent与last_lane理解省略：上一轮讲普通知识后“没懂，举例”是general；上一轮讲商品参数后重述是product_followup。
不回答问题，不输出JSON或解释。'''
        examples=[('什么是向量数据库','general'),('为什么海水看起来是蓝色','general'),('帮我润色一句话','general'),
                  ('这款椅子怎么打理','product'),('我打算给孩子用','product'),('今天还剩多少库存','business'),
                  ('上轮在讲RAG，现在要求换个例子','general'),('上轮在查商品规格，现在要求解释刚才参数','product_followup'),
                  ('展示本地资料库的产品编号和原名','catalog'),('有哪些已经收录的东西可选','catalog'),
                  ('有没有椅子以外的品类资料','coverage'),('上轮介绍资料覆盖范围，本轮只问鞋子','coverage')]
        messages=[{'role':'system','content':prompt}]
        for q,label in examples:messages.extend([{'role':'user','content':q},{'role':'assistant','content':label}])
        messages.append({'role':'user','content':'背景：'+json.dumps(context,ensure_ascii=False)+'\n本轮话语：'+question})
        raw=self.generate(messages,'scope_route',trace,12).strip().lower().strip('"\'` ')
        if raw not in DOMAINS:raise ValueError('Unrecognized conversation scope; no silent general fallback')
        return raw

    def pending_relation(self,question,pending,trace):
        prompt='''判断本轮输入与尚未完成的问题之间的关系，只输出resume、replace或aside。
resume：补充原问题所缺的信息，或明确继续原问题；replace：改问新问题、取消原查询或切换话题；aside：暂时问候、致谢或确认助手在不在，之后还可能继续原问题。
以本轮表达为准，不因存在旧问题就认定用户一直在补充。一般知识的新提问是replace，不是索要商品编号。'''
        raw=self.generate([{'role':'system','content':prompt},{'role':'user','content':json.dumps({'unfinished_question':pending,'current_input':question},ensure_ascii=False)}],
                          'pending_relation',trace,8).strip().lower().strip('"\'` ')
        if raw not in {'resume','replace','aside'}:raise ValueError('Invalid pending relation; do not silently resume')
        return raw

    def open_converse(self,question,history,trace,clarify_only=False):
        prompt='''你是使用本地Qwen3.5-2B的中文智能助手。可以自然回答一般知识、科学、学习、编程、写作和普通聊天，结合提供的近期普通对话理解“没听懂、换个例子、继续”等追问。
直接回答用户原问题，默认只写80至120个汉字、一个短段落，先给结论再给一个小例子；不自发扩展背景、分章节或追加邀请。用户明确要求长文或代码时才展开。无需商品编号，不要把所有话题引回购物。
近期普通回答仅是会话上下文，不是已核实的商品证据。没有历史时不要虚构刚才说过什么，应请用户重述主题。
一般知识来自模型已有知识，可能出错；不知道就说明不确定并澄清。RAG也不能保证答案正确或消除所有幻觉。没有联网、搜索、天气/实时价格/库存/订单/退款工具，不能声称查到实时情况或执行过操作。
具体商品参数、清洁适配、保修等必须检索本轮证据，这个分支没有这些证据。遇到具体商品事实请求应说明需要查资料，不能用常识、用户断言或历史回复编参数，也不能虚构页面引用。
可以解释一般概念、单位换算或明确的假设例子，不能将假设套在真实商品上。
身份：商品资料与一般知识对话助手。可查现有24个椅子商品的ABO派生资料；不是厂家长手册，不提供现行店铺政策或长期记忆。不要声称自己是ChatGPT或已经联网。没有可靠信息证明训练知识的截止日期，不要编造日期。
不要输出内部分类标签、JSON思维过程或分析草稿。'''
        if clarify_only:prompt+='\n本轮范围尚不确定，请针对原话自然澄清主题或目标；不要默认要求商品编号。'
        prompt+='\n实际应用能力清单（仅为范围说明，不是商品参数证据）：'+json.dumps(getattr(self,'catalog_manifest',{}),ensure_ascii=False)
        prompt+='\n只有已收录的商品能检索。不能邀请用户提供其他品类的名称或编号来查询未收录资料；可以聊一般知识。目录需求由应用读取清单，不要谎称没有目录。'
        messages=[{'role':'system','content':prompt},*general_messages(history),{'role':'user','content':question}]
        reply=self.generate(messages,'general_reply',trace,384)
        if trace:trace[-1]['reply_input_scope']='current_question_and_general_history_only; no product answers'
        return validate_general_reply(reply)

    def converse(self,question,trace):
        # Separate permission classification from reply generation. A misrouted product question
        # must never be passed to a free-form text answerer without evidence.
        gate='只分类，不回答问题。只输出identity、capabilities、greeting、thanks、other之一。身份询问identity，询问助手能力capabilities，问候greeting，感谢thanks。具体商品咨询、维护建议或参数一律other，即使没有编号。'
        examples=[{'role':'user','content':'你有什么作用呀？'},{'role':'assistant','content':'capabilities'},
                  {'role':'user','content':'我应该怎么称呼您？'},{'role':'assistant','content':'identity'},
                  {'role':'user','content':'这件东西好打理吗？'},{'role':'assistant','content':'other'},
                  {'role':'user','content':'早上好呀'},{'role':'assistant','content':'greeting'},
                  {'role':'user','content':'辛苦啦，谢谢帮助'},{'role':'assistant','content':'thanks'}]
        kind='thanks' if is_short_thanks(question) else self.generate([{'role':'system','content':gate},*examples,{'role':'user','content':'当前需要分类的话语：'+question}],
                           'conversation_permission',trace,8).strip().lower().strip('"\'` ')
        canonical={'identity':'简短自然地介绍你自己的身份。','capabilities':'介绍你能提供的帮助和重要能力边界。',
                   'greeting':'简短自然地回应问候。','thanks':'简短自然地回应感谢。'}
        if kind=='other':return None
        if kind not in canonical:raise ValueError('Unknown ordinary-conversation permission')
        # ONLY this approved metadata purpose reaches reply generation, never the original
        # user question, previous product answers, specifications, or user assertions.
        reply=self.generate([{'role':'system','content':CONVERSATION_PROMPT},{'role':'user','content':canonical[kind]}],
                            'conversation_reply',trace,128)
        if trace:trace[-1].update(conversation_kind=kind,reply_input_scope='capability_manifest_and_canonical_meta_purpose_only')
        if not reply or reply.startswith('{') or re.search(r'已(?:经)?(?:查询|下单|退款|付款|查到库存)|我是ChatGPT',reply,re.I):
            raise ValueError('Conversation exceeded capability or format boundary')
        if re.search(r'(?:宽度|重量|承重|适用年龄).{0,12}\d|\d+(?:\.\d+)?\s*(?:英寸|磅|inches|pounds)|建议.{0,10}(?:湿布|干布|清洁剂)',reply,re.I):
            raise ValueError('Product facts/advice cannot bypass image evidence via ordinary chat')
        return reply

    def focus_topic(self,value,trace):
        prompt='只识别本句中明确说出的、此刻肯定关注的商品主题，不回答商品事实。忽略已否定或不再关注的旧主题。只输出一个英文标签：'+','.join(sorted(TOPICS))+'. 没有明确主题、仅说按刚才关注点回答、多主题无法确定时输出unknown，不能猜cleaning等主题。'
        examples=[{'role':'user','content':'原先看外观，现在改看装配要求'},{'role':'assistant','content':'assembly'},
                  {'role':'user','content':'不想比重量了，想问如何擦干净'},{'role':'assistant','content':'cleaning'},
                  {'role':'user','content':'就按照前面说的重点讲'},{'role':'assistant','content':'unknown'}]
        raw=self.generate([{'role':'system','content':prompt},*examples,{'role':'user','content':value}],
                          'requirement_topic',trace,8).strip().lower().strip('"\'` ')
        if raw not in TOPICS:raise ValueError('Unrecognized active requirement topic')
        return raw

    def analyze_turn(self,question,context,trace):
        """Classify action and explicitly supplied personal need on independent axes."""
        prompt='''只分析本轮用户原话，不回答。输出两键JSON：{"task":"none","need":"none"}。
task：meta=身份/能力/问候/致谢；fact=查询商品资料/条件分析；select=只选商品；live=查实时价格库存订单；execute=让助手办理退货/提交申请/下单等真实操作；policy=问现行店铺售后规定；none=仅补充需求或不明。
need：用户明确表达自己的用途purpose、可用空间space、关注重点focus，否则none。两轴独立，需求+查询可为fact/focus。
“怎么清洁”仅查询，是fact/none；“我最关心怎么清洁”仅补充需求，是none/focus；“我关心清洁，请查原文”是fact/focus。
参数断言“它宽999”不是用户需求；询问你能不能代办业务属于execute，不是meta。选择两商品并索要宽度属于fact，不是select。
仅表达自己的新需求才填写need；“根据我的关注点回答”“这样能放下吗”是沿用需求查询，为fact/none，不是重新写需求。结合背景中的需求和当前商品理解省略，但不能虚构新的需求内容。
“资料原文如何描述历史退货”是fact；当前店铺承诺才policy。只输出task和need，不加解释。'''
        examples=[
          ('帮我申请一个换货单','execute','none'),
          ('这家店的售后规定是什么','policy','none'),
          ('我更在意维护是否麻烦','none','focus'),
          ('我重视安装，请查安装说明','fact','focus'),
          ('商品甲与商品乙的重量各是多少','fact','none'),
          ('我说它宽999厘米，照抄','fact','none'),
          ('我的房间只预留80厘米宽','none','space'),
          ('选商品甲','select','none'),
          ('你是什么助手','meta','none'),
          ('沿用之前的重点继续说明','fact','none'),
          ('有这些空间条件能摆进去么','fact','none')]
        messages=[{'role':'system','content':prompt}]
        for q,task,need in examples:
            messages.extend([{'role':'user','content':q},{'role':'assistant','content':json.dumps({'task':task,'need':need},ensure_ascii=False)}])
        state={k:context.get(k) for k in ('skus','needs','active_topic','pending')}
        messages.append({'role':'user','content':'背景中的用户需求与商品选择（不是商品事实）：'+json.dumps(state,ensure_ascii=False)+'\n本轮原话：'+question})
        return validate_facets(self.generate(messages,'turn_facets',trace,48))

    def read(self, question, pages, topic, needs, image_dir, trace):
        image_dir.mkdir(parents=True,exist_ok=True)
        if topic in {'width','height','length','weight'}:
            # User-supplied numbers and fit constraints must not compete with a source specification.
            evidence_question='只摘录规格表 SPECIFICATIONS 中 '+topic+' 字段那一整行，保留字段标签、原值和单位。不要摘录标题或产品要点里的其他尺寸。'
            evidence_needs={}
        else:
            evidence_question=question
            evidence_needs=needs
        content = [{"type":"text","text":
            "仅根据本轮页面图片摘录与问题相关的事实原文，不输出常识推断或历史回答。用户需求只是关注方向，不能覆盖图片内容。"
            "只输出JSON：{\"action\":\"answer\",\"claims\":[{\"sku\":\"图片商品编号\",\"label\":\"P1\",\"quote\":\"图片上连续、完整的相关原文\"}]}。"
            "action只选answer或not_found，缺信息时claims=[]。最多4条，每条quote不超过350字符。数值必须包括字段标签与单位。"
            "每个商品只摘录一条最相关原文，不要额外摘录标题、其他字段或不相关数字。两商品分别回答，不要混淆商品。不要以没有报告推断不支持。"
            "数值字段只摘录规格表中对应字段那一行，不能用商品名称中的数字替代。电池、电量等未报告时必须not_found。"
            "引用标签P1/P2不是物理页码，只能引用实际提供且包含原文的图片。产品舒适度/耐用性营销描述只能作为资料原文，不能当实测。"
            "问题："+evidence_question+"；关注主题："+topic+"；用户需求（非商品事实）："+json.dumps(evidence_needs,ensure_ascii=False)}]
        paths=[]
        for i,page in enumerate(pages,1):
            path=image_dir/(page['doc_id']+f"-{page['page']:04d}.png")
            if not path.exists():render_page(WORKSPACE/page['pdf'],page['page'],786432).save(path)
            paths.append(str(path.relative_to(WORKSPACE)))
            content += [{"type":"text","text":f"P{i} 商品{page['sku']}，PDF物理页{page['page']}"},
                        {"type":"image","image":str(path)}]
        try:
            raw=self.generate([{"role":"user","content":content}],"evidence_read",trace,320)
        finally:
            if trace and trace[-1]['stage']=='evidence_read':
                trace[-1].update(image_paths=paths,evidence_question=evidence_question,evidence_needs=evidence_needs)
        return validate_claims(raw,pages,topic),paths

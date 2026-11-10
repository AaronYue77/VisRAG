"""Conversation lanes: general history is context, never product evidence."""
import re

DOMAINS={'general','product','product_followup','business','unclear','mixed','catalog','coverage'}

def explicit_task_change(question):
    """Explicit stop/change control grammar, not a whitelist of answerable topics."""
    return bool(re.search(r'换.{0,3}话题|另一个问题|取消刚才',question) or re.search(
        r'^(?:那|嗯|好的|请|我(?:想)?|我们)?[，,\s]*(?:暂时|暂且|现在|先)?(?:不再|不用再|不用|不要再|不要|别再|别|不)(?:查|查询|问|讨论|继续)',question.strip()))

def invented_fact_request(question):
    return bool(re.search(r'(?:编造|瞎编|胡编|编(?:一个|个|出|些|点)|猜(?:测|一下|一个|个)?).*(?:参数|电池|这款|商品)',question))

def missing_history_reference(question,history,skus):
    """A missing-history check, not an answer or an open-topic whitelist."""
    return not history and not skus and bool(re.search(
        r'(?:刚才|之前|前面|上轮)(?:的)?(?:解释|回答|问题|内容)|我(?:刚才|之前|上轮).{0,3}(?:问|说)了?(?:什么|哪|啥)|(?:刚才|上轮)没(?:听)?懂',question))

def significant_lane(history):
    for h in reversed(history):
        if h.get('lane') in {'product','general','catalog'}:return h['lane']
        if 'lane' not in h and h['action'] in {'answer','requirements','selected','not_found','chat'}:
            return 'general' if h['action']=='chat' else 'product'
    return 'none'

def product_guard(question,skus,history,ids):
    """Finite factual-scope guard, not a list of allowed general questions."""
    if ids:return 'product'
    if re.search(r'(?:代我|替我).*(?:退货|退款|换货|下单|提交)|(?:帮我|请).{0,3}(?:申请|办理|提交|取消订单|下单)',question):return 'business'
    if re.search(r'(?:这(?:个|家)?|该|当前)(?:店铺|店|商店).*(?:退|售后|保修)',question):return 'product'
    if re.search(r'(?:查|看|问|帮|请).*(?:现在|今天|当前|实时).*(?:价|货|库存|订单|物流|天气)|(?:现在|今天|实时).*(?:有没有货|有货|库存|价格|售价|多少钱|天气)|(?:查|跟踪|取消).*(?:订单|物流)|(?:还有|有没有|是否有)货',question):return 'business'
    if skus and re.search(r'换.{0,2}(?:一款|另一款|一个商品)|刚才那款',question):return 'product'
    if re.search(r'(?:这把|这款|该|所选|当前的?).{0,10}(?:椅子|商品|产品)|这把椅',question):return 'product'
    if skus and re.search(r'(?:它|这把|这款|这件|这玩意|所选).*(?:宽|重|高|长|尺寸|材料|材质|颜色|电池|组装|安装|清洁|擦|承重|保修)',question):return 'product'
    if skus and re.search(r'(?:买回去|买回来).*(?:装|擦|用|清)',question):return 'product'
    if skus and re.search(r'(?:我|家中|家里|可用|预留).*(?:宽|空间|厘米|英寸)',question) and re.search(r'\d',question):return 'product'
    if significant_lane(history)=='product' and (invented_fact_request(question) or re.search(r'忽略.*(?:资料|页面|证据)|没有证据.*常识',question)):return 'product_followup'
    # Open definition forms are not a finite topic whitelist. Strong specific-product
    # guards above still take priority over attempts to phrase a fact request as a concept.
    if re.search(r'什么是|什么叫|.{1,30}(?:是什么意思|是什么|是啥)',question):return 'general'
    if re.search(r'商品(?:参数|规格|编号)',question):return 'product'
    if not re.search(r'换个话题|另一个问题|另外想问',question) and re.search(r'没(?:听)?懂|不明白|再(?:讲|说|写|举|解释)|换.{0,8}(?:例子|说法|表达)|更.{0,8}(?:口语|通俗|简单)|刚才.{0,8}(?:参数|规格|答案|解释)',question):
        lane=significant_lane(history)
        if lane in {'general','product'}:return 'general' if lane=='general' else 'product_followup'
    return None

def scope_context(history,skus,needs,pending=None,manifest=None):
    recent=[]
    for h in history[-3:]:
        lane=h.get('lane','general' if h['action']=='chat' else 'product')
        recent.append({'lane':lane,'user':h['user'][:400],
                       'assistant':h['assistant_untrusted'][:700] if lane=='general' else '[商品回答不进入普通知识上下文]'} )
    return {'selected_products':skus,'user_needs':needs,'pending':pending,'catalogue':manifest,'last_lane':significant_lane(history),'recent':recent}

def general_messages(history):
    result=[]
    for h in history:
        if h.get('lane')=='general':
            result.extend([{'role':'user','content':h['user']},{'role':'assistant','content':h['assistant_untrusted']}])
    return result

def validate_general_reply(reply):
    if not isinstance(reply,str) or not reply.strip():raise ValueError('Empty open conversation reply')
    if re.search(r'(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])',reply,re.I):raise ValueError('Specific product identifier cannot enter an ungrounded reply')
    if re.search(r'【第\d+页】|abo2?-[A-Z0-9]+:p\d+',reply):raise ValueError('General reply cannot fabricate document citations')
    if re.search(r'(?:我已经|我已|已为你|已替你|帮你成功).{0,15}(?:下单|退款|提交|查询到实时)',reply):raise ValueError('General reply cannot claim tool execution')
    if re.search(r'(?:这款|这把|该商品|所选商品).{0,20}(?:宽度|重量|材质|承重|电池|清洁|组装|舒适|耐用).{0,20}(?:是|为|需要|支持|用|很)',reply):raise ValueError('Specific product fact cannot bypass RAG')
    if not re.search(r'假设|例如|举例|示例',reply) and re.search(r'(?:椅子|商品|产品).{0,8}(?:宽度|重量|承重|高度).{0,6}\d',reply):
        raise ValueError('Uncited product measurement cannot enter general conversation')
    return reply.strip()

def quoted_glosses(claims):
    """Explain only words actually present in current verified quotations."""
    text='\n'.join(c['quote'] for c in claims)
    terms=[(r'\bdamp cloth\b','damp cloth 指湿布'),(r'\bdry cloth\b','dry cloth 指干布'),
           (r'\bno assembly\b','No assembly 表示无需组装'),(r'\binches\b','inches 是英寸'),(r'\bpounds\b','pounds 是磅')]
    found=[meaning for pattern,meaning in terms if re.search(pattern,text,re.I)]
    return '\n原文词义说明：'+'；'.join(found)+'。这里只解释本轮引文用语，没有补充未提供的商品信息。' if found else ''

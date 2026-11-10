"""Resolve conversational objects from explicit selected entities, not old answers."""
import re
from full_catalogue import CATEGORIES


def ordinal_choice(text,last_task):
    """An ordered reference belongs to the latest shopping task, not stale lists."""
    if not last_task or last_task.get('lane')!='product':return None
    if re.search(r'函数|代码|程序|算法|故事|诗|句子|例子|人|朋友',text):return None
    found=list(re.finditer(r'第([0-9]{1,2}|[一二三四五六七八九十两])(?:个|款|件)?',text))
    if not found:
        bare=re.fullmatch(r'\s*([1-9一二三四五六七八九十两])\s*',text)
        found=[bare] if bare else []
    if not found:return None
    if len(found)>1:return {'ambiguous':True}
    value=found[0][1]
    index=int(value)-1 if value.isdigit() else ('一二三四五六七八九十'.index(value.replace('两','二')))
    remainder=text[:found[0].start()]+text[found[0].end():]
    remainder=re.sub(r'选中|选择|就|选|这个|商品|产品|吧|呢|[\s？?！!。]','',remainder)
    return {'index':index,'task':'select' if not remainder else ('features' if feature_request(remainder) else ('overview' if view_request(remainder) else 'field'))}


def bare_category_choice(text,last_task,manifest):
    if not last_task or last_task.get('lane') not in {'catalog','product'}:return False
    value=re.sub(r'[？?！!。\s]+$','',text.strip())
    value=re.sub(r'^(?:选中|选择|就要|就|选|看看|查看)\s*','',value)
    value=re.sub(r'(?:类商品|类产品|类)$','',value.rstrip('呢吧 ')).strip()
    if value.upper() in manifest['product_types']:return True
    return any(re.fullmatch(pattern,value,re.I) and any(k in manifest['product_types'] for k in kinds) for pattern,en,kinds in CATEGORIES)


def feature_request(text):
    return bool(re.search(r'功能|用途|使用特点|特性|特色|功能属性|哪些属性|什么属性',text))


def view_request(text):
    if re.search(r'材质|材料|品牌|颜色|重量|多重|宽度|高度|长度|型号|尺寸|多少钱|库存|订单|退款',text):return False
    return bool(re.search(r'查看|看看|看下|看一下|了解|介绍|展示|显示|\bview\b|\bshow\b',text,re.I))


def selected_name_reference(text,entities):
    explicit=re.search(r'这个商品|这款商品|该商品|所选(?:的)?商品|这款|这件|这双|这个包|这包',text)
    if explicit:return True
    for entity in entities:
        kind=entity.get('product_type','')
        if re.search(r'(?:这个|这款|该|刚才(?:的)?|选中(?:的)?)\s*'+re.escape(kind),text,re.I):return True
        for pattern,en,kinds in CATEGORIES:
            if kind in kinds and re.search(r'(?:这个|这款|该|刚才(?:的)?|选中(?:的)?)\s*(?:'+pattern+')',text,re.I):return True
    return False


def named_reference_types(text):
    result=set()
    for pattern,en,kinds in CATEGORIES:
        if re.search(r'(?:这个|这款|该|刚才(?:的)?|选中(?:的)?)\s*(?:'+pattern+')',text,re.I):result.update(kinds)
    return result


def product_task(text,entities,last_task,has_explicit_id=False):
    """Return a grounded task only with a positive object reference or action."""
    # These are language roles, not allowed ordinary topics. Conceptual/programming
    # requests remain open even if a product was selected earlier.
    if re.search(r'这个函数|这段代码|这个程序|算法|动态规划|正则|字符串|什么是|心理学|营销方法|定义|原理',text):return None
    named_types=named_reference_types(text)
    if not has_explicit_id and entities and named_types and not any(e.get('product_type') in named_types for e in entities):
        return {'intent':'clarify','reason':'selected_object_type_conflict',
                'clarification':'当前选中的商品类别与这次提到的对象不一致。请先搜索或提供你要查询的那款商品编号。'}
    named=selected_name_reference(text,entities)
    pronoun=bool(re.search(r'^(?:那|请问|我说的?|我是说|那么)?\s*[他它](?:的|还|有|能|可|是|支持)',text))
    implicit=bool(re.fullmatch(r'(?:功能|功能属性|属性|用途|使用特点)(?:是什么|有哪些|有什么|呢|怎么样)?[？?！!。\s]*',text))
    recent_product=bool(last_task and last_task.get('lane')=='product')
    linked=has_explicit_id or named or ((pronoun or implicit) and recent_product)
    if not linked:return None
    if (pronoun or implicit) and not named and len(entities)>1:
        return {'intent':'clarify','reason':'ambiguous_selected_entity','clarification':'你指的是哪一款商品？请提供编号，或明确第一个/第二个。'}
    if feature_request(text):
        return {'intent':'fact','reason':'resolved_product_features','product_task':'features'}
    if view_request(text):
        return {'intent':'fact','reason':'resolved_product_view','product_task':'overview'}
    if re.search(r'品牌|材质|颜色|重量|多重|宽度|高度|长度|尺寸|型号|参数|规格|属性|配件|清洁|安装|组装|承重|续航',text):
        return {'intent':'fact','reason':'resolved_product_property','product_task':'field'}
    return None

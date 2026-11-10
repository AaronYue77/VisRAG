"""Catalog statistics, public search plans and source-extractive product overviews."""
import json
import re
import unicodedata
from full_cards import evidence_fields,display_values
from full_retrieval import EXPANSIONS

# Public language aliases map to actual source types; unrecognized queries still
# use semantic query planning. This table does not limit the searchable catalog.
CATEGORIES=[
 (r'运动鞋|跑鞋|sneakers?|running shoes','shoes sneakers',['SHOES','TECHNICAL_SPORT_SHOE']),
 (r'靴子|短靴|长靴|boots?','boot',['BOOT']),
 (r'凉鞋|sandals?','sandal',['SANDAL']),
 (r'鞋垫|insoles?|shoe inserts?','shoe insert',['SHOE_INSERT']),
 (r'鞋子|鞋类|鞋(?!架|柜|带|刷|油)|footwear|\bshoes\b','shoes',['SHOES','BOOT','SANDAL','TECHNICAL_SPORT_SHOE']),
 (r'手机壳|手机保护套|phone case','phone case',['CELLULAR_PHONE_CASE']),
 (r'耳机|headphones?','headphones',['HEADPHONES']),
 (r'双肩包|背包|backpacks?','backpack',['BACKPACK']),
 (r'椅子|chairs?','chair',['CHAIR','BEAN_BAG_CHAIR']),
 (r'沙发|sofas?','sofa',['SOFA']),
 (r'灯泡|light bulbs?','light bulb',['LIGHT_BULB']),
 (r'咖啡|coffee','coffee',['COFFEE']),
 (r'地毯|rugs?','rug',['RUG']),
 (r'手提包|handbags?','handbag',['HANDBAG']),
 (r'帽子|hats?','hat',['HAT']),
 (r'桌子|tables?','table',['TABLE']),
]
LABELS={'brand':'品牌','color':'颜色','material':'材质','model_number':'型号','product_type':'原始类别',
        'fabric_type':'面料','style':'款式','size':'尺寸标记','width':'宽度','height':'高度','length':'长度','weight':'重量'}


def catalogue_request(text):
    if re.search(r'这(?:个|款|件)商品|该商品|所选商品',text):return False
    if re.search(r'写|代码|原理|定义|什么是|如何设计',text):return False
    return bool(re.search(r'商品列表|类目列表|收录范围|资料覆盖范围',text) or
      (re.search(r'商品|产品|品类|类目|知识库|资料库|库里',text) and
       re.search(r'有哪些|有什么|哪些|什么种类|什么类型|都有些什么|都有什么|包含什么|收录什么',text)))


def category_search_request(text,manifest):
    if re.search(r'除了|以外|其他类目|其它类目|收录哪些类目|什么是|原理|为什么|怎么|如何|代码|函数|算法|字符串|解释|讲解',text):return False
    if not re.search(r'查|找|搜索|看看|看一下|浏览|有哪些|有没有|想买',text,re.I):return False
    return any(re.search(pattern,text,re.I) and any(k in manifest['product_types'] for k in kinds) for pattern,en,kinds in CATEGORIES)


def continuation_request(text):
    clean=re.sub(r'[\s，,。.!！?？]+','',text)
    return bool(re.fullmatch(r'(?:那|我说的|我是说|除此之外)?(?:还有(?:呢|吗|什么|哪些|别的|其他|其它|更多)?(?:相关信息|信息|资料|内容)?|继续|接着说|再说(?:点|一些)|再多介绍一点)',clean))


def overview_request(text):
    return bool(re.search(r'(?:相关|更多|其他|其它|详细|基本|全部|完整).{0,5}(?:信息|资料|介绍|内容)|(?:介绍|了解).{0,8}(?:这款|这个商品|该商品)|商品(?:的)?(?:信息|资料|详情)|(?:这款|这个商品).{0,8}(?:有什么|有哪些)',text))


def catalog_page(manifest,offset=0,budget=14):
    types=manifest['product_types']
    preferred=[('CELLULAR_PHONE_CASE','手机壳'),('SHOES','鞋'),('BOOT','靴子'),('SANDAL','凉鞋'),
      ('TECHNICAL_SPORT_SHOE','运动鞋'),('HEADPHONES','耳机'),('BACKPACK','背包'),('CHAIR','椅子'),
      ('SOFA','沙发'),('TABLE','桌子'),('BED','床'),('LIGHT_BULB','灯泡'),('COFFEE','咖啡'),('RUG','地毯')]
    used={k for k,l in preferred}
    ordered=[(k,label,types[k]) for k,label in preferred if k in types]
    ordered += [(k,k,n) for k,n in types.items() if k not in used]
    chunk=ordered[offset:offset+budget]
    if not chunk:return '这份目录的全部原始类目已展示完，可以指定类目查商品。',offset,[]
    lines=[f'本地ABO历史商品资料共有 {manifest["records"]:,} 条、{len(types)} 个原始类目。'+('例如：' if offset==0 else '接着上次目录，还有：')]
    lines += [f'- {label}（{kind}）：{n:,} 条' for kind,label,n in chunk]
    lines.append('可以直接说“我想查询鞋子类的商品”，或说“还有呢”继续看类目。这些是历史资料，不代表实时在售库存。')
    return '\n'.join(lines),offset+len(chunk),[{'product_type':k,'label':l,'records':n} for k,l,n in chunk]


def normalize_search(text,manifest,filters=None,*,separate_terms=True):
    filters=dict(filters or {});query=text.strip()
    query=re.sub(r'^(?:请)?(?:我(?:想|要|想要)|帮我|请帮我)?\s*(?:查询|查看|查找|搜索|找一找|找|看看|看一下|浏览|search|find)[:：\s]*','',query,flags=re.I)
    query=re.sub(r'^(?:选中|选择|就要|就|选)\s*','',query)
    query=re.sub(r'[呢吧？?！!。\s]+$','',query)
    query=re.sub(r'(?:类的?商品|类的?产品|类商品|类产品|商品|产品)[？?。\s]*$','',query).strip()
    inferred=[];matched=None
    if query.upper() in manifest['product_types']:
        inferred=[query.upper()];matched='exact_source_type';query=query.upper()
    for pattern,english,kinds in ([] if inferred else CATEGORIES):
        if re.search(pattern,query,re.I):
            inferred=[k for k in kinds if k in manifest['product_types']]
            replacement=f' {english} ' if separate_terms else english
            query=re.sub(pattern,replacement,query,flags=re.I);matched=pattern;break
    if inferred:
        if filters.get('product_type') and filters['product_type'] not in inferred:
            raise ValueError('Requested category conflicts with the explicit product_type filter')
        if not filters.get('product_type') and not filters.get('product_types'):filters['product_types']=inferred
    pattern='|'.join(re.escape(k) for k in sorted(EXPANSIONS,key=len,reverse=True))
    query=re.sub(pattern,lambda m:f' {EXPANSIONS[m[0]]} ' if separate_terms else EXPANSIONS[m[0]],query)
    query=re.sub(r'类的?|的商品|的产品',' ',query)
    query=' '.join(query.split())
    return {'original_query':text,'query':query,'filters':filters,'category_alias':matched,
      'inferred_product_types':inferred,'needs_semantic_translation':bool(re.search(r'[\u4e00-\u9fff]',query))}


def normalized(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).casefold()


def overview_items(product,doc):
    raw=json.loads(product['raw_json']);items=[]
    for i,(key,value) in enumerate(evidence_fields(raw)):
        items.append({'item_id':f'field:{key}:{i}','label':LABELS.get(key,key),'value':value,'source_field':key})
    for i,value in enumerate(display_values(raw,'bullet_point')):
        items.append({'item_id':f'bullet:{i}','label':'产品要点（源文）','value':value.strip(),'source_field':'bullet_point'})
    for key,label in [('marketplace','登记平台'),('country','站点国家/地区代码')]:
        if raw.get(key):items.append({'item_id':'source:'+key,'label':label,'value':str(raw[key]),'source_field':key})
    for item in items:
        found=[] if item['item_id'].startswith('source:') else [p['page_id'] for p in doc['pages'] if normalized(item['value']) in normalized(p['text'])]
        item.update(citation=found[0] if found else product['listing_key']+':metadata',
          listing_key=product['listing_key'],raw_sha256=product['raw_sha256'],
          source_member=product['source_member'],source_line=product['source_line'])
    return items


def overview_chunk(product,doc,shown,budget=6,kind='overview'):
    items=overview_items(product,doc);remaining=[i for i in items if i['item_id'] not in shown]
    # First provide fields; a continuation reveals source notes or remaining fields.
    first=not shown;fields=[i for i in remaining if i['item_id'].startswith('field:')]
    features=[i for i in remaining if i['source_field']=='bullet_point']
    chunk=(features[:budget] if kind=='features' and features else (fields[:budget] if first and fields else remaining[:budget]))
    title=('这款商品的基本资料：' if first else '这款商品还记录了以下资料：')
    if kind=='features':title='这款商品在源资料中记载的功能与使用要点：' if features else '资料没有单独的功能说明，可以确认以下已记录信息：'
    lines=[title,product['listing_key'],product['title']]
    lines += [f'- {i["label"]}：{i["value"]}' for i in chunk]
    if not chunk:lines=['这款商品的已收录字段和产品要点已经展示完，没有更多未展示的资料。',product['listing_key']]
    elif len(remaining)>len(chunk):lines.append('还有未展示的源资料，可以继续问“还有呢”。')
    else:lines.append('以上是当前已收录的资料；未报告的参数不能据此补充。')
    return '\n'.join(lines),chunk

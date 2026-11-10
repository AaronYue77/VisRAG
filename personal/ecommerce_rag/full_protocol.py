"""Stratified new multi-category evaluation, separate from old frozen A/B."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import time
from full_catalog import ROOT,connect,field
from full_cards import ensure_card,photo,evidence_fields,display_values
from common import WORKSPACE,write_json,write_jsonl,file_hash,operation

EVAL=WORKSPACE/'data/ecommerce_rag/abo-full-evaluation-v2'
TYPES=['CELLULAR_PHONE_CASE','SHOES','GROCERY','HOME_BED_AND_BATH','CHAIR','SOFA','BOOT','SANDAL',
 'PET_SUPPLIES','SPORTING_GOODS','TABLE','RUG','HANDBAG','LIGHT_BULB','KITCHEN','HAT','BACKPACK','HEADPHONES','BED','COFFEE']
SETTINGS={'version':'full-ab-v2','embedding':'intfloat/multilingual-e5-small','embedding_revision':'614241f622f53c4eeff9890bdc4f31cfecc418b3','modes':['bm25','dense','hybrid','rerank'],'retrieval_top_k':5,'rrf_c':60,
 'rerank_candidates':20,'image_budget':2,'max_pixels_per_page':786432,'reader':'unchanged stage_a.Reader',
 'reader_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc','reader_max_new_tokens':160,
 'scope':'all147702 metadata rows globally searchable; sampled QA is evaluation, not corpus sampling',
 'selection':'B default selected by DEV family hit@5, then MRR and lower p50; test never selects params',
 'families':'connected components of shared main_image_id or brand/model; all variants stay one group',
 'qa':'new templates, labels from original source fields, historical fields not current policy',
 'visual':'candidate-conditional on-demand VisRAG optional; not a full147702-image visual index'}


def compact(text):return ''.join(c for c in text.casefold() if c.isalnum())


def select():
    if (EVAL/'selected.json').exists():return json.loads((EVAL/'selected.json').read_text('utf-8'))
    EVAL.mkdir(parents=True,exist_ok=True);db=connect()
    old=json.loads((ROOT.parent/'abo-cards-v1/selected-products.json').read_text('utf-8'))
    excluded={r['item']['item_id'] for r in old};used=set();selected=[]
    for kind in TYPES:
        candidates=[dict(r) for r in db.execute('''SELECT p.* FROM products p JOIN images i ON p.main_image_id=i.image_id
                  WHERE p.product_type=? AND p.domain='amazon.com' ''',(kind,))]
        # Some categories mostly live outside amazon.com; preserve their original domain.
        if len(candidates)<20:candidates=[dict(r) for r in db.execute('SELECT p.* FROM products p JOIN images i ON p.main_image_id=i.image_id WHERE p.product_type=?',(kind,))]
        candidates.sort(key=lambda r:hashlib.sha256(('full-ab-20261108:'+r['listing_key']).encode()).hexdigest())
        chosen=[]
        for r in candidates:
            item=json.loads(r['raw_json']);facts=evidence_fields(item)
            if r['sku'] in excluded or r['family_key'] in used or not display_values(item,'brand'):continue
            if len(compact(display_values(item,'brand')[0]))<2:continue
            if not any(k in {'color','material','weight','width','height','length'} for k,v in facts):continue
            if not any(str(v.get('language_tag','')).startswith('en') for v in item.get('item_name',[])):continue
            used.add(r['family_key']);chosen.append(r)
            if len(chosen)==4:break
        if len(chosen)!=4:raise ValueError('Not enough eligible independent families for '+kind)
        for i,r in enumerate(chosen):selected.append({'id':r['id'],'listing_key':r['listing_key'],'sku':r['sku'],
           'family_key':r['family_key'],'product_type':kind,'split':'dev' if i%2==0 else 'test'})
    assert len(selected)==80 and len({r['family_key'] for r in selected})==80
    write_json(EVAL/'selected.json',selected);db.close();return selected


def prepare():
    if (EVAL/'protocol.json').exists():print('Full evaluation already frozen',flush=True);return
    chosen=select();db=connect();products=[dict(db.execute('SELECT * FROM products WHERE id=?',(r['id'],)).fetchone()) for r in chosen]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assets=list(pool.map(photo,products))
    write_json(EVAL/'asset-preparation.json',[{'listing_key':r['listing_key'],**asset} for r,(_,asset) in zip(products,assets)])
    qs=[];retrieval=[];documents=[]
    for idx,(selected,r) in enumerate(zip(chosen,products)):
        item=json.loads(r['raw_json']);doc=ensure_card(r);documents.append({k:doc[k] for k in ('listing_key','pdf','sha256','version','raw_sha256')})
        fields=evidence_fields(item)
        second=next(((k,v) for wanted in ('width','weight','color','material','height','length') for k,v in fields if k==wanted),None)
        brand=next((k,v) for k,v in fields if k=='brand')
        for key,value in [brand,second]:
            words={'brand':'品牌 brand','color':'颜色 color','material':'材质 material','width':'宽度 width','weight':'重量 weight','height':'高度 height','length':'长度 length'}
            evidence=next(p for p in doc['pages'] if compact(key+': '+value) in compact(p['text']))
            qs.append({'question_id':'F-'+r['sku']+'-'+key,'question':r['sku']+' 的 '+words[key]+' 字段原文是什么？请保留数值和原始单位。',
             'request':{'skus':[r['sku']],'domain':r['domain']},'reference_answer':value,'reference_pages':[evidence['page_id']],
             'expected_action':'answer','expected_fact':value,'field':key,**{k:selected[k] for k in ('split','family_key','product_type')},
             'listing_key':r['listing_key'],'source_member':r['source_member'],'source_line':r['source_line']})
        if idx%4 in (0,1):
            # This explicitly requests an absent database-provenance field, not an invented value.
            qs.append({'question_id':'F-'+r['sku']+'-unknown','question':r['sku']+' 的官方商品记录是否报告了“2026年11月8日实时库存数量”？请给出该数量。',
             'request':{'skus':[r['sku']],'domain':r['domain']},'reference_answer':'实时库存需要业务工具','reference_pages':[],
             'expected_action':'requires_tool','expected_fact':None,'field':'route',**{k:selected[k] for k in ('split','family_key','product_type')},'listing_key':r['listing_key']})
        # Full title and shortened name both use a GLOBAL index, no SKU/type/domain filter.
        title=r['title'];clean=re.sub(re.escape(r['sku']),'',title,flags=re.I)
        tokens=re.findall(r'[\w]+',clean)
        short=' '.join(tokens[:min(9,len(tokens))])
        for variant,text in [('title',clean),('short_name',short)]:
            retrieval.append({'question_id':'FR-'+r['sku']+'-'+variant,'question':'Find product: '+text,
              'reference_listing_key':r['listing_key'],'reference_family':r['family_key'],'reference_answer':r['title'],
              'split':selected['split'],'product_type':r['product_type'],'suite':variant,'filters':{}})
        if (idx+1)%10==0:print('Full evaluation cards',idx+1,'/80',flush=True)
    # Explicit missing-field refusal questions, without realtime keywords.
    for split in ('dev','test'):
        rows=[(s,r) for s,r in zip(chosen,products) if s['split']==split]
        for s,r in rows[:10]:
            item=json.loads(r['raw_json'])
            absent=next((key for key in ('material','color','width','weight') if key not in {k for k,v in evidence_fields(item)}),None)
            if not absent:absent='manufacturing_timestamp_utc'
            qs.append({'question_id':'F-'+r['sku']+'-absent','question':r['sku']+' 的 '+absent+' 字段原始值是什么？没有报告就回答未报告。',
              'request':{'skus':[r['sku']],'domain':r['domain']},'reference_answer':'NOT_FOUND','reference_pages':[],
              'expected_action':'not_found','expected_fact':None,'field':'unknown',**{k:s[k] for k in ('split','family_key','product_type')},'listing_key':r['listing_key']})
    for split in ('dev','test'):
        for i,(question,action) in enumerate([('帮我办理退款','requires_tool'),('查询今天订单物流','requires_tool'),('当前店铺退货政策是什么','not_found'),('B000000000 的颜色是什么','not_found')]):
            qs.append({'question_id':'F-'+split+'-boundary-'+str(i),'question':question,'request':{},'reference_answer':action,'reference_pages':[],
             'expected_action':action,'expected_fact':None,'field':'route','split':split,'family_key':'boundary-'+split,'product_type':'boundary','listing_key':None})
    write_jsonl(EVAL/'questions.jsonl',qs);write_jsonl(EVAL/'retrieval-questions.jsonl',retrieval);write_json(EVAL/'documents.json',documents)
    dev={r['family_key'] for r in chosen if r['split']=='dev'};test={r['family_key'] for r in chosen if r['split']=='test'}
    assert not dev&test
    protocol={'settings':SETTINGS,'catalog_records':147702,'evaluated_products':len(chosen),'types':TYPES,
      'qa_questions':len(qs),'retrieval_questions':len(retrieval),'questions_sha256':file_hash(EVAL/'questions.jsonl'),
      'retrieval_questions_sha256':file_hash(EVAL/'retrieval-questions.jsonl'),'documents_sha256':file_hash(EVAL/'documents.json'),
      'catalog_sha256':json.loads((ROOT/'manifest.json').read_text('utf-8'))['database_sha256'],
      'annotation_note':'Source-derived templates and image-available English-title strata, not official/manual natural-language benchmark. No end-user traffic or deployment guarantee.',
      'split_note':'80 independent source families, two dev + two test per20 types; old24 products excluded. Full corpus stays searchable at inference, gold labels only used after prediction.',
      'assets_downloaded':sum(asset['status']=='downloaded' for _,asset in assets)}
    write_json(EVAL/'protocol.json',protocol);db.close()
    operation('全量检索的跨类目评测协议冻结','全库检索保持147702；按20类选80个不同原始关联家族抽评，排除旧24；真实主图按需下载、生成紧凑PDF；先冻结题/证据和配置。',
       f'80商品×20类，{len(retrieval)}道全库无SKU检索题，{len(qs)}道字段/缺失/边界题。',[EVAL/'protocol.json',EVAL/'questions.jsonl',EVAL/'retrieval-questions.jsonl'],
       '评测抽样不等于知识库抽样；dev/test家族不重叠；自动模板与来源完整字段偏好造成的局限保留。')
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


if __name__=='__main__':prepare()

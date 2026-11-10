"""Predeclare B retrieval experiments; never overwrite A data or predictions."""
from pathlib import Path
import json
import sys
import hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'rag_common'))
from common import WORKSPACE,read_jsonl,write_json,write_jsonl,file_hash,timestamp,operation

CORPUS=WORKSPACE/'data/ecommerce_rag/abo-dialogue-v2'
ROOT=WORKSPACE/'data/ecommerce_rag/b-evaluation-v1'
INDEX=WORKSPACE/'data/ecommerce_rag/b-index-v1'
MODES=['bm25','bge','text_rrf','visual','multimodal_rrf','rerank']
VISUAL_INDEX='b-abo73-pages-v1'
SETTINGS={'version':'b-v1','modes':MODES,'single_top_k':1,'comparison_top_k':2,'rrf_c':60,
 'chunk':'4 PDF text blocks, no overlap; dense chunks max512 tokens','candidate_pages':6,
 'max_pixels':786432,'reader':'stage_a.Reader','reader_prompt':'abo-grounded-json-v2-labels',
 'reader_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc','max_new_tokens':160,
 'bge_revision':'7999e1d3359715c523056ef9478215996d62a620',
 'visual_revision':'95ef596df871b606167cb7e4b7215caf1bfdf761',
 'reranker_revision':'2cfc18c9415c912f9d8155881c133215df768a70',
 'rewrite_policy':'On explicit Reader NOT_FOUND only, at most one deterministic lexical rewrite using public question. No gold, no repeated loop.',
 'retry_budget':1,'retry_mode':'selected dev mode; new pages only; total supplied images <=4',
 'test_policy':'Predeclared all-mode comparison, do not select parameters on test; integration choice uses dev only.',
 'runtime':'BGE/reranker CPU; original VisRAG model separate Python3.11 CUDA process; Qwen separate Python3.14 process.'}


def query_key(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def rewrite(question):
    import re
    terms=[(r'多宽|宽度|横向|宽窄','width 宽度 dimensions'),(r'多重|重量|搬动','weight 重量 pounds'),
           (r'材质|材料','material 材质'),(r'颜色|color','color 颜色'),(r'扶手|图片|照片','armrest chair image 扶手'),
           (r'清洁|擦|打理','wipe clean cloth cleaning 清洁'),(r'组装|安装','assembly assemble required 组装')]
    expansion=' '.join(v for p,v in terms if re.search(p,question,re.I))
    return (question+' '+expansion).strip()[:1200] if expansion else question


def build():
    if ROOT.exists():raise FileExistsError('B protocol already exists; preserve frozen version')
    ROOT.mkdir(parents=True)
    old=WORKSPACE/'data/ecommerce_rag/abo-cards-v1'
    questions=read_jsonl(old/'questions.jsonl');rows=[]
    for q in questions:
        row={**q,'question_id':'B-'+q['question_id'],'reference_pages':[p.replace('abo-','abo2-',1) for p in q['reference_pages']],
             'suite':'mapped_A_attributes','source_question_id':q['question_id']}
        rows.append(row)
    # Natural phrasing removes exact English field words, but keeps SKU as explicit request context.
    for q in questions:
        if q['field'] not in ('width','weight'):continue
        row={**q,'question_id':'B-natural-'+q['question_id'],'question':('这把椅子横向要占多大的地方？请按规格表给出宽度和原始单位。' if q['field']=='width' else '搬动它要考虑多重？请给出资料里的重量和原始单位。'),
             'reference_pages':[p.replace('abo-','abo2-',1) for p in q['reference_pages']],
             'suite':'natural_phrasing','source_question_id':q['question_id']}
        rows.append(row)
    pages=read_jsonl(CORPUS/'pages.jsonl');ids={p['page_id'] for p in pages}
    assert all(set(q['reference_pages'])<=ids for q in rows)
    write_jsonl(ROOT/'questions.jsonl',rows)
    tasks={}
    for q in rows:
        for text in [q['question'],rewrite(q['question'])]:tasks[query_key(text)]={'query_key':query_key(text),'text':text}
    write_jsonl(ROOT/'queries.public.jsonl',list(tasks.values()))
    write_json(ROOT/'protocol.json',{'declared_at':timestamp(),'settings':SETTINGS,'questions':len(rows),
      'splits':{s:sum(q['split']==s for q in rows) for s in ('dev','test')},
      'questions_sha256':file_hash(ROOT/'questions.jsonl'),'public_queries_sha256':file_hash(ROOT/'queries.public.jsonl'),
      'corpus_sha256':{n:file_hash(CORPUS/n) for n in ('documents.json','pages.jsonl','index-manifest.json')},
      'original_A_questions_sha256':file_hash(old/'questions.jsonl'),
      'boundary':'Same 24 products, 73-page existing rich corpus. Original A test already observed historically; mapped questions are regression comparison, not new unseen-label evidence. Natural paraphrases share product/answer clusters, not independent samples. New protocol does not overwrite A.',
      'visual_query_note':'Public query file has only text and hash. Visual model does not receive reference answers/pages or split.'})
    operation('预声明B阶段数据与受控对照','固定73页资料与同一Reader/像素/候选预算；映射旧152题并新增48道自然措辞题，按原商品组划分；公开查询文件仅文本和哈希。',
      f'{len(rows)}题，开发/回归测试各100题；6种检索模式及受限重试预声明。',[ROOT/'protocol.json',ROOT/'questions.jsonl',ROOT/'queries.public.jsonl'],
      '旧A测试已见过，B映射组如实称回归对照；不声称全新独立泛化。参考字段不进入检索器。')

if __name__=='__main__':build()

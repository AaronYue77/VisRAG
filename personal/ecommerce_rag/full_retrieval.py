"""Global full-corpus BM25/dense/RRF/cross-encoder retrieval, with exact filters."""
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from full_catalog import ROOT,connect,terms
from common import WORKSPACE,file_hash,write_json
from retrieval import rrf
from full_encoder import Encoder as DenseEncoder,MODEL_REVISION as EMBED_REVISION,VECTOR_FILE,MANIFEST_FILE

MODES=('bm25','dense','hybrid','rerank')
STOP=set('the a an is are this that it its of for with and or to find product described as please show me what which does have does how much color brand material model'.split())
EXPANSIONS={'鞋':'shoes','运动鞋':'sneakers shoes','手机壳':'cellular phone case','双肩包':'backpack','背包':'backpack',
 '耳机':'headphones','椅子':'chair','沙发':'sofa','桌子':'table','床':'bed','灯泡':'light bulb','咖啡':'coffee','狗粮':'pet food',
 '颜色':'color','材质':'material','重量':'weight','品牌':'brand','宽度':'width','高度':'height','长度':'length','黑色':'black','白色':'white','红色':'red','蓝色':'blue'}


def expand(query):
    additions=[v for k,v in EXPANSIONS.items() if k in query]
    return (query+' '+' '.join(additions)).strip()[:2000]


def exact_skus(query):
    return list(dict.fromkeys(re.findall(r'(?<![A-Z0-9])(?:B[0-9][A-Z0-9]{8}|[0-9]{9}[0-9X])(?![A-Z0-9])',query.upper())))


class FullRetriever:
    def __init__(self):
        self.db=connect();self.encoder=None;self.vectors=None;self.reranker=None
        self.manifest=json.loads((ROOT/'manifest.json').read_text('utf-8'))
        if self.manifest['records']!=147702:raise ValueError('Incomplete catalog')
        self.query_cache=ROOT/'query-vectors-e5';self.query_cache.mkdir(exist_ok=True)

    def close(self):
        self.db.close()

    def row(self,identity):
        r=self.db.execute('SELECT * FROM products WHERE id=?',(identity,)).fetchone()
        return dict(r) if r else None

    def get(self,sku,domain=None):
        where='sku=?';args=[sku.upper()]
        if domain:where+=' AND domain=?';args.append(domain)
        return [dict(r) for r in self.db.execute('SELECT * FROM products WHERE '+where+' ORDER BY domain',args)]

    def dense_query(self,query):
        key=hashlib.sha256((query+EMBED_REVISION).encode()).hexdigest();path=self.query_cache/(key+'.npy')
        meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text('utf-8'))
            if record['query']!=query or record['sha256']!=file_hash(path):raise ValueError('Query cache mismatch')
            return np.load(path,allow_pickle=False)[0],True
        if self.encoder is None:self.encoder=DenseEncoder()
        vector=self.encoder.encode([query],query=True)
        np.save(path,vector,allow_pickle=False);write_json(meta,{'query':query,'revision':EMBED_REVISION,'sha256':file_hash(path)})
        return vector[0],False

    def search(self,query,mode='bm25',top_k=5,filters=None,expand_query=False):
        if mode not in MODES:raise ValueError('Unknown retrieval mode')
        if not isinstance(query,str) or not 0<len(query.strip())<=2000:raise ValueError('Query must contain 1..2000 characters')
        if not 1<=top_k<=50:raise ValueError('top_k must be 1..50')
        filters=filters or {}
        if set(filters)-{'product_type','product_types','domain'}:raise ValueError('Only product_type/product_types/domain filters allowed')
        started=time.perf_counter();effective=expand(query) if expand_query else query
        conditions=[];args=[]
        for k,v in filters.items():
            if k=='product_types':
                if not isinstance(v,list) or not 1<=len(v)<=20 or any(not isinstance(x,str) or x not in self.manifest['product_types'] for x in v):
                    raise ValueError('product_types must contain1..20 known category names')
                conditions.append('p.product_type IN ('+','.join('?' for _ in v)+')');args.extend(v);continue
            if not isinstance(v,str):raise ValueError('Filter values must be strings')
            conditions.append('p.'+k+'=?');args.append(v)
        where=' AND '.join(conditions) or '1=1'
        trace={'mode':mode,'global_index_records':147702,'query':query,'effective_query':effective,'filters':filters,
               'dense_revision':EMBED_REVISION if mode!='bm25' else None,'rrf_c':60,'candidate_budget':20}
        tokens=list(dict.fromkeys(t for t in terms(effective).split() if t not in STOP))[:64]
        if not tokens:return {'items':[],'seconds':time.perf_counter()-started,'trace':trace}
        match=' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        # FTS5 BM25 returns smaller values for stronger matches. Convert only for display.
        sparse=self.db.execute('''SELECT p.id,bm25(products_fts,8,5,2,1) AS score
          FROM products_fts JOIN products p ON p.id=products_fts.rowid
          WHERE products_fts MATCH ? AND '''+where+' ORDER BY score,p.listing_key LIMIT 100',[match,*args]).fetchall()
        sparse_ids=[r['id'] for r in sparse];sparse_scores={r['id']:-r['score'] for r in sparse}
        dense_ids=[];dense_scores={};dense_seconds=0.0
        if mode!='bm25':
            stamp=time.perf_counter()
            if self.vectors is None:
                manifest=json.loads(MANIFEST_FILE.read_text('utf-8'))
                if manifest['rows']!=147702 or manifest['source_database_sha256']!=self.manifest['database_sha256']:raise ValueError('Stale/incomplete semantic index')
                self.vectors=np.load(VECTOR_FILE,mmap_mode='r',allow_pickle=False)
            vector,cached=self.dense_query(effective)
            if conditions:
                indices=np.array([r[0]-1 for r in self.db.execute('SELECT p.id FROM products p WHERE '+where,args)],dtype='int64')
            else:indices=np.arange(len(self.vectors))
            if len(indices):
                scores=self.vectors[indices]@vector;n=min(100,len(indices))
                top=np.argpartition(-scores,n-1)[:n];top=top[np.argsort(-scores[top],kind='stable')]
                dense_ids=[int(indices[i]+1) for i in top];dense_scores={int(indices[i]+1):float(scores[i]) for i in top}
            dense_seconds=time.perf_counter()-stamp;trace['dense_query_cache_hit']=cached
        if mode=='bm25':order,scores=sparse_ids,sparse_scores
        elif mode=='dense':order,scores=dense_ids,dense_scores
        else:order,scores=rrf([sparse_ids,dense_ids])
        before=list(order);rerank_seconds=0
        if mode=='rerank' and order:
            from b_retrieval import CrossEncoder
            candidates=order[:20]
            if self.reranker is None:self.reranker=CrossEncoder()
            stamp=time.perf_counter();output=self.reranker.score(query,[self.row(i)['dense_text'] for i in candidates])
            scores=dict(zip(candidates,output));order=sorted(candidates,key=lambda i:(-scores[i],i));rerank_seconds=time.perf_counter()-stamp
        items=[]
        for rank,identity in enumerate(order[:top_k],1):
            row=self.row(identity);items.append({**row,'rank':rank,'score':scores[identity]})
        trace.update(sparse_ranking=sparse_ids,dense_ranking=dense_ids,before_rerank=before[:20],
                     selected_listing_keys=[r['listing_key'] for r in items],dense_seconds=dense_seconds,rerank_seconds=rerank_seconds)
        return {'items':items,'seconds':time.perf_counter()-started,'trace':trace}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--mode',choices=MODES,default='bm25')
    a=p.parse_args();result=FullRetriever().search(a.query,a.mode)
    print(json.dumps({'items':[{k:r[k] for k in ('listing_key','product_type','title','rank','score')} for r in result['items']],'seconds':result['seconds']},ensure_ascii=False,indent=2))

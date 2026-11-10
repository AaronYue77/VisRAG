"""True sparse/dense/visual retrieval, rank fusion and a separate cross-encoder."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
from b_protocol import CORPUS,ROOT,INDEX,SETTINGS,query_key,VISUAL_INDEX
from stage_a_index import BM25Retriever,route
from retrieval import DenseEncoder,rrf
from common import WORKSPACE,file_hash,read_jsonl,write_json,write_jsonl,operation,issue
from core import tokenize


class CrossEncoder:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        root=WORKSPACE/'models/bge-reranker-base'
        manifest=json.loads((root/'download-manifest.json').read_text('utf-8'))
        if manifest['revision']!=SETTINGS['reranker_revision']:raise ValueError('Wrong reranker revision')
        self.tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(root,local_files_only=True,dtype=torch.float32).to('cpu').eval()
        torch.set_num_threads(4)

    def score(self,query,texts):
        import torch
        values=[]
        with torch.inference_mode():
            for start in range(0,len(texts),4):
                part=texts[start:start+4]
                inputs=self.tokenizer([query]*len(part),part,padding=True,truncation=True,max_length=384,return_tensors='pt')
                values.extend(self.model(**inputs).logits.reshape(-1).float().tolist())
        return values


def build_text():
    INDEX.mkdir(parents=True,exist_ok=True)
    if (INDEX/'text-manifest.json').exists():raise FileExistsError('Text index already complete')
    base=BM25Retriever(CORPUS,'blocks',0);encoder=DenseEncoder();started=time.perf_counter()
    values=encoder.encode([c['text'] for c in base.chunks])
    np.save(INDEX/'text-reps.npy',values,allow_pickle=False);write_jsonl(INDEX/'chunks.jsonl',base.chunks)
    write_json(INDEX/'text-manifest.json',{'model':'BAAI/bge-small-zh-v1.5','revision':SETTINGS['bge_revision'],
       'pooling':'CLS + L2','max_length':512,'device':'cpu','chunks':len(base.chunks),'shape':list(values.shape),
       'corpus_pages_sha256':file_hash(CORPUS/'pages.jsonl'),'chunks_sha256':file_hash(INDEX/'chunks.jsonl'),
       'vectors_sha256':file_hash(INDEX/'text-reps.npy'),'encoding_seconds':time.perf_counter()-started})
    tasks=read_jsonl(ROOT/'queries.public.jsonl');cache=INDEX/'query-cache';cache.mkdir(exist_ok=True)
    for i,t in enumerate(tasks):
        start=time.perf_counter();vector=encoder.encode([t['text']],query=True)
        np.save(cache/(t['query_key']+'.npy'),vector,allow_pickle=False)
        write_json(cache/(t['query_key']+'.json'),{'text':t['text'],'revision':SETTINGS['bge_revision'],
          'seconds':time.perf_counter()-start,'sha256':file_hash(cache/(t['query_key']+'.npy'))})
        if i%30==0:print('BGE queries',i+1,'/',len(tasks),flush=True)
    operation('建立B阶段独立BGE文本索引','4文字块窗口，CPU BGE CLS/L2，固定512token；公开问题/改写只编码文本，不读取gold。',
       f'{len(base.chunks)}块、{len(tasks)}查询，512维。',[INDEX/'text-manifest.json',INDEX/'chunks.jsonl'],
       '不是BM25词频冒充向量；保留截断预算与中英混合语料的模型局限。')


class BRetriever:
    def __init__(self,online_visual=False):
        self.base=BM25Retriever(CORPUS,'blocks',0);self.documents=self.base.documents
        self.pages=self.base.pages;self.by_id=self.base.by_id;self.chunks=self.base.chunks
        manifest=json.loads((INDEX/'text-manifest.json').read_text('utf-8'))
        for name,key in [('text-reps.npy','vectors_sha256'),('chunks.jsonl','chunks_sha256')]:
            if file_hash(INDEX/name)!=manifest[key]:raise ValueError('Text index changed')
        if manifest['corpus_pages_sha256']!=file_hash(CORPUS/'pages.jsonl'):raise ValueError('Stale B index')
        self.vectors=np.load(INDEX/'text-reps.npy',allow_pickle=False)
        self.encoder=None;self.reranker=None;self.online_visual=online_visual
        self.visual_root=WORKSPACE/'data/visrag_repro'/VISUAL_INDEX
        self.visual_queries=WORKSPACE/'data/visrag_repro/b-abo73-queries-v1'
        self.visual_pages=None;self.visual_vectors=None

    def dense_query(self,text):
        key=query_key(text);folder=INDEX/'query-cache';folder.mkdir(exist_ok=True)
        path=folder/(key+'.npy');meta=folder/(key+'.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text('utf-8'))
            if record['text']!=text or record['revision']!=SETTINGS['bge_revision'] or file_hash(path)!=record['sha256']:raise ValueError('Bad dense query cache')
            return np.load(path,allow_pickle=False),record['seconds'],True
        if self.encoder is None:self.encoder=DenseEncoder()
        started=time.perf_counter();vector=self.encoder.encode([text],query=True);seconds=time.perf_counter()-started
        np.save(path,vector,allow_pickle=False);write_json(meta,{'text':text,'revision':SETTINGS['bge_revision'],'seconds':seconds,'sha256':file_hash(path)})
        return vector,seconds,False

    def visual_scores(self,text):
        if self.visual_pages is None:
            manifest=json.loads((self.visual_root/'index-manifest.json').read_text('utf-8'))
            if file_hash(self.visual_root/'reps.npy')!=manifest['reps_sha256']:raise ValueError('Visual index changed')
            if file_hash(self.visual_root/'page-map.jsonl')!=manifest['page_map_sha256']:raise ValueError('Visual mapping changed')
            config=json.loads((self.visual_root/'config.json').read_text('utf-8'))
            if config['documents_sha256']!=file_hash(CORPUS/'documents.json'):raise ValueError('Stale visual corpus')
            self.visual_pages=read_jsonl(self.visual_root/'page-map.jsonl');self.visual_vectors=np.load(self.visual_root/'reps.npy',allow_pickle=False)
        key=query_key(text);p=self.visual_queries/(key+'.npy');meta=self.visual_queries/(key+'.json')
        cached=p.exists() and meta.exists();process_seconds=0.0
        if not cached:
            if not self.online_visual:raise KeyError('Query not pre-encoded; use online visual mode')
            task=INDEX/'online-queries';task.mkdir(exist_ok=True);request=task/(key+'.json')
            write_json(request,{'query_key':key,'text':text})
            started=time.perf_counter()
            subprocess.run([str(WORKSPACE/'.venv-visrag/Scripts/python.exe'),str(Path(__file__).parent.parent/'visrag_repro/b_one_query.py'),str(request)],check=True)
            process_seconds=time.perf_counter()-started
        record=json.loads(meta.read_text('utf-8'))
        if record['text']!=text or file_hash(p)!=record['sha256']:raise ValueError('Bad visual query cache')
        vector=np.load(p,allow_pickle=False);scores=(vector@self.visual_vectors.T)[0]
        return {p['page_id']:float(v) for p,v in zip(self.visual_pages,scores)},record['encoding_seconds'],cached,process_seconds

    @staticmethod
    def reserve(order,by_id,skus,budget):
        if len(skus)>1 and budget>=len(skus):
            first=[next(p for p in order if by_id[p]['sku']==sku) for sku in skus]
            order=first+[p for p in order if p not in first]
        return order[:budget]

    def search(self,question,request,mode='bm25',top_k=None):
        started=time.perf_counter();decision=route(question,request,self.documents)
        trace={'mode':mode,'rrf_c':60 if 'rrf' in mode or mode=='rerank' else None,'index':'b-index-v1','dense_query_seconds':0.0,'visual_query_seconds':0.0,'rerank_seconds':0.0}
        if decision['action']!='retrieve':return {'plan':decision,'pages':[],'trace':trace,'seconds':time.perf_counter()-started}
        skus=decision['skus'];budget=top_k or (2 if len(skus)>1 else 1)
        allowed={p['page_id'] for p in self.pages if p['sku'] in skus and p['store_id']==decision['store_id']}
        indices=[i for i,c in enumerate(self.chunks) if c['page_id'] in allowed]
        sparse=self.base.bm25.get_scores(tokenize(question));sparse_page={p:max(float(sparse[i]) for i in indices if self.chunks[i]['page_id']==p) for p in allowed}
        order=lambda scores:sorted(allowed,key=lambda p:(-scores[p],p))
        sparse_ids=order(sparse_page);rankings=[sparse_ids];dense_page={};visual={}
        if mode in {'bge','text_rrf','multimodal_rrf','rerank'}:
            query,seconds,cached=self.dense_query(question);trace.update(dense_query_seconds=seconds,dense_query_cache_hit=cached)
            scores=(query@self.vectors.T)[0];dense_page={p:max(float(scores[i]) for i in indices if self.chunks[i]['page_id']==p) for p in allowed}
            dense_ids=order(dense_page);rankings.append(dense_ids)
        if mode in {'visual','multimodal_rrf','rerank'}:
            visual,seconds,cached,process=self.visual_scores(question);trace.update(visual_query_seconds=seconds,visual_query_cache_hit=cached,visual_process_seconds=process)
            visual_ids=order(visual)
        if mode=='bm25':ordering,scores=sparse_ids,sparse_page
        elif mode=='bge':ordering,scores=dense_ids,dense_page
        elif mode=='text_rrf':ordering,scores=rrf(rankings)
        elif mode=='visual':ordering,scores=visual_ids,visual
        elif mode in {'multimodal_rrf','rerank'}:ordering,scores=rrf(rankings+[visual_ids])
        else:raise ValueError('Unknown retrieval mode')
        before=list(ordering)
        if mode=='rerank':
            candidates=self.reserve(ordering,self.by_id,skus,min(6,len(ordering)))
            pairs=[(pid,c['text']) for pid in candidates for c in self.chunks if c['page_id']==pid]
            signature=hashlib.sha256(json.dumps({'query':question,'pairs':pairs,'revision':SETTINGS['reranker_revision'],'max_length':384},sort_keys=True).encode()).hexdigest()
            folder=INDEX/'rerank-cache';folder.mkdir(exist_ok=True);cache=folder/(signature+'.json')
            if cache.exists():record=json.loads(cache.read_text('utf-8'));cached=True
            else:
                if self.reranker is None:self.reranker=CrossEncoder()
                start=time.perf_counter();values=self.reranker.score(question,[v for _,v in pairs])
                record={'scores':{pid:max(score for (key,_),score in zip(pairs,values) if key==pid) for pid in candidates},'seconds':time.perf_counter()-start,
                        'pairs':len(pairs),'revision':SETTINGS['reranker_revision']};write_json(cache,record);cached=False
            scores=record['scores'];ordering=sorted(candidates,key=lambda p:(-scores[p],p))
            trace.update(rerank_seconds=record['seconds'],rerank_cache_hit=cached,rerank_candidates=candidates,rerank_pairs=record['pairs'])
        selected=self.reserve(ordering,self.by_id,skus,budget)
        pages=[{**self.by_id[pid],'rank':i,'score':scores[pid],'bm25_score':sparse_page[pid],'dense_score':dense_page.get(pid),'visual_score':visual.get(pid)} for i,pid in enumerate(selected,1)]
        trace.update(eligible_pages=len(allowed),excluded_pages=len(self.pages)-len(allowed),top_k=budget,
                     sparse_ranking=sparse_ids,dense_ranking=rankings[1] if len(rankings)>1 else [],visual_ranking=order(visual) if visual else [],
                     before_rerank=before,page_ranking=ordering,selected_pages=selected)
        seconds=time.perf_counter()-started
        trace['warm_lookup_wall_seconds']=seconds
        trace['component_compute_seconds_estimate']=trace['dense_query_seconds']+trace['visual_query_seconds']+trace['rerank_seconds']
        return {'plan':decision,'pages':pages,'trace':trace,'seconds':seconds}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['text']);args=parser.parse_args()
    build_text()

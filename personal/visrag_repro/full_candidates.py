"""Separate original VisRAG runtime for public full-catalog candidate images."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from retrieve import model_runtime,call_encoder,INSTRUCTION,MODEL_REVISION
from common import WORKSPACE,file_hash,write_json


def main(task_path):
    task=json.loads(Path(task_path).read_text('utf-8'))
    if set(task)!={'query','pages','output'}:raise ValueError('Public query/images only')
    root=WORKSPACE/'data/ecommerce_rag/abo-full-v1/visual-cache';root.mkdir(parents=True,exist_ok=True)
    model=tokenizer=encode=None;runtime={}
    def vector(key,inputs):
        nonlocal model,tokenizer,encode,runtime
        path=root/(key+'.npy');meta=root/(key+'.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text('utf-8'))
            if record['revision']!=MODEL_REVISION or record['sha256']!=file_hash(path):raise ValueError('Bad visual cache')
            return np.load(path,allow_pickle=False),0.0,True
        if model is None:model,tokenizer,encode,runtime=model_runtime()
        values,seconds,peak=call_encoder(model,tokenizer,encode,inputs)
        np.save(path,values,allow_pickle=False);write_json(meta,{'revision':MODEL_REVISION,'sha256':file_hash(path),'encoding_seconds':seconds,'peak_cuda_gib':peak})
        return values,seconds,False
    qkey=hashlib.sha256(('query:'+task['query']+MODEL_REVISION).encode()).hexdigest()
    query,qseconds,qcached=vector(qkey,[INSTRUCTION+task['query']]);scores=[]
    for page in task['pages']:
        image_path=(WORKSPACE/page['image_path']).resolve()
        if not image_path.is_relative_to(WORKSPACE) or file_hash(image_path)!=page['image_sha256']:raise ValueError('Image path or hash mismatch')
        key=hashlib.sha256(('image:'+page['image_sha256']+MODEL_REVISION).encode()).hexdigest()
        with Image.open(image_path) as image:
            v,seconds,cached=vector(key,[image.convert('RGB')])
        scores.append({'page_id':page['page_id'],'listing_key':page['listing_key'],'score':(query@v.T).item(),
                        'encoding_seconds':seconds,'cache_hit':cached})
    output=Path(task['output']).resolve()
    if not output.is_relative_to(WORKSPACE):raise ValueError('Keep result inside workspace')
    write_json(output,{'query':task['query'],'scores':scores,'query_seconds':qseconds,'query_cache_hit':qcached,'runtime':runtime,
       'scope':'Conditional visual rerank within global text candidates only, not full image-corpus retrieval.'})


if __name__=='__main__':main(sys.argv[1])

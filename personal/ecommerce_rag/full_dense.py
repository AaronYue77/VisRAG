"""Resumable GPU document encoding for EVERY full ABO listing."""
import argparse
import json
import time
import numpy as np
from full_catalog import ROOT,DB,EXPECTED,connect
from common import WORKSPACE,write_json,file_hash,operation
from retrieval import EMBED_REVISION
from full_encoder import Encoder,MODEL_NAME,MODEL_REVISION,DIMENSION,VECTOR_FILE,MANIFEST_FILE


def build(batch_size=32):
    import torch
    from transformers import AutoTokenizer,AutoModel
    target=VECTOR_FILE;meta=MANIFEST_FILE;progress=ROOT/'dense-e5-progress.json'
    if meta.exists():print('Full dense index exists',flush=True);return
    if not DB.exists():raise FileNotFoundError('Build full_catalog first')
    source=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    settings={'model':MODEL_NAME,'revision':MODEL_REVISION,'rows':EXPECTED,'dimension':DIMENSION,
      'max_length':512,'pooling':'attention-mask mean then float32 L2; passage/query prefixes','storage_dtype':'float32',
      'source_database_sha256':source['database_sha256'],'document_representation':'full_catalog.dense_text; original metadata retained separately'}
    done=0;previous_seconds=0;limit_hits=0
    if progress.exists():
        old=json.loads(progress.read_text('utf-8'))
        if old['settings']!=settings:raise ValueError('Incompatible incomplete embedding run')
        done=old['completed'];previous_seconds=old['encoding_seconds'];limit_hits=old['token_limit_hits']
        matrix=np.lib.format.open_memmap(target,mode='r+')
    else:
        if target.exists():raise FileExistsError('Untracked partial vector file preserved')
        matrix=np.lib.format.open_memmap(target,mode='w+',dtype='float32',shape=(EXPECTED,DIMENSION))
    device='cuda:0' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    encoder=Encoder(device);tokenizer=encoder.tokenizer;model=encoder.model
    db=connect();started=time.perf_counter();cursor=db.execute('SELECT id,dense_text FROM products WHERE id>? ORDER BY id',(done,))
    peak=0
    while batch:=cursor.fetchmany(batch_size):
        inputs=tokenizer(['passage: '+r['dense_text'] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
        limit_hits+=int((inputs['attention_mask'].sum(1)>=512).sum())
        with torch.inference_mode():
            inputs=inputs.to(device);hidden=model(**inputs).last_hidden_state.float()
            pooled=hidden.masked_fill(~inputs['attention_mask'][...,None].bool(),0).sum(1)/inputs['attention_mask'].sum(1)[...,None]
            vectors=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
        assert np.isfinite(vectors).all()
        matrix[done:done+len(batch)]=vectors;done+=len(batch)
        if device.startswith('cuda'):peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
        if done%1024==0 or done==EXPECTED:
            matrix.flush();elapsed=previous_seconds+time.perf_counter()-started
            write_json(progress,{'settings':settings,'completed':done,'encoding_seconds':elapsed,'token_limit_hits':limit_hits})
            print(f'Full E5 {done}/{EXPECTED}, {elapsed:.1f}s, peak {peak:.2f} GiB',flush=True)
    assert done==EXPECTED
    norms=np.linalg.norm(matrix,axis=1)
    assert np.isfinite(matrix).all() and np.max(np.abs(norms-1))<1e-4
    matrix.flush();db.close()
    manifest={**settings,'complete':True,'device':device,'weight_dtype':'float16' if device.startswith('cuda') else 'float32',
      'encoding_seconds':previous_seconds+time.perf_counter()-started,'peak_cuda_gib':peak,'token_limit_hits':limit_hits,
      'vector_sha256':file_hash(target),'row_identity':'row index + 1 = products.id; no product sampling',
      'limitation':'Representative text is truncated at512; sparse and raw JSON retain the complete source. E5 multilinguality does not guarantee correctness on every source language.'}
    write_json(meta,manifest)
    operation('全量E5多语义索引完成','独立GPU编码全部listing，按官方passage前缀和mask mean/L2保存float32矩阵；原BGE失败索引保留，不与Qwen同时占显存。',
       f'{done}×{DIMENSION}向量，{manifest["encoding_seconds"]:.1f}秒；序列达到512token上限{limit_hits}条。',[target,meta],
       '每条元数据都有对应向量，有限/归一化和行号映射通过；截断局限明确保留。')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=32);build(parser.parse_args().batch_size)

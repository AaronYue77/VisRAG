"""Content-addressed dense chunk cache; changed text re-encodes only that chunk."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from b_protocol import INDEX,SETTINGS
from common import WORKSPACE,read_jsonl,file_hash,write_json,operation
from retrieval import DenseEncoder


class ChunkCache:
    def __init__(self):
        self.folder=INDEX/'content-vectors';self.folder.mkdir(exist_ok=True)
        self.encoder=None;self.hits=0;self.computed=0
    @staticmethod
    def key(text):
        identity={'text':text,'revision':SETTINGS['bge_revision'],'max_length':512,'pooling':'CLS+L2','dtype':'float32','dimension':512}
        return hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def get(self,text,seed=None):
        key=self.key(text);path=self.folder/(key+'.npy');meta=self.folder/(key+'.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text('utf-8'))
            if record['key']!=key or record['sha256']!=file_hash(path):raise ValueError('Cached vector changed')
            vector=np.load(path,allow_pickle=False);self.hits+=1
        else:
            if seed is not None:vector=np.asarray(seed,dtype='float32')
            else:
                if self.encoder is None:self.encoder=DenseEncoder()
                vector=self.encoder.encode([text])[0];self.computed+=1
            np.save(path,vector,allow_pickle=False)
            write_json(meta,{'key':key,'sha256':file_hash(path),'model_revision':SETTINGS['bge_revision'],'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
        assert vector.shape==(512,) and np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector),1,atol=1e-5)
        return vector


def verify():
    chunks=read_jsonl(INDEX/'chunks.jsonl');original=np.load(INDEX/'text-reps.npy',allow_pickle=False)
    cache=ChunkCache()
    # Seed only from already verified real encoder outputs; never invent vectors.
    for c,v in zip(chunks,original):cache.get(c['text'],seed=v)
    reused=ChunkCache();a=np.stack([reused.get(c['text']) for c in chunks]);assert np.allclose(a,original,atol=1e-5,rtol=0)
    modified=[dict(c) for c in chunks];modified[0]['text']+='\nCACHE INVALIDATION PROBE: isolated non-product test text.'
    changed=ChunkCache();b=np.stack([changed.get(c['text']) for c in modified])
    assert changed.computed==1 and changed.hits==len(chunks)-1
    assert np.array_equal(a[1:],b[1:]) and not np.array_equal(a[0],b[0])
    out=WORKSPACE/'reports/maintenance/b-stage-20261104'
    write_json(out/'dense-incremental.json',{'chunks':len(chunks),'unchanged_cache_hits':reused.hits,'unchanged_computed':reused.computed,
       'one_chunk_changed_hits':changed.hits,'one_chunk_changed_computed':changed.computed,'unchanged_other_rows_exactly_equal':True,
       'scope':'An isolated text probe, not a change to official corpus or benchmark. BM25 global IDF still recomputed; no claim of concurrent vector database.'})
    (out/'向量增量缓存验证.txt').write_text(f'方法：以实际BGE输出填充内容寻址缓存，保持所有块不变再读；在独立列表中只追加一条明确标记的缓存探针文字。\n结果：不变{reused.hits}块全部复用，0次编码；只改1块后复用{changed.hits}块，只实际编码1块，其他向量逐字节相同。\n键含原文、模型revision、最大token、pooling、dtype和维度；元数据保存向量SHA。\n主73页PDF、主索引、问题和评测未改。\n边界：本地逐块重用，不是生产向量库并发增量服务；视觉页已预计算，当前未实现局部图像patch增量。','utf-8')
    operation('验证BGE向量内容增量缓存','从真实编码输出填缓存，未改文本全复用，再仅改变独立探针块，核对其余向量字节。',
        f'{len(chunks)}块；不变0次编码；只改1块仅编码1次。',[out/'向量增量缓存验证.txt',out/'dense-incremental.json'],
        '原PDF/主索引未改；没有将探针文本写入真实商品事实；BM25统计仍重建。')

if __name__=='__main__':verify()

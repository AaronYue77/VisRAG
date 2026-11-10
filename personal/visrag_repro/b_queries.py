"""Original VisRAG query encoding; input accepts only query text/hash, no labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from retrieve import model_runtime,call_encoder,INSTRUCTION
from common import WORKSPACE,read_jsonl,write_json,write_jsonl,file_hash,operation,issue
import numpy as np

def main():
    root=WORKSPACE/'data/ecommerce_rag/b-evaluation-v1'
    output=WORKSPACE/'data/visrag_repro/b-abo73-queries-v1';output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError('Keep completed query encoding')
    tasks=read_jsonl(root/'queries.public.jsonl')
    assert all(set(r)=={'text','query_key'} for r in tasks)
    model,tokenizer,encode,runtime=model_runtime();rows=[];vectors=[]
    for task in tasks:
        target=output/(task['query_key']+'.npy')
        meta=output/(task['query_key']+'.json')
        if target.exists() and meta.exists():
            record=__import__('json').loads(meta.read_text('utf-8'))
            assert record['model_input']==INSTRUCTION+task['text'] and record['sha256']==file_hash(target)
            vector=np.load(target,allow_pickle=False)
        else:
            vector,seconds,peak=call_encoder(model,tokenizer,encode,[INSTRUCTION+task['text']])
            np.save(target,vector,allow_pickle=False)
            record={**task,'model_input':INSTRUCTION+task['text'],'encoding_seconds':seconds,'peak_cuda_gib':peak,'sha256':file_hash(target)}
            write_json(meta,record)
        rows.append({**record,'vector_row':len(rows)});vectors.append(vector)
        if len(rows)%20==0:print(f'Visual queries {len(rows)}/{len(tasks)}',flush=True)
    np.save(output/'query_reps.npy',np.concatenate(vectors),allow_pickle=False);write_jsonl(output/'query-map.jsonl',rows)
    write_json(output/'manifest.json',{'runtime':runtime,'queries':len(rows),'input_sha256':file_hash(root/'queries.public.jsonl'),
              'query_map_sha256':file_hash(output/'query-map.jsonl'),'vectors_sha256':file_hash(output/'query_reps.npy')})
    operation('B阶段真实问题的原版视觉编码','独立3.11环境；复用原版encode与instruction，输入只有文本/哈希；2304维L2，逐题保存耗时，结束进程释放GPU。',
      f'{len(rows)}个唯一查询/改写向量已保存。',[output/'query-map.jsonl',output/'manifest.json'],'编码不使用gold/划分；页面与查询同一原始VisRAG向量空间。')

if __name__=='__main__':
    try:main()
    except Exception as e:
        k=issue('B视觉查询编码失败：'+str(e),'检查原版环境和输入。','保留中间向量及对应校验信息，修复后继续。')
        operation('B视觉查询失败','独立原版编码',str(e),status='失败',related_issues=k);raise

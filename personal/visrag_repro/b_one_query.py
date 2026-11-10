"""Bounded live visual query worker, called before the reader owns CUDA."""
import json,sys
from pathlib import Path
import numpy as np
from retrieve import model_runtime,call_encoder,INSTRUCTION
from common import WORKSPACE,write_json,file_hash,operation

task=json.loads(Path(sys.argv[1]).read_text('utf-8'))
assert set(task)=={'query_key','text'}
import hashlib
assert hashlib.sha256(task['text'].encode()).hexdigest()==task['query_key']
folder=WORKSPACE/'data/visrag_repro/b-abo73-queries-v1';folder.mkdir(exist_ok=True)
path=folder/(task['query_key']+'.npy')
model,tokenizer,encode,runtime=model_runtime()
vector,seconds,peak=call_encoder(model,tokenizer,encode,[INSTRUCTION+task['text']])
np.save(path,vector,allow_pickle=False)
write_json(folder/(task['query_key']+'.json'),{**task,'model_input':INSTRUCTION+task['text'],'encoding_seconds':seconds,'peak_cuda_gib':peak,'sha256':file_hash(path)})
operation('B交互问题视觉编码','独立VisRAG子进程先编码问题，退出释放显存，再由主进程加载问答模型。',task['query_key'],[path],'只编码文本，不读取用户历史回答或参考标签。')

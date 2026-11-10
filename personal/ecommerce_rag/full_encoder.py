"""Official multilingual E5 mean-pooling and query/passage prefixes."""
import numpy as np
from full_catalog import ROOT
from common import WORKSPACE

MODEL_NAME='intfloat/multilingual-e5-small'
MODEL_REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'
DIMENSION=384
VECTOR_FILE=ROOT/'dense-e5.npy'
MANIFEST_FILE=ROOT/'dense-e5-manifest.json'


class Encoder:
    def __init__(self,device='cpu'):
        import torch
        from transformers import AutoTokenizer,AutoModel
        self.device=device;torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(WORKSPACE/'models/multilingual-e5-small',local_files_only=True)
        self.model=AutoModel.from_pretrained(WORKSPACE/'models/multilingual-e5-small',local_files_only=True,
            dtype=torch.float16 if device.startswith('cuda') else torch.float32).to(device).eval()
    def encode(self,texts,query=False,batch_size=16):
        import torch
        vectors=[]
        with torch.inference_mode():
            for i in range(0,len(texts),batch_size):
                part=[('query: ' if query else 'passage: ')+s for s in texts[i:i+batch_size]]
                inputs=self.tokenizer(part,max_length=512,padding=True,truncation=True,return_tensors='pt').to(self.device)
                hidden=self.model(**inputs).last_hidden_state.float()
                mask=inputs['attention_mask'][...,None].bool()
                pooled=hidden.masked_fill(~mask,0).sum(1)/inputs['attention_mask'].sum(1)[...,None]
                vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
        return np.concatenate(vectors)

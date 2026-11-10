"""Conditional visual rerank; full metadata recall first, source photos on demand."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from full_catalog import ROOT
from full_cards import ensure_card,ensure_photo_card
from common import WORKSPACE,write_json,file_hash
from core import render_page


def rerank(query,items,reader=None):
    candidates=items[:6];pages=[]
    directory=ROOT/'visual-tasks';directory.mkdir(exist_ok=True)
    for row in candidates:
        doc=ensure_photo_card(row,ensure_card(row))
        for page in doc['pages']:
            if not page['photo_page']:continue
            path=directory/(page['doc_id']+'-photo.png')
            if not path.exists():render_page(WORKSPACE/page['pdf'],page['page'],786432).save(path)
            pages.append({'page_id':page['page_id'],'listing_key':row['listing_key'],'image_path':str(path.relative_to(WORKSPACE)),
                          'image_sha256':file_hash(path)})
            break
    if not pages:return items,{'scope':'no candidate photos available','visual_candidates':0}
    key=hashlib.sha256(json.dumps({'query':query,'pages':pages},sort_keys=True).encode()).hexdigest()
    output=directory/(key+'.result.json');task=directory/(key+'.task.json');cached=output.exists()
    start=time.perf_counter()
    if not cached:
        write_json(task,{'query':query,'pages':pages,'output':str(output)})
        if reader is not None and reader.reader is not None:reader.unload()
        subprocess.run([str(WORKSPACE/'.venv-visrag/Scripts/python.exe'),str(Path(__file__).parent.parent/'visrag_repro/full_candidates.py'),str(task)],check=True)
    result=json.loads(output.read_text('utf-8'));scores={r['listing_key']:r['score'] for r in result['scores']}
    eligible=sorted([r for r in candidates if r['listing_key'] in scores],key=lambda r:(-scores[r['listing_key']],r['listing_key']))
    rest=[r for r in items if r['listing_key'] not in scores]
    ordering=[{**r,'rank':i,'visual_score':scores.get(r['listing_key'])} for i,r in enumerate(eligible+rest,1)]
    return ordering,{'scope':result['scope'],'visual_candidates':len(pages),'seconds':time.perf_counter()-start,
                     'cache_hit':cached,'model_revision':'95ef596df871b606167cb7e4b7215caf1bfdf761',
                     'worker_source_sha256':file_hash(Path(__file__).parent.parent/'visrag_repro/full_candidates.py'),
                     'scores':result['scores'],'result_file':str(output.relative_to(WORKSPACE))}

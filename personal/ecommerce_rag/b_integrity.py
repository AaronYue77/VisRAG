"""Verify B products against the opening A preservation snapshot."""
import ast
import json
from pathlib import Path
import numpy as np
import b_protocol
from b_evaluate import protocol
from b_retrieval import BRetriever
from common import WORKSPACE,file_hash,read_jsonl,write_json,operation,issue


def main():
    out=WORKSPACE/'reports/maintenance/b-stage-20261104'
    protected=json.loads((out/'protected-before.json').read_text('utf-8'))
    changes=[e['path'] for e in protected if not (WORKSPACE/e['path']).exists() or file_hash(WORKSPACE/e['path'])!=e['sha256']]
    protocol();differences=[];count=0
    for split in ('dev','test'):
        for mode in b_protocol.MODES+['retry']:
            run=WORKSPACE/f'reports/ecommerce_rag/b-{split}-{mode}-v1'
            rows=read_jsonl(run/'results.jsonl')
            assert len(rows)==len({r['question_id'] for r in rows})==100
            assert (run/'逐题结果.txt').is_file() and (run/'缓存与有限重试记录.txt').is_file()
            for r in rows:
                assert r['split']==split
                if r['reference_pages']:
                    expected=float(bool(r['metrics']['answer_correct']) and set(r['reference_pages'])<=set(r['citations']))
                    if expected!=r['metrics']['evidence_aligned_fact']:differences.append([split,mode,r['question_id'],expected,r['metrics']['evidence_aligned_fact']])
            count+=len(rows)
    engine=BRetriever()
    task=read_jsonl(b_protocol.ROOT/'queries.public.jsonl')[0]
    engine.visual_scores(task['text'])  # Initialize the lazy page arrays, using an existing public query cache.
    vectors={}
    for label,array in [('text',engine.vectors),('visual',engine.visual_vectors)]:
        vectors[label]={'shape':list(array.shape),'finite':bool(np.isfinite(array).all()),
                        'max_norm_error':float(np.max(np.abs(np.linalg.norm(array,axis=1)-1)))}
        assert vectors[label]['finite'] and vectors[label]['max_norm_error']<1e-4
    code=Path(__file__).resolve().parent
    files=list(code.glob('b_*.py'))+list((code.parent/'visrag_repro').glob('b_*.py'))
    for p in files:ast.parse(p.read_text('utf-8'))
    assert not changes,changes
    assert not differences,differences
    result={'protected_existing_files':len(protected),'existing_files_changed':changes,
            'primary_records':count,'uniform_score_differences':differences,'vector_checks':vectors,
            'new_B_python_syntax_files':len(files),'original_protocol_integrity':True}
    write_json(out/'final-integrity.json',result)
    (out/'原A保护与B完整性核验.txt').write_text('B最终文件与评测完整性核验\n'+json.dumps(result,ensure_ascii=False,indent=2)+
        '\n原A源码/数据/历史报告与开工SHA逐一核对未变。手工日志、总表和说明按要求追加，不在保护原件集合。\n','utf-8')
    key=issue('临时验收脚本错误读取惰性视觉向量，另一次Git检查从外层工作区执行',
       'BRetriever.visual_vectors在首次visual_scores前为None；真实Git仓库是upstream/VisRAG，外层并非仓库。均为人工检查命令问题，未修改产物。',
       '改成可复用b_integrity.py，先用现有公开查询缓存初始化数组；Git检查定位到upstream/VisRAG。',
       '本次SHA、范数、评分、语法检查全部通过；Git diff --check在真实仓库返回0。',status='已解决')
    operation('最终核对原A保护与B产物','对照开工逐文件SHA；核验协议、1400条唯一题ID/划分、全参考证据计分、向量范数和新B源码语法。',
       json.dumps(result,ensure_ascii=False),[out/'原A保护与B完整性核验.txt'],
       '原A保护项无变化；统一计分无差异；没有覆盖旧测试或提交用户改动。',related_issues=key)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

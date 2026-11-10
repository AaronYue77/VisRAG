"""Export reproducible figures from frozen B predictions, including negative results."""
import os
from b_protocol import MODES
from common import WORKSPACE,read_jsonl,operation
os.environ['MPLCONFIGDIR']=str(WORKSPACE/'cache/matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=WORKSPACE/'reports/ecommerce_rag';out=root/'b-stage-acceptance-v1';out.mkdir(exist_ok=True)
    modes=MODES+['retry','a-topic-reference']
    labels=['BM25\nraw query','BGE','Text\nRRF','VisRAG','Multimodal\nRRF','Cross\nreranker','BGE +\nretry','A topic\nreference*']
    x=np.arange(len(modes));fig,ax=plt.subplots(figsize=(11.5,5.6),layout='constrained')
    for split,offset,color in [('dev',-.19,'#477fa5'),('test',.19,'#d88e51')]:
        values=[json.loads((root/f'b-{split}-{m}-v1/summary.json').read_text('utf-8'))['evidence_aligned_accuracy'] for m in modes]
        bars=ax.bar(x+offset,values,.36,label=split.capitalize(),color=color)
        ax.bar_label(bars,labels=[f'{v*100:.1f}' for v in values],padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.13),ylabel='Answer + annotated evidence alignment',
           title='B fixed regression comparison: 80 answerable questions per split')
    ax.set_yticks(np.arange(0,1.01,.2),[f'{v}%' for v in range(0,101,20)]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(loc='upper left',ncols=2,frameon=False)
    fig.supxlabel('* A topic reference added after six-arm selection. Same reader; not the full A dialogue benchmark.\nShared-product paraphrases and historical labels limit generalization.',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/('retrieval-comparison.'+ext),dpi=180)
    plt.close(fig)
    # Failure rates conditional on each field, with denominators shown explicitly.
    matrix=[];fields=[]
    for mode in MODES+['retry']:
        rows=[r for r in read_jsonl(root/f'b-test-{mode}-v1/results.jsonl') if r['reference_pages']]
        if not fields:fields=sorted(set((r['field'],r['suite']=='natural_phrasing') for r in rows))
        matrix.append([sum(not r['metrics']['evidence_aligned_fact'] for r in rows if (r['field'],r['suite']=='natural_phrasing')==field)/sum((r['field'],r['suite']=='natural_phrasing')==field for r in rows) for field in fields])
    counts=[sum((r['field'],r['suite']=='natural_phrasing')==f for r in rows) for f in fields]
    fig,ax=plt.subplots(figsize=(10.5,5.0),layout='constrained')
    picture=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto')
    ax.set(yticks=np.arange(7),yticklabels=labels[:7],xticks=np.arange(len(fields)),
           xticklabels=[f'{f.replace("visual_", "")}\n'+('natural\n' if n else '')+f'n={c}' for (f,n),c in zip(fields,counts)],
           title='Test regression failures by field (answer + annotated evidence)')
    for i,line in enumerate(matrix):
        for j,v in enumerate(line):ax.text(j,i,f'{100*v:.0f}%',ha='center',va='center',color='white' if v>.65 else 'black',fontsize=9)
    fig.colorbar(picture,ax=ax,label='Failure fraction',shrink=.85)
    for ext in ('png','svg'):fig.savefig(out/('failure-by-field.'+ext),dpi=180)
    plt.close(fig)
    operation('导出B受控对照与失败分布图','只读取冻结原始结果，绘制开发/回归测试证据对齐及按字段失败分布；补充A参考明确标星。',
       '两幅PNG和SVG已保存，包含重排退化、视觉问题和后补参考。',[out/'retrieval-comparison.png',out/'failure-by-field.png'],
       '不将共享商品改写当独立样本，不把探索性区间当线上收益保证。')


if __name__=='__main__':main()

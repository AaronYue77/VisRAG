"""Export scientific plots from saved A results; never rerun model predictions."""
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(1,str(Path(__file__).resolve().parent.parent/"rag_common"))
from common import WORKSPACE,operation
os.environ["MPLCONFIGDIR"]=str(WORKSPACE/"cache/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=WORKSPACE/"reports/ecommerce_rag"
out=root/"a-stage-acceptance-v1"
names=["abo-a-dev-v1","abo-a-dev-blocks-v2","abo-a-dev-labels-v3"]
values=[json.loads((root/name/"summary.json").read_text("utf-8")) for name in names]
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
for shift,key,label,color in [(-.18,"answerable_fact_accuracy","Fact + action", "#346e9c"),(.18,"evidence_aligned_fact_accuracy","Fact + action + evidence page", "#d38834")]:
    bars=ax.bar([i+shift for i in range(3)],[v[key]*100 for v in values],width=.34,label=label,color=color)
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{round(v[key]*56)}/56",ha="center",fontsize=10)
ax.set(xticks=[0,1,2],xticklabels=["Page BM25\nPrompt v1","Block BM25\nPrompt v1","Block BM25\nCitation prompt v2"],ylim=(0,115),ylabel="Correct answerable tasks (%)",title="A development ablation: gains and regressions retained")
ax.legend(loc="upper left",frameon=False,fontsize=9);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/"dev-ablation.png",dpi=180);fig.savefig(out/"dev-ablation.svg");plt.close(fig)
rows=json.loads((out/"resolution-paired.json").read_text("utf-8"))
fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
for r in rows:axes[0].plot([0,1],[r["high_generation_seconds"],r["low_generation_seconds"]],"o-",alpha=.6,color="#346e9c")
axes[0].set(xticks=[0,1],xticklabels=["786432 pixels/page","196608 pixels/page"],ylabel="Generation time (seconds)",title="Eight paired dev diagnostics")
axes[1].bar(["High resolution","Low resolution"],[statistics.mean(r["high_visual_tokens"] for r in rows),statistics.mean(r["low_visual_tokens"] for r in rows)],color=["#346e9c","#79ac91"])
axes[1].set(ylabel="Mean visual tokens from image grid",title="Same questions and retrieved pages")
fig.savefig(out/"resolution-paired.png",dpi=180);fig.savefig(out/"resolution-paired.svg");plt.close(fig)
operation("导出A阶段实验图","Matplotlib读取原始汇总与同题分辨率数据，不重跑推理、不改写预测。","分块/引用消融和分辨率-开销图已输出。",[out/"dev-ablation.png",out/"resolution-paired.png"],"对应56道开发可回答题及8道配对诊断，图中保留引用退化；随后检查渲染。")

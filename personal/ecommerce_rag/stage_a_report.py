"""Build reviewable A-stage reports from immutable prediction files."""
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import statistics

from stage_a import percentile
from stage_a_index import DEFAULT_CORPUS
from common import WORKSPACE, read_jsonl, write_json, operation, file_hash

REPORTS=WORKSPACE/"reports/ecommerce_rag"
RECORDS=WORKSPACE/"记录"


def cluster_keys(rows):
    parents={r["family_key"]:r["family_key"] for r in rows}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    for row in rows:
        for key in row.get("document_families",[]):parents[find(key)]=find(row["family_key"])
    return {r["question_id"]:find(r["family_key"]) for r in rows}


def paired(before,after,metric):
    a={r["question_id"]:r for r in before if r["reference_pages"]}
    b={r["question_id"]:r for r in after if r["reference_pages"]}
    assert set(a)==set(b)
    labels=cluster_keys(list(a.values()));groups={}
    for key in sorted(a):groups.setdefault(labels[key],[]).append(b[key]["metrics"][metric]-a[key]["metrics"][metric])
    observed=statistics.mean(v for values in groups.values() for v in values)
    rng=random.Random(20261101);keys=sorted(groups);samples=[]
    for _ in range(4000):samples.append(statistics.mean(v for key in rng.choices(keys,k=len(keys)) for v in groups[key]))
    totals=[sum(values) for values in groups.values() if sum(values)!=0]
    if totals:
        extremes=sum(abs(sum(s*v for s,v in zip(signs,totals)))>=abs(sum(totals))-1e-10 for signs in itertools.product((-1,1),repeat=len(totals)))
        p=extremes/(2**len(totals))
    else:p=1.0
    return {"questions":len(a),"clusters":len(groups),"mean_difference":observed,"cluster_bootstrap_95ci":[percentile(samples,.025),percentile(samples,.975)],
            "cluster_sign_flip_two_sided_p":p,"test_assumption":"Cluster differences symmetric under null; custom small dev set. Not randomized online A/B or population proof.",
            "improved_questions":[k for k in a if b[k]["metrics"][metric]>a[k]["metrics"][metric]],
            "regressed_questions":[k for k in a if b[k]["metrics"][metric]<a[k]["metrics"][metric]]}


def main():
    RECORDS.mkdir(exist_ok=True)
    out=REPORTS/"a-stage-acceptance-v1";out.mkdir(exist_ok=True)
    names=["abo-a-dev-v1","abo-a-dev-blocks-v2","abo-a-dev-labels-v3","abo-a-dev-lowres-v1","abo-a-dev-oracle-v1","abo-a-test-final-v1"]
    runs={name:read_jsonl(REPORTS/name/"results.jsonl") for name in names}
    summaries={name:json.loads((REPORTS/name/"summary.json").read_text("utf-8")) for name in names}
    pairs={"page_to_blocks_answer":paired(runs[names[0]],runs[names[1]],"answer_correct"),
           "page_to_blocks_evidence":paired(runs[names[0]],runs[names[1]],"evidence_aligned_fact"),
           "citation_prompt_only":paired(runs[names[1]],runs[names[2]],"evidence_aligned_fact")}
    write_json(out/"paired-analysis.json",pairs)
    required=["question_id","question","reference_answer","reference_pages","retrieved_pages","answer","citations","retrieval_seconds","generation_seconds","model","model_revision","config_sha256","source_sha256"]
    verified={}
    for name,rows in runs.items():
        config_path=REPORTS/name/"config.json";config=json.loads(config_path.read_text("utf-8"))
        for r in rows:
            assert all(k in r for k in required),r["question_id"]
            assert r["config_sha256"]==file_hash(config_path)
            assert all((WORKSPACE/p).is_file() for p in r.get("image_paths",[]))
            if r["answer_source"]=="rule_router":assert r["model"] is None and r["generation_seconds"]==0
        assert len(rows)==config["planned_questions"]
        verified[name]={"records":len(rows),"results_sha256":file_hash(REPORTS/name/"results.jsonl"),"config_sha256":file_hash(config_path),"complete":True}
    write_json(out/"verification.json",verified)
    lines=["A阶段验收报告：电商真实商品多模态RAG","#"*100,"",
      "定位：AI应用开发个人项目。当前是带规则路由的RAG工作流；尚未实现自主Agent、业务工具调用或模型训练。",
      "完成链路：官方商品数据 → 来源可追溯PDF卡 → 原生文字/必要OCR → BM25文字块候选 → 父页面去重 → 页面图片 → Qwen3.5-2B → 引用/拒答/评测/TXT。",
      "数据：Amazon官方ABO，24个CHAIR商品、48页、152道自建题，开发/测试各76题。8页是受控光栅化OCR样例。",
      "重要边界：ABO原数据是JSON属性和图片；PDF为本项目排版转换件，不是厂家原始手册，QA不是官方问答标注。原始数据也有字段质量问题。",
      "来源：https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html；授权和归属见data/ecommerce_rag/abo-cards-v1/ATTRIBUTION.txt。",
      "分组：同商品全部页面和问题在同一集合。排除共享主图/型号和标题Jaccard>=0.45的近似产品；family_key为自建启发式组，不是官方parent SKU，不能保证识别全部未知变体。",
      "测试前锁定：blocks窗口4块、0重叠、Top-2、786432像素/页、固定模型revision和引用提示v2。测试无oracle、未用于调参。",
      "型号/站点与日期：按ASIN和amazon.com范围过滤；快照登记日不是资料原始发布日期或实际有效期，没有伪造厂家历史版本。",
      "路由：静态资料检索、缺商品/上下文冲突澄清、未知商品/范围/日期/政策缺失拒答、实时业务requires_tool、场外unsupported。requires_tool只是状态，未执行真实工具。",
      "", "正式测试（一次冻结运行）：",json.dumps(summaries[names[-1]],ensure_ascii=False,indent=2),"",
      "三个开发集受控版本：每组76题；可回答56题。先只改变分块，再只改变引用提示。"]
    for name in names[:3]:
        s=summaries[name]
        lines.append(f"{name}: Recall@2={s['recall_at_k']:.4f}; 事实匹配={s['answerable_fact_accuracy']:.4f}; 证据对齐事实={s['evidence_aligned_fact_accuracy']:.4f}; 生成P50={s['generation_p50_seconds']:.3f}s")
    lines += ["","配对统计（按跨商品比较连接后的商品组重采样，保留退化）：",json.dumps(pairs,ensure_ascii=False,indent=2),
      "小样本提升不能直接称为稳定业务收益。CI/p值用于说明此次自建开发子集，不能替代新商品/真实用户验证。",
      "", "OCR对照：", "关闭OCR时8页无文字；文字块Top-2开发集Recall由0.8393到1.0000。整页Top-2对照的命中率变化并非同方向，说明分块/空页回退会影响指标；未声称OCR对任何检索策略都提升。",
      "文件：a-retrieval-dev-grid-v2/检索对照实验.txt 与 a-retrieval-dev-no-ocr-v1/检索对照实验.txt。",
      "", "分辨率与oracle诊断：", "8题同题诊断，低分辨率与高分辨率保持其他配置相同；低分辨率8/8，高分辨率对应题7/8。少量诊断不能证明低分辨率普遍更优，正式测试保留已跑全开发集的高分辨率。",
      "oracle只用开发集人工正确页，7/8；扶手题仍答错，说明不是缺正确证据。人工正确页诊断不能计入正式检索分数。",
      "", "尚未解决的实际错误与评分风险："]
    for name in (names[2],names[-1]):
        for r in runs[name]:
            if r["failure_category"]!="none":
                lines += [name+" | "+r["question_id"]+" | "+r["failure_category"],"参考："+r["reference_answer"],"模型："+r["answer"],"引用："+str(r["citations"]),"原始输出："+r["raw_output"]]
                if r.get("output_guard")=="invalid_output_or_citation":lines.append("人工排查：原始输出未满足JSON/引用协议；即便裸答案事实正确，系统仍拒答。原始结果保留，未放宽解析器或重跑测试追分。")
    lines += ["字段/数字单位匹配只是一种自动评分。长描述同义改写可能被低估，原始字段异常也可能被照抄；证据页对齐不是完整引用蕴含评测。错误原文、引用和图片全部保留。",
      "模板问题含商品编号和中英字段提示，48页小规模语料较简单；不能声称真实客服普遍准确率。",
      "单商品卡仅2页，Top-2即覆盖该商品全部页面，高召回部分来自任务设计；跨商品有4页而预算2页，才暴露本次检索排序短板。后续应扩到更长资料和自然语言问题。",
      "未接入用户图片搜索、自然扫描手册大库、视频、向量数据库、重排、LoRA、GRPO、MCP、订单退款操作；按B/C/Agent阶段推进。",
      "", "记录入口：",str(RECORDS/"逐题结果汇总.txt"),str(REPORTS/names[-1]/"逐题结果.txt"),"upstream/VisRAG/personal/a_baseline/docs/操作方法与结果记录.txt",
      "upstream/VisRAG/personal/a_baseline/docs/问题与解决记录.txt","", "验收：六组完整模型实验结果的字段、配置SHA、图片路径及条数均验证通过；API/单元测试另外记录。"]
    (RECORDS/"A阶段完成情况与验收报告.txt").write_text("\n".join(lines),"utf-8",newline="\n")
    (out/"实验对照与统计.txt").write_text("\n".join(lines),"utf-8",newline="\n")
    resolution=[]
    high={r["question_id"]:r for r in runs[names[2]]}
    for low in runs[names[3]]:
        h=high[low["question_id"]]
        resolution.append({"question_id":low["question_id"],"high_correct":h["metrics"]["answer_correct"],"low_correct":low["metrics"]["answer_correct"],
           "high_generation_seconds":h["generation_seconds"],"low_generation_seconds":low["generation_seconds"],
           "high_visual_tokens":h["visual_tokens_from_grid"],"low_visual_tokens":low["visual_tokens_from_grid"],
           "high_peak_cuda_gib":h["peak_cuda_gib"],"low_peak_cuda_gib":low["peak_cuda_gib"]})
    write_json(out/"resolution-paired.json",resolution)
    operation("汇总A阶段验收、消融与统计","从不可改写的JSONL读取，验证字段/配置哈希/图片；按商品连接组做配对bootstrap和符号翻转；分别报告事实与引用退化。",
        f"{sum(len(v) for v in runs.values())}条完整管线实验记录（含路由不调用模型的题）可核对；正式测试{len(runs[names[-1]])}题。",[RECORDS/"A阶段完成情况与验收报告.txt",out/"实验对照与统计.txt",out/"verification.json"],"不把模拟16题、oracle诊断或无标签交互混为正式测试；失败与统计局限明确保留。")


if __name__=="__main__":main()

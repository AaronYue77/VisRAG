"""Shared evidence prompting, scoring and training input for the 8B experiments."""
from collections import defaultdict
import json
import re
import statistics

from common import write_json


PROMPT_C1 = (
    "\n核对步骤（无需输出过程）：先在图中找到问题指定的商品编号和字段名；"
    "逐字符抄写该字段的数值和原单位，别取标题中相似数字，也不要换算。"
    "两个商品都找到才逐项作答并引用两张证据页。只引用支持答案的图片。"
    "任一所需字段看不清或未报告，输出指定的 NOT_FOUND JSON。"
)


def compact_answer(answer):
    return re.sub(r"\s+", "", answer).casefold().replace("：", ":").replace("｜", "|")


def score(row, result, pages):
    raw = result["raw_output"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    valid = (isinstance(parsed, dict) and set(parsed) == {"action", "answer", "citations"}
             and parsed["action"] in ("answer", "not_found") and isinstance(parsed["answer"], str)
             and isinstance(parsed["citations"], list)
             and all(isinstance(label, str) and re.fullmatch(r"P[1-9]\d*", label) for label in parsed["citations"])
             and not result.get("invalid_citations"))
    exact = compact_answer(result["answer"]) == compact_answer(row["reference_answer"])
    cited, gold = set(result["citations"]), set(row["reference_pages"])
    answerable = row["expected_action"] == "answer"
    strict = (valid and parsed["action"] == row["expected_action"] and exact and cited == gold
              and (answerable or (parsed["answer"] == "NOT_FOUND" and parsed["citations"] == [])))
    return {"schema_valid": bool(valid), "exact_value_unit": bool(exact) if answerable else None,
            "correct_evidence": cited == gold if answerable else None,
            "gold_in_candidates": gold <= {page["page_id"] for page in pages} if answerable else None,
            "strict_pass": bool(strict), "false_refusal": answerable and result["action"] == "not_found",
            "unsupported_answer": not answerable and result["action"] == "answer"}


def training_input(reader, row, images, cfg):
    import torch

    pages = row["retrieved_pages"]
    page_labels = {page["page_id"]: f"P{i}" for i, page in enumerate(pages, 1)}
    labels = [page_labels[page] for page in row["reference_pages"]]
    target = json.dumps({"action": row["expected_action"], "answer": row["reference_answer"],
                         "citations": labels}, ensure_ascii=False)
    inputs, info = reader.prepare(row["question"], row["request"], pages, images, cfg["max_pixels"], PROMPT_C1)
    prefix = inputs["input_ids"].shape[1]
    tokenizer = reader.vlm.processor.tokenizer
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    eos = target_ids.new_tensor([[tokenizer.eos_token_id]])
    target_ids = torch.cat((target_ids, eos), dim=1)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], target_ids), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(target_ids)), dim=1)
    types = inputs["mm_token_type_ids"]
    inputs["mm_token_type_ids"] = torch.cat((types, types.new_zeros(target_ids.shape)), dim=1)
    # Predict response tokens only. The last prefix position predicts the first response token.
    positions = torch.arange(prefix - 1, inputs["input_ids"].shape[1] - 1, device="cuda")
    return inputs, target_ids, positions, info


def report(out, completed):
    groups = defaultdict(list)
    for (variant, _), row in completed.items():
        groups[(variant, row["split"], "all")].append(row)
        groups[(variant, row["split"], row["suite"])].append(row)
    metrics = []
    for (variant, split, suite), rows in sorted(groups.items()):
        totals = {"variant": variant, "split": split, "suite": suite, "n": len(rows),
                  "median_generation_seconds": statistics.median(row["generation_seconds"] for row in rows)}
        for key in rows[0]["metrics"]:
            values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
            totals[key] = {"passed": sum(values), "n": len(values)}
        metrics.append(totals)
    changes = []
    for a, b in (("C0", "C1"), ("C1", "C2")):
        for (variant, qid), row in completed.items():
            if variant != a:
                continue
            later = completed[(b, qid)]
            if row["metrics"]["strict_pass"] != later["metrics"]["strict_pass"]:
                changes.append({"comparison": f"{a} to {b}", "split": row["split"], "suite": row["suite"],
                                "question_id": qid, "question": row["question"], "reference": row["reference_answer"],
                                "before_pass": row["metrics"]["strict_pass"], "before": row["raw_output"],
                                "after_pass": later["metrics"]["strict_pass"], "after": later["raw_output"]})
    write_json(out / "metrics.json", metrics)
    write_json(out / "paired-changes.json", changes)
    intro = ("C0 原提示，C1 原提示加核对说明，C2 重载 LoRA 后沿用 C1 提示。\n"
             "正式对照使用相同的 B 证据顺序、图片分辨率、贪心解码与输出限制。\n"
             "严格通过要求原始 JSON 合规、字段值与单位完全匹配、引用页集合正确。缺失题要求显式拒答。\n"
             "oracle-dev 单列正确页诊断。各组为小样本描述性比较。\n")
    (out / "三组对照与退化案例.txt").write_text(intro + "\n" + json.dumps(metrics, ensure_ascii=False, indent=2)
                                             + "\n\n逐题变化\n" + json.dumps(changes, ensure_ascii=False, indent=2), "utf-8")
    for row in metrics:
        if row["suite"] == "all":
            print(row["variant"], row["split"], row["strict_pass"], flush=True)

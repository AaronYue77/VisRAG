"""Controlled prompt versus visual LoRA SFT experiment, using the same B evidence."""
import argparse
from collections import defaultdict
import json
import math
import random
import re
import statistics
import time

import peft
import torch
import transformers

from c_protocol import DATA, load, norm, prepare
from common import WORKSPACE, file_hash, issue, operation, read_jsonl, write_json
from audit import append_question
from stage_a import MODEL_REVISION, PROMPT_VERSION, Reader, snapshot


PROMPT_C1 = (
    "\n核对步骤（无需输出过程）：先在图中找到问题指定的商品编号和字段名；"
    "逐字符抄写该字段的数值和原单位，别取标题中相似数字，也不要换算。"
    "两个商品都找到才逐项作答并引用两张证据页。只引用支持答案的图片。"
    "任一所需字段看不清或未报告，输出指定的 NOT_FOUND JSON。"
)


def compact_answer(answer):
    return norm(answer).replace("：", ":").replace("｜", "|")


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


def train(out, rows, cfg):
    torch.manual_seed(cfg["seed"])
    eligible = [row for row in rows if row["split"] == "train"]
    examples = [row for row in eligible if set(row["reference_pages"]) <=
                {page["page_id"] for page in row["retrieved_pages"]}]
    reader = Reader(cfg["max_tokens"])
    model = reader.vlm.model
    # Freeze the vision tower and early text layers. Fit query/value projections in the final two full-attention layers.
    query_layers = [name for name, module in model.named_modules()
                    if name.endswith(".self_attn.q_proj") and "language_model" in name]
    if len(query_layers) < cfg["attention_layers"]:
        raise RuntimeError("Base model does not have the expected attention projections")
    targets = [name.removesuffix("q_proj") + proj for name in query_layers[-cfg["attention_layers"]:]
               for proj in ("q_proj", "v_proj")]
    model = peft.get_peft_model(model, peft.LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=0, target_modules=targets, bias="none"))
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    if not params or any("lora_" not in name for name, param in model.named_parameters() if param.requires_grad):
        raise RuntimeError("Unexpected trainable parameter scope")
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    details = {"targets": targets, "trainable_parameters": sum(p.numel() for p in params),
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "training_questions": len(examples), "excluded_candidate_misses": sorted(
                   {r["question_id"] for r in eligible} - {r["question_id"] for r in examples}),
               "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
               "base_revision": MODEL_REVISION, "protocol_sha256": file_hash(DATA / "protocol.json"),
               "gpu": torch.cuda.get_device_name(), "precision": "BF16 base, FP32 LoRA"}
    write_json(out / "training-config.json", details)
    print("Training settings:", json.dumps(details), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rng = random.Random(cfg["seed"])
    training = []
    accumulation = cfg["gradient_accumulation"]
    for epoch in range(cfg["epochs"]):
        rng.shuffle(examples)
        for start in range(0, len(examples), accumulation):
            batch = examples[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for row in batch:
                inputs, targets_ids, positions, info = training_input(reader, row, out / "images", cfg)
                logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits
                loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets_ids.flatten())
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite loss at {row['question_id']}")
                (loss / len(batch)).backward()
                record = {"epoch": epoch + 1, "example": len(training) + 1, "question_id": row["question_id"],
                          "loss": value, "assistant_tokens": targets_ids.numel(),
                          "visual_tokens": info["visual_tokens_from_grid"]}
                training.append(record)
                with (out / "training-loss.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                del inputs, logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
            optimizer.step()
            if start == 0 or len(training) % 16 == 0:
                print(f"Epoch {epoch + 1} examples={len(training)} loss={value:.4f} grad_norm={grad_norm:.3f} "
                      f"peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    adapter = out / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    result = {**details, "examples_seen": len(training), "gpu_seconds": elapsed, "gpu_hours": elapsed / 3600,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "epoch_loss": [statistics.mean(r["loss"] for r in training if r["epoch"] == epoch + 1)
                             for epoch in range(cfg["epochs"])],
              "adapter_sha256": file_hash(adapter / "adapter_model.safetensors")}
    write_json(out / "training-summary.json", result)
    (out / "训练记录.txt").write_text("冻结底座与视觉塔。监督损失只计算回答及结束 token。\n" +
                                    json.dumps(result, ensure_ascii=False, indent=2) + "\n\n" +
                                    (out / "training-loss.jsonl").read_text("utf-8"), "utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def evaluate(out, rows, cfg, source_sha):
    adapter = out / "adapter"
    summary = json.loads((out / "training-summary.json").read_text("utf-8"))
    if (summary["adapter_sha256"] != file_hash(adapter / "adapter_model.safetensors")
            or summary["protocol_sha256"] != file_hash(DATA / "protocol.json")
            or summary["base_revision"] != MODEL_REVISION):
        raise ValueError("Adapter or frozen protocol identity differs from training")
    reader = Reader(cfg["max_tokens"])
    eval_rows = [row for row in rows if row["split"] != "train"]
    # A previous interrupted evaluation can resume with the exact same immutable config.
    result_file = out / "results.jsonl"
    prior = read_jsonl(result_file) if result_file.exists() else []
    completed = {(row["variant"], row["question_id"]): row for row in prior}
    config_sha = file_hash(out / "config.json")
    if len(prior) != len(completed) or any(row["config_sha256"] != config_sha or
                                         row["source_sha256"] != source_sha for row in prior):
        raise ValueError("Incompatible or duplicate previous evaluation records")

    def run_variant(variant, suite, prompt, oracle=False):
        for row in suite:
            key = (variant, row["question_id"])
            if key in completed:
                continue
            pages = row["gold_pages"] if oracle else row["retrieved_pages"]
            result = reader.read(row["question"], row["request"], pages, out / "images", cfg["max_pixels"], prompt)
            metrics = score(row, result, pages)
            record = {**{k: v for k, v in row.items() if k != "gold_pages"}, **result,
                      "run_name": out.name, "run_directory": str(out), "variant": variant,
                      "routing": {"reason": variant, "prompt": PROMPT_VERSION if variant == "C0" else "stage-c-evidence-v1"},
                      "retrieved_pages": pages, "retrieval_seconds": 0.0 if oracle else row["retrieval_seconds"],
                      "diagnostic_oracle": oracle, "status": "ok", "metrics": metrics,
                      "model": "Qwen/Qwen3.5-2B" + (" + LoRA" if variant == "C2" else ""),
                      "model_revision": MODEL_REVISION + (" + " + summary["adapter_sha256"] if variant == "C2" else ""),
                      "source_sha256": source_sha, "config_sha256": config_sha}
            with result_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            append_question(out, record)
            completed[key] = record
            count = len(completed)
            if count == 1 or count % 8 == 0:
                print(f"{variant} {count} evaluations recorded, latest={row['question_id']} strict={metrics['strict_pass']}", flush=True)

    run_variant("C0", eval_rows, "")
    run_variant("C1", eval_rows, PROMPT_C1)
    oracle_rows = [row for row in eval_rows if row["split"] == "dev" and row["suite"] == "unit"
                   and not completed[("C1", row["question_id"])]["metrics"]["strict_pass"]]
    run_variant("oracle-dev", oracle_rows, PROMPT_C1, oracle=True)
    reader.vlm.model = peft.PeftModel.from_pretrained(reader.vlm.model, adapter).eval()
    run_variant("C2", eval_rows, PROMPT_C1)
    report(out, completed)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--run-name", default="stage-c-lora-v1")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    rows, protocol = load()
    cfg = protocol["settings"]
    out = WORKSPACE / "reports/ecommerce_rag" / args.run_name
    started = time.perf_counter()
    try:
        if args.command == "train":
            out.mkdir(parents=True)
            source_sha = snapshot(out)
            write_json(out / "config.json", {"settings": cfg, "prompt_version": PROMPT_VERSION,
                      "prompt_addition": PROMPT_C1, "protocol_sha256": file_hash(DATA / "protocol.json"),
                      "source_sha256": source_sha, "base_revision": MODEL_REVISION})
            train(out, rows, cfg)
        else:
            config = json.loads((out / "config.json").read_text("utf-8"))
            if config["settings"] != cfg or config["prompt_addition"] != PROMPT_C1:
                raise ValueError("Evaluation settings differ from the frozen training configuration")
            evaluate(out, rows, cfg, config["source_sha256"])
    except Exception as error:
        key = issue(f"C阶段 {args.command} 失败", str(error), "保留当前运行目录和已有结果，按错误定位。")
        operation(f"C阶段 {args.command}", "按冻结协议执行独立训练或评测。", str(error), [out],
                  status="失败", related_issues=key)
        raise
    operation(f"C阶段 {args.command}", "按冻结ABO协议使用同一Qwen3.5-2B与B召回页执行。",
              f"完成，耗时 {time.perf_counter() - started:.1f} 秒", [out],
              "训练见训练记录与adapter哈希，评测见三组对照与逐题TXT。")


if __name__ == "__main__":
    main()

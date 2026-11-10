"""Larger fixed-evidence C evaluation, disjoint from all original C families."""
import argparse
from collections import defaultdict
import json
from pathlib import PureWindowsPath
import re
import time

import peft

from c_protocol import DATA, load as load_original
from audit import append_question
from common import WORKSPACE, file_hash, operation, read_jsonl, timestamp, write_json, write_jsonl
from full_cards import choose_pages, ensure_card
from full_catalog import connect
from full_evidence import FIELDS
from full_protocol import EVAL
from stage_a import MODEL_REVISION, Reader, snapshot
from stage_c import PROMPT_C1, report, score


EXTENDED = DATA.parent / "abo-stage-c-extended-v1"
ADAPTER_RUN = WORKSPACE / "reports/ecommerce_rag/stage-c-lora-20261109"
NUMERIC = {"height", "weight", "width", "length"}


def freeze():
    original, protocol = load_original()
    old_families = {family for row in original for family in row["family_keys"]}
    by_type = defaultdict(dict)
    for row in read_jsonl(EVAL / "questions.jsonl"):
        if row["expected_action"] != "answer" or row["field"] not in FIELDS:
            continue
        family = row["family_key"]
        if family in old_families:
            raise ValueError("C extended evaluation overlaps the original C family set")
        options = by_type[row["product_type"]].setdefault(family, [])
        options.append(row)
    if len(by_type) != 20 or any(len(families) != 4 for families in by_type.values()):
        raise ValueError("Expected the fixed A/B set of 20 categories with four families each")

    def make_row(group, suite, db):
        labels = [FIELDS[item["field"]][0] for item in group]
        skus = [item["request"]["skus"][0] for item in group]
        expected = [item["reference_answer"] for item in group]
        if suite == "two_pages":
            question = (f"核对两份规格卡。{skus[0]} 的{labels[0]}和 {skus[1]} 的{labels[1]}各是多少？"
                        "请按顺序写成 SKU: 值 | SKU: 值，保留资料原文与单位。")
            answer = " | ".join(f"{sku}: {value}" for sku, value in zip(skus, expected))
        elif suite == "missing":
            question = f"请查 {skus[0]} 的出厂序列号（serial number）。资料缺失时按协议说明。"
            answer = "NOT_FOUND"
        else:
            question = f"填写规格单时，{skus[0]} 的{labels[0]}一栏应该填什么？只给资料原值，保留单位。"
            answer = expected[0]
        started = time.perf_counter()
        pages = []
        for item in group:
            product = dict(db.execute("SELECT * FROM products WHERE listing_key=?",
                                      (item["listing_key"],)).fetchone())
            card = ensure_card(product, allow_download=False)
            if suite == "missing" and any("serial number" in p["text"].casefold() or "序列号" in p["text"]
                                          for p in card["pages"]):
                raise ValueError(f"Missing target unexpectedly present for {item['listing_key']}")
            pages.extend(choose_pages(card, question, 1 if len(group) == 2 else 2))
        for i, page in enumerate(pages, 1):
            page["rank"] = i
        ref_pages = [] if suite == "missing" else [p for item in group for p in item["reference_pages"]]
        domains = {item["request"]["domain"] for item in group}
        return {"question_id": "CX-" + "-".join(skus) + "-" + suite, "question": question,
                "request": {"skus": skus, "store_id": domains.pop() if len(domains) == 1 else "all_ABO_domains"},
                "reference_answer": answer, "reference_pages": ref_pages, "split": "test", "suite": suite,
                "expected_action": "not_found" if suite == "missing" else "answer", "retrieved_pages": pages,
                "retrieval_seconds": time.perf_counter() - started,
                "product_type": group[0]["product_type"], "family_keys": [row["family_key"] for row in group],
                "listing_keys": [row["listing_key"] for row in group]}

    rows = []
    db = connect()
    try:
        for families in by_type.values():
            items = [min(options, key=lambda row: (row["field"] not in NUMERIC, row["field"]))
                     for options in families.values()]
            rows.extend([make_row(items[:1], "unit", db), make_row(items[1:2], "unit", db),
                         make_row(items[2:], "two_pages", db), make_row(items[:1], "missing", db)])
    finally:
        db.close()
    if len(rows) != 80 or len({row["question_id"] for row in rows}) != 80:
        raise ValueError("Extended set must contain 80 unique cases")
    EXTENDED.mkdir(parents=True)
    write_jsonl(EXTENDED / "questions.jsonl", rows)
    write_json(EXTENDED / "protocol.json", {"settings": protocol["settings"], "questions": len(rows),
                "questions_sha256": file_hash(EXTENDED / "questions.jsonl"),
                "source": "A/B-v2 evaluated families, fresh C question wording, never used to fit C adapter",
                "original_c_protocol_sha256": file_hash(DATA / "protocol.json"),
                "selection": "20 types, four families per type; 40 single, 20 two-product, 20 missing"})


def load_extended():
    protocol = json.loads((EXTENDED / "protocol.json").read_text("utf-8"))
    if protocol["questions_sha256"] != file_hash(EXTENDED / "questions.jsonl"):
        raise ValueError("Extended frozen questions changed")
    rows = read_jsonl(EXTENDED / "questions.jsonl")
    for row in rows:
        for page in row["retrieved_pages"]:
            page["pdf"] = PureWindowsPath(page["pdf"]).as_posix()
    paths = {page["pdf"]: page["pdf_sha256"] for row in rows for page in row["retrieved_pages"]}
    if any(file_hash(WORKSPACE / path) != sha for path, sha in paths.items()):
        raise ValueError("Extended evaluation PDF changed")
    return rows, protocol


def run(name):
    if not (EXTENDED / "protocol.json").exists():
        freeze()
    rows, protocol = load_extended()
    adapter = ADAPTER_RUN / "adapter"
    training = json.loads((ADAPTER_RUN / "training-summary.json").read_text("utf-8"))
    adapter_sha = file_hash(adapter / "adapter_model.safetensors")
    if (training["base_revision"] != MODEL_REVISION or training["adapter_sha256"] != adapter_sha
            or training["protocol_sha256"] != protocol["original_c_protocol_sha256"]):
        raise ValueError("Adapter identity differs from the original frozen training run")
    out = WORKSPACE / "reports/ecommerce_rag" / name
    out.mkdir(parents=True)
    source_sha = snapshot(out)
    write_json(out / "config.json", {**protocol, "source_sha256": source_sha,
               "base_revision": MODEL_REVISION, "adapter_sha256": adapter_sha, "prompt_addition": PROMPT_C1})
    config_sha = file_hash(out / "config.json")
    cfg = protocol["settings"]
    reader = Reader(cfg["max_tokens"])
    completed = {}
    for variant, prompt in (("C0", ""), ("C1", PROMPT_C1), ("C2", PROMPT_C1)):
        if variant == "C2":
            reader.vlm.model = peft.PeftModel.from_pretrained(reader.vlm.model, adapter).eval()
        for i, row in enumerate(rows, 1):
            pages = row["retrieved_pages"]
            result = reader.read(row["question"], row["request"], pages, out / "images", cfg["max_pixels"], prompt)
            record = {**row, **result, "metrics": score(row, result, pages), "variant": variant,
                      "timestamp": timestamp(), "routing": {"reason": variant, "max_pixels": cfg["max_pixels"]},
                      "status": "ok", "run_name": name, "run_directory": str(out), "source_sha256": source_sha,
                      "config_sha256": config_sha, "model": "Qwen/Qwen3.5-2B" + (" + LoRA" if variant == "C2" else ""),
                      "model_revision": MODEL_REVISION + (" + " + adapter_sha if variant == "C2" else "")}
            completed[(variant, row["question_id"])] = record
            with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            append_question(out, record)
            if i % 10 == 0:
                print(f"{variant} {i}/{len(rows)}", flush=True)
    report(out, completed)
    operation("扩展 C 阶段固定证据评测", "20 类 80 个家族，冻结后固定候选、像素、解码和模型，复用原 C 适配器做三组逐题对照。",
              f"完成 {len(completed)} 条逐题记录", [out], "扩展题来自既有 A/B 评测家族，与原 C 全部家族互斥。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    run(args.run_name)

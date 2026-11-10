"""Replay the same frozen user turns through baseline and new evidence paths."""
import argparse
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata

from full_app import FullApplication
from audit import append_question
from common import WORKSPACE, file_hash, operation, read_jsonl, write_json
from full_cards import asset_key
from full_catalog import ROOT


PROTOCOL = WORKSPACE / "data/ecommerce_rag/adaptive-evaluation-v1/turns.json"


def normalize(text):
    return re.sub(r"[^\w.]", "", unicodedata.normalize("NFKC", text).casefold())


def score(case, row, response):
    answer = normalize(row["answer"])
    citations = set(row["citations"])
    facts = case.get("facts", [])
    passed = row["status"] == "ok" and row["action"] == case["action"]
    passed = passed and all(normalize(term) in answer for term in case.get("terms", []))
    if case.get("terms_any"):
        passed = passed and any(normalize(term) in answer for term in case["terms_any"])
    if case.get("candidate_type"):
        kinds = {item["product_type"] for item in response["candidates"]}
        passed = passed and kinds == {case["candidate_type"]}
    if case["action"] == "chat":
        passed = passed and bool(answer) and not citations and not row["retrieved_pages"]
    if case["action"] in {"not_found", "clarify", "requires_tool"}:
        passed = passed and not citations
    if case.get("photo"):
        passed = passed and case["photo"] in citations
    if case.get("policy"):
        passed = passed and case["policy"] in citations
    for key, field, value in facts:
        has_value = normalize(value) in answer
        supported = f"{key}:metadata:{field}" in citations
        if not supported:
            path = ROOT / "cards" / asset_key({"listing_key": key}) / "document.json"
            if path.exists():
                doc = json.loads(path.read_text("utf-8"))
                supported = any(page["page_id"] in citations and
                                normalize(f"{field}: {value}") in normalize(page["text"])
                                for page in doc["pages"])
        if len({item[0] for item in facts}) > 1:
            sku = key.split("@")[0]
            has_value = bool(re.search(re.escape(normalize(sku)) +
                                       r"(?:(?!b[0-9][a-z0-9]{8}).)*?" + re.escape(normalize(value)), answer))
        passed = passed and has_value and supported
    return {"pass": bool(passed), "expected_action": case["action"], "group": case["group"],
            "measurement": "route and bounded field/photo/section checks; chat content is not scored"}


def run(name, protocol):
    cases = json.loads(protocol.read_text("utf-8"))
    out = WORKSPACE / "reports/ecommerce_rag" / name
    out.mkdir(parents=True)
    write_json(out / "config.json", {"protocol": str(protocol.relative_to(WORKSPACE)),
                                    "protocol_sha256": file_hash(protocol),
                                    "variants": ["legacy", "adaptive"], "mode": "bm25", "stage": "B",
                                    "retry": False, "visual_rerank": False,
                                    "order": "alternate variants every turn; common CUDA reader"})
    config_sha = file_hash(out / "config.json")
    app = FullApplication("B", mode="bm25", report_root=out, audit=False, legacy_evidence=True, model="2b")
    sessions = {}
    records = []
    try:
        for index, case in enumerate(cases):
            variants = ("legacy", "adaptive") if index % 2 == 0 else ("adaptive", "legacy")
            for variant in variants:
                app.legacy_evidence = variant == "legacy"
                if variant not in sessions:
                    sessions[variant] = app.session()
                session = sessions[variant]
                response = session.respond(case["question"], case.get("request"))
                row = read_jsonl(session.out / "results.jsonl")[-1]
                refs = [f"{key}:metadata:{field}" for key, field, _ in case.get("facts", [])]
                refs += [case[key] for key in ("photo", "policy") if key in case]
                metrics = score(case, row, response)
                record = {**row, "variant": variant, "case_id": case["id"], "run_name": name,
                          "config_sha256": config_sha, "session_config_sha256": row["config_sha256"],
                          "session_run_directory": row["run_directory"],
                          "run_directory": str(out),
                          "reference_answer": json.dumps({k: case[k] for k in ("action", "facts", "terms", "terms_any", "candidate_type") if k in case}, ensure_ascii=False),
                          "reference_pages": refs, "metrics": metrics}
                with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                append_question(out, record)
                records.append(record)
                print(case["id"], variant, row["action"], metrics["pass"], flush=True)
    finally:
        app.close()
    summary = []
    for group in sorted({case["group"] for case in cases}):
        for variant in ("legacy", "adaptive"):
            items = [r for r in records if r["metrics"]["group"] == group and r["variant"] == variant]
            summary.append({"group": group, "variant": variant, "n": len(items),
                            "passed": sum(r["metrics"]["pass"] for r in items),
                            "median_seconds": statistics.median(r["total_seconds"] for r in items),
                            "errors": sum(r["status"] != "ok" for r in items)})
    write_json(out / "summary.json", summary)
    paired = {}
    for record in records:
        paired.setdefault(record["case_id"], {})[record["variant"]] = record
    changes = [{"case": key, **{variant: {"pass": result["metrics"]["pass"], "answer": result["answer"]}
                               for variant, result in pair.items()}}
               for key, pair in paired.items() if pair["legacy"]["metrics"]["pass"] != pair["adaptive"]["metrics"]["pass"]]
    (out / "同题对照.txt").write_text(
        f"{len(cases)} 道冻结的本地功能样例。字段值与图片标注在运行前核对。"
        "新旧使用同一全量 BM25 和本地 Qwen，差异是证据处理策略。"
        "政策是新增覆盖范围，分开报告。普通话题只计路由。样本小，耗时为当前本机实测。\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2) + "\n\n配对变化\n"
        + json.dumps(changes, ensure_ascii=False, indent=2), "utf-8")
    operation("全量入口自适应证据同题对照", "冻结自然问法，交替运行旧证据链与新证据链，逐条检查答案和证据。",
              f"完成 {len(records)} 条逐题记录", [out], "查看同题对照和原始 JSONL。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    run(args.run_name, args.protocol.resolve())

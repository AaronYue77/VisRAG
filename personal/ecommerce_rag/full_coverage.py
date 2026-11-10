"""Audit every ABO listing on CPU and save one annotated example per category."""
import argparse
from collections import Counter
import json
import re
import time

from full_catalog import ROOT, connect
from audit import append_question
from common import WORKSPACE, file_hash, operation, timestamp, write_json
from full_evidence import FIELDS, VERSION, read_fields


def source_values(raw, field):
    if field in {"width", "height", "length", "weight"}:
        entries = raw.get("item_weight", []) if field == "weight" else [raw.get("item_dimensions", {}).get(field)]
        values = [f"{entry['value']} {entry.get('unit', '')}".strip() for entry in entries
                  if isinstance(entry, dict) and "value" in entry]
    else:
        entries = [entry for entry in raw.get(field, [])
                   if isinstance(entry, dict) and entry.get("value") is not None]
        english = [entry for entry in entries if str(entry.get("language_tag", "")).startswith("en")]
        chinese = [entry for entry in entries if entry.get("language_tag") == "zh_CN"]
        values = [str(entry["value"]).strip() for entry in english or chinese or entries[:1]]
    return list(dict.fromkeys(value for value in values if value))


def run(name):
    out = WORKSPACE / "reports/ecommerce_rag" / name
    out.mkdir(parents=True)
    import full_evidence
    from pathlib import Path

    source_sha = file_hash(Path(full_evidence.__file__))
    manifest = json.loads((ROOT / "manifest.json").read_text("utf-8"))
    write_json(out / "config.json", {"evidence_version": VERSION, "source_sha256": source_sha,
                                    "catalog_sha256": manifest["database_sha256"], "fields": list(FIELDS)})
    config_sha = file_hash(out / "config.json")
    types, sites, coverage = Counter(), Counter(), Counter()
    sampled = {}
    errors = []
    started = time.perf_counter()
    db = connect()
    try:
        fts_count = db.execute("SELECT count(*) FROM products_fts").fetchone()[0]
        for row in db.execute("SELECT * FROM products ORDER BY id"):
            product = dict(row)
            kind, site, key = product["product_type"], product["domain"], product["listing_key"]
            types[kind] += 1
            sites[site] += 1
            try:
                raw = json.loads(product["raw_json"])
                expected = {field: source_values(raw, field) for field in FIELDS}
                result = read_fields([product], tuple(FIELDS))
                received = {claim["field"]: claim["values"] for claim in result["claims"]}
                populated = {field: values for field, values in expected.items() if values}
                if key != f"{product['sku']}@{site}" or received != populated:
                    raise ValueError("identity or source values differ")
                for claim, page in zip(result["claims"], result["retrieved_pages"], strict=True):
                    if (claim["listing_key"] != key or page["listing_key"] != key or
                            claim["citation"] != f"{key}:metadata:{claim['field']}" or
                            page["page_id"] != claim["citation"] or
                            claim["source"] != {"member": product["source_member"], "line": product["source_line"],
                                                "sha256": product["raw_sha256"]}):
                        raise ValueError("source lineage differs")
                coverage.update(populated.keys())
                choices = [field for field in ("weight", "height", "width", "length", *FIELDS) if field in populated]
                # Prefer a real answered field when a category has one.
                if kind not in sampled or (choices and not sampled[kind][2]):
                    field = choices[0] if choices else "weight"
                    sampled[kind] = (product, field, expected[field])
            except Exception as error:
                errors.append({"listing_key": key, "error": str(error)})
    finally:
        db.close()
    for i, (kind, (product, field, expected)) in enumerate(sorted(sampled.items()), 1):
        stamp = time.perf_counter()
        result = read_fields([product], (field,))
        key = product["listing_key"]
        row = {**result, "question_id": f"COV-{i:04d}", "question": f"{key} 的{FIELDS[field][0]}是什么？",
               "request": {"skus": [product["sku"]], "domain": product["domain"], "product_type": kind},
               "reference_answer": " / ".join(expected) if expected else "源字段缺失",
               "reference_pages": [f"{key}:metadata:{field}"] if expected else [],
               "retrieval_seconds": time.perf_counter() - stamp, "generation_seconds": 0.0,
               "status": "ok", "model": None, "model_revision": None, "timestamp": timestamp(),
               "config_sha256": config_sha, "source_sha256": source_sha, "run_name": name,
               "run_directory": str(out), "routing": {"strategy": "source_fields_cpu_audit"},
               "metrics": {"pass": result["action"] == ("answer" if expected else "not_found") and
                           (result["claims"][0]["values"] if result["claims"] else []) == expected}}
        with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        append_question(out, row)
    summary = {"listings": sum(types.values()), "fts_rows": fts_count, "types": len(types), "sites": len(sites),
               "sampled_questions": len(sampled), "errors": len(errors), "field_availability": dict(coverage),
               "category_counts": dict(types), "site_counts": dict(sites), "seconds": time.perf_counter() - started}
    write_json(out / "summary.json", summary)
    write_json(out / "errors.json", errors)
    (out / "全库覆盖核查.txt").write_text(
        "逐条对照原始结构化值、商品站点身份与引用来源，另按类目输出一题。"
        "仅核查结构化取值链覆盖，图像和自由生成另用抽样评测。\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print({key: value for key, value in summary.items() if not key.endswith("_counts")}, flush=True)
    if errors or summary["listings"] != manifest["records"] or fts_count != manifest["records"]:
        raise ValueError(f"Coverage mismatch. See {out / 'errors.json'}")
    operation("全量商品结构化证据核查", "CPU 逐条核对原始字段、站点与来源标识，按类目输出独立逐题记录。",
              f"{summary['listings']} 条记录，{len(types)} 类，{len(sites)} 站点，{len(errors)} 错误",
              [out], "见全库覆盖核查、逐题结果和原始 JSONL。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    run(args.run_name)

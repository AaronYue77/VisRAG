"""Evaluate 120 new catalogue queries and export top-five relevance pools."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata

from c8_protocol import DATA, load
from audit import append_question
from common import WORKSPACE, file_hash, timestamp, write_json, write_jsonl
from full_catalog import field
from full_retrieval import EMBED_REVISION, FullRetriever


MODES = ("bm25", "dense", "hybrid")
QUERIES = DATA / "retrieval-queries.jsonl"


def compact(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def model_number(product):
    return field(json.loads(product["raw_json"]), "model_number")


def freeze(retriever):
    cases, _ = load()
    selected, labels = set(), set()
    queries = []
    for case in (row for row in cases if row["split"] == "test"):
        for listing in case["listing_keys"]:
            sku, site = listing.split("@", 1)
            product, = retriever.get(sku, site)
            model = model_number(product)
            brand = product["brand"]
            if (product["family_key"] in selected or len(compact(model)) < 4 or not brand
                    or len(model) > 45 or not model.isascii()):
                continue
            index = len(queries)
            variant = index % 3
            if variant == 0:
                question = f"{brand} {model}"
                filters = {}
            elif variant == 1:
                question = f"{brand} {model} {product['product_type'].replace('_', ' ').lower()}"
                filters = {"product_type": product["product_type"]}
            else:
                question = f"{brand} {model}"
                filters = {"domain": site}
            if (question, tuple(filters.items())) in labels:
                continue
            labels.add((question, tuple(filters.items())))
            selected.add(product["family_key"])
            queries.append({"question_id": f"R8-{len(queries) + 1:03d}", "question": question,
                            "filters": filters, "gold": listing, "brand": brand, "model_number": model,
                            "family_key": product["family_key"], "product_type": product["product_type"]})
            if len(queries) == 120:
                write_jsonl(QUERIES, queries)
                write_json(DATA / "retrieval-protocol.json", {"sha256": file_hash(QUERIES), "n": len(queries),
                    "modes": MODES, "top_k": 5, "relevance": "exact source brand and model, within explicit scope"})
                return
    raise ValueError("Insufficient distinct model-number queries")


def run(name):
    retriever = FullRetriever()
    try:
        if not QUERIES.is_file():
            freeze(retriever)
        protocol = json.loads((DATA / "retrieval-protocol.json").read_text("utf-8"))
        if protocol["sha256"] != file_hash(QUERIES):
            raise ValueError("Frozen retrieval queries changed")
        from common import read_jsonl

        queries = read_jsonl(QUERIES)
        out = WORKSPACE / "reports/ecommerce_rag" / name
        out.mkdir(parents=True)
        source_sha = file_hash(Path(__file__))
        write_json(out / "config.json", {**protocol, "index_records": retriever.manifest["records"],
                   "source_sha256": source_sha, "catalog_sha256": retriever.manifest["database_sha256"]})
        config_sha = file_hash(out / "config.json")
        summaries = {mode: [] for mode in MODES}
        pools = []
        for number, query in enumerate(queries, 1):
            found = {mode: retriever.search(query["question"], mode, 5, query["filters"]) for mode in MODES}
            candidates = {item["listing_key"]: item for result in found.values() for item in result["items"]}
            pool = []
            for key, item in candidates.items():
                actual = model_number(item)
                exact = compact(actual) == compact(query["model_number"]) and compact(item["brand"]) == compact(query["brand"])
                pool.append({"listing_key": key, "title": item["title"], "brand": item["brand"], "model_number": actual,
                             "same_family": item["family_key"] == query["family_key"], "source_relevant": exact,
                             "ranks": {mode: next((r["rank"] for r in found[mode]["items"] if r["listing_key"] == key), None)
                                       for mode in MODES}})
            relevant = {r["listing_key"] for r in pool if r["source_relevant"]} | {query["gold"]}
            pools.append({**query, "pool": pool})
            for mode, result in found.items():
                pages = [{**{k: item[k] for k in ("listing_key", "rank", "score", "sku", "title")},
                          "page_id": item["listing_key"] + ":metadata", "text_source": "ABO_catalogue_metadata"}
                         for item in result["items"]]
                ranks = [p["rank"] for p in pages if p["listing_key"] in relevant]
                metrics = {"exact_hit_at_5": any(p["listing_key"] == query["gold"] for p in pages),
                           "source_hit_at_5": bool(ranks), "pool_recall_at_5": len(ranks) / len(relevant),
                           "mrr_at_5": 1 / min(ranks) if ranks else 0}
                record = {"run_name": name, "run_directory": str(out), "question_id": query["question_id"],
                          "question": query["question"], "request": query["filters"], "variant": mode,
                          "reference_answer": f"{query['brand']} {query['model_number']}",
                          "reference_pages": [key + ":metadata" for key in sorted(relevant)],
                          "retrieved_pages": pages, "answer": "仅检索，未调用问答模型", "citations": [],
                          "generation_seconds": 0, "retrieval_seconds": result["seconds"],
                          "model": "intfloat/multilingual-e5-small" if mode != "bm25" else None,
                          "model_revision": EMBED_REVISION if mode != "bm25" else None,
                          "config_sha256": config_sha, "source_sha256": source_sha,
                          "timestamp": timestamp(), "status": "ok", "metrics": metrics}
                with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                append_question(out, record)
                summaries[mode].append({**metrics, "seconds": result["seconds"]})
            if number % 20 == 0:
                print(f"Retrieval {number}/{len(queries)}", flush=True)
        write_jsonl(out / "pooled-candidates.jsonl", pools)
        summary = [{"mode": mode, "n": len(results), "exact_hits": sum(r["exact_hit_at_5"] for r in results),
                    "source_hits": sum(r["source_hit_at_5"] for r in results),
                    "mean_pool_recall": statistics.mean(r["pool_recall_at_5"] for r in results),
                    "mrr": statistics.mean(r["mrr_at_5"] for r in results),
                    "p50_seconds": statistics.median(r["seconds"] for r in results)}
                   for mode, results in summaries.items()]
        write_json(out / "summary.json", summary)
        (out / "检索对照.txt").write_text(
            "120 条型号检索，每题合并三路 Top-5。相关性以源品牌和型号、显式范围为依据。"
            "池内 Recall 的分母是池中可确认相关和已知目标，不是全库所有相关条目。"
            "此为可核验规则标注，开放语义相关性仍需人工复核。\n"
            + json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    finally:
        retriever.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    run(args.run_name)

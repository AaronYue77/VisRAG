"""Freeze a small, family-disjoint ABO experiment and B page candidates."""
from collections import Counter
from hashlib import sha256
import json
import re
import time
from pathlib import PureWindowsPath

from full_catalog import ROOT, connect
from full_cards import choose_pages, ensure_card, evidence_fields
from full_protocol import EVAL
from common import WORKSPACE, file_hash, operation, read_jsonl, write_json, write_jsonl


DATA = ROOT.parent / "abo-stage-c-v1"
TYPES = ("TABLE", "CHAIR", "SOFA", "BED", "HEADPHONES", "BACKPACK", "KITCHEN", "PET_SUPPLIES")
FIELDS = {"width": "宽度", "weight": "重量", "height": "高度", "length": "长度"}
CONFIG = {
    "version": "abo-stage-c-v1",
    "max_pixels": 460800,
    "max_tokens": 96,
    "seed": 9026,
    "epochs": 2,
    "learning_rate": 0.0002,
    "lora_r": 8,
    "lora_alpha": 16,
    "attention_layers": 2,
    "batch_size": 1,
    "gradient_accumulation": 2,
}


def norm(text):
    return re.sub(r"\s+", "", text).casefold()


def source_fields(product):
    grouped = {}
    for key, value in evidence_fields(json.loads(product["raw_json"])):
        grouped.setdefault(key, []).append(value.strip())
    return grouped


def select_products(db):
    old_skus = set()
    for corpus in ("abo-cards-v1", "abo-dialogue-v2"):
        raw = (ROOT.parent / corpus / "selected-products.json").read_text("utf-8")
        old_skus.update(re.findall(r'"((?:B[0-9][A-Z0-9]{8}|[0-9]{9}[0-9X]))"', raw))
    used = {item["family_key"] for item in json.loads((EVAL / "selected.json").read_text("utf-8"))}
    for sku in old_skus:
        used.update(row[0] for row in db.execute("SELECT family_key FROM products WHERE sku=?", (sku,)))
    selected = []
    for kind in TYPES:
        candidates = db.execute("SELECT id,listing_key,family_key FROM products WHERE product_type=? AND domain=?",
                                (kind, "amazon.com")).fetchall()
        candidates = sorted(candidates, key=lambda row: sha256(("stage-c-v1:" + row["listing_key"]).encode()).digest())
        family_count = 0
        for candidate in candidates:
            if candidate["family_key"] in used:
                continue
            product = dict(db.execute("SELECT * FROM products WHERE id=?", (candidate["id"],)).fetchone())
            fields = source_fields(product)
            numeric = [key for key in FIELDS if len(fields.get(key, [])) == 1 and
                       re.fullmatch(r"\d+(?:\.\d+)? [A-Za-z]+", fields[key][0])]
            if not numeric:
                continue
            key = numeric[family_count % len(numeric)]
            doc = ensure_card(product, allow_download=False)
            if not any(norm(key + ":" + fields[key][0]) in norm(page["text"]) for page in doc["pages"]):
                continue
            split = ("train",) * 4 + ("dev", "test", "test")
            selected.append((product, doc, key, fields[key][0], split[family_count]))
            used.add(product["family_key"])
            family_count += 1
            if family_count == len(split):
                break
        if family_count != 7:
            raise RuntimeError(f"Insufficient eligible families in {kind}")
    return selected


def question_row(items, question, facts, suite):
    started = time.perf_counter()
    products = [item[0] for item in items]
    pages = []
    for product, doc, *_ in items:
        pages.extend(choose_pages(doc, question, budget=1 if len(items) == 2 else 2))
    page_keys = ("page_id", "doc_id", "page", "sku", "score", "text_source", "listing_key",
                 "version", "pdf", "pdf_sha256")
    pages = [{**{key: page[key] for key in page_keys}, "rank": rank}
             for rank, page in enumerate(pages, 1)]
    gold = []
    for (_, doc, field, value, _), _ in zip(items, facts):
        page = next(page for page in doc["pages"] if norm(field + ":" + value) in norm(page["text"]))
        gold.append({**{key: (doc["version"] if key == "version" else page[key]) for key in page_keys}, "rank": len(gold) + 1})
    skus = [product["sku"] for product in products]
    answer = "NOT_FOUND" if not facts else facts[0] if len(facts) == 1 else " | ".join(
        f"{sku}: {value}" for sku, value in zip(skus, facts))
    return {
        "question_id": "C-" + "-".join(skus) + "-" + suite,
        "question": question, "reference_answer": answer,
        "reference_pages": [page["page_id"] for page in gold], "gold_pages": gold,
        "expected_action": "answer" if facts else "not_found",
        "retrieved_pages": pages, "retrieval_seconds": time.perf_counter() - started,
        "request": {"skus": skus, "store_id": "amazon.com"}, "split": items[0][4],
        "family_keys": [p["family_key"] for p in products], "listing_keys": [p["listing_key"] for p in products],
        "source_lines": [{"member": p["source_member"], "line": p["source_line"], "sha256": p["raw_sha256"]}
                         for p in products],
        "product_type": products[0]["product_type"], "suite": suite,
    }


def prepare():
    if (DATA / "protocol.json").exists():
        questions, protocol = load()
        print(f"Verified existing frozen protocol: {len(questions)} questions")
        return
    with connect() as db:
        selected = select_products(db)
    rows = []
    pairs = {}
    for item in selected:
        product, _, field, value, split = item
        sku = product["sku"]
        label = FIELDS[field]
        question = {"train": f"{sku} 的{label}（{field}）是多少？抄写原始数值与单位。",
                    "dev": f"请按卡片原值给出 {sku} 的{label} {field}，保留单位。",
                    "test": f"规格对照时 {sku} 的{label}（{field}）应填什么数值和原单位？"}[split]
        rows.append(question_row([item], question, [value], "unit"))
        rows.append(question_row([item],
            f"请给出 {sku} 的出厂序列号（serial number），只有资料明确报告这个字段才回答。", [], "missing"))
        pairs.setdefault((product["product_type"], split), []).append(item)
    for (_, split), group in pairs.items():
        for first, second in zip(group[::2], group[1::2]):
            # B obtains one page per product for two-SKU comparisons.
            skus = [first[0]["sku"], second[0]["sku"]]
            question = (f"逐个读取 {skus[0]} 的{FIELDS[first[2]]}（{first[2]}）与 "
                        f"{skus[1]} 的{FIELDS[second[2]]}（{second[2]}）。"
                        "按提问顺序用 SKU: 值 | SKU: 值 表示，保留原单位。")
            rows.append(question_row([first, second], question, [first[3], second[3]], "two_pages"))
    families = {split: sorted({key for row in rows if row["split"] == split for key in row["family_keys"]})
                for split in ("train", "dev", "test")}
    assert sum(map(len, families.values())) == len(set().union(*families.values())) == 56
    assert len(rows) == len({row["question_id"] for row in rows})
    for row in rows:
        if row["suite"] == "missing":
            assert all("serial number" not in page["text"].casefold() and "序列号" not in page["text"]
                       for item in selected if item[0]["listing_key"] in row["listing_keys"] for page in item[1]["pages"])
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / "questions.jsonl", rows)
    protocol = {"settings": CONFIG, "families": families, "questions": dict(Counter(row["split"] for row in rows)),
                "questions_sha256": file_hash(DATA / "questions.jsonl"), "source": "Amazon Berkeley Objects",
                "retrieval": "B exact ASIN@domain and its within-product page ranker, candidate order frozen",
                "annotation": "Source-derived structured fields and missing serial number, eight sampled categories"}
    write_json(DATA / "protocol.json", protocol)
    operation("C阶段冻结家族互斥的ABO证据问答", "按8类选取原始数值字段，排除已有A/B评测商品家族，固定B页排序、PDF哈希与来源行号。",
              str(protocol["questions"]), [DATA / "questions.jsonl", DATA / "protocol.json"],
              "56个独立家族，三集合互斥。标签来自源字段，图片是同一派生PDF渲染。")
    print(json.dumps(protocol["questions"]), flush=True)


def load():
    protocol = json.loads((DATA / "protocol.json").read_text("utf-8"))
    if file_hash(DATA / "questions.jsonl") != protocol["questions_sha256"]:
        raise ValueError("Frozen C question or candidate data changed")
    rows = read_jsonl(DATA / "questions.jsonl")
    # The frozen protocol was authored on Windows. Interpret its relative paths
    # portably without rewriting the JSONL or changing any file hashes.
    for row in rows:
        for page in row["retrieved_pages"] + row["gold_pages"]:
            page["pdf"] = PureWindowsPath(page["pdf"]).as_posix()
    checks = {page["pdf"]: page["pdf_sha256"]
              for row in rows for page in row["retrieved_pages"] + row["gold_pages"]}
    for path, expected in checks.items():
        if file_hash(WORKSPACE / path) != expected:
            raise ValueError(f"Frozen C evidence changed: {path}")
    return rows, protocol


if __name__ == "__main__":
    prepare()

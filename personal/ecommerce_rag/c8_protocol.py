"""Freeze new family-isolated 8B cases with source-backed, derived specification tables."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import PureWindowsPath
import re

from full_catalog import ROOT, connect
from common import WORKSPACE, file_hash, read_jsonl, write_json, write_jsonl
from full_cards import evidence_fields


DATA = ROOT.parent / "abo-8b-eval-v1"
GROUPS = ("single", "double", "table", "evidence", "missing")
SPLITS = {"train": (12, 12, 12, 12, 12), "dev": (5, 5, 5, 5, 5), "test": (40, 45, 35, 30, 30)}
FIELDS = {"brand": "品牌", "color": "颜色", "model_number": "型号", "material": "材质",
          "width": "宽度", "height": "高度", "length": "长度", "weight": "重量"}
NUMERIC = ("weight", "height", "width", "length")
VERSION = "abo-8b-eval-v1"


def historical_families():
    families = set()
    for folder in ("abo-stage-c-v1", "abo-stage-c-extended-v1"):
        for row in read_jsonl(ROOT.parent / folder / "questions.jsonl"):
            families.update(row["family_keys"])
    for folder in ("abo-full-evaluation-v1", "abo-full-evaluation-v2"):
        file = ROOT.parent / folder / "selected.json"
        if file.is_file():
            families.update(row["family_key"] for row in json.loads(file.read_text("utf-8")))
    for folder in ("abo-cards-v1", "abo-dialogue-v2"):
        file = ROOT.parent / folder / "selected-products.json"
        if file.is_file():
            skus = set(re.findall(r'"((?:B[0-9][A-Z0-9]{8}|[0-9]{9}[0-9X]))"', file.read_text("utf-8")))
            db = connect()
            try:
                for sku in skus:
                    families.update(row[0] for row in db.execute("SELECT family_key FROM products WHERE sku=?", (sku,)))
            finally:
                db.close()
    return families


def choose_products(count, excluded, version=VERSION, category_limit=36):
    db = connect()
    try:
        types = [row[0] for row in db.execute("SELECT product_type FROM products GROUP BY product_type "
                                             "HAVING count(*)>=150 ORDER BY count(*) DESC, product_type LIMIT ?",
                                             (category_limit,))]
        choices = defaultdict(list)
        for kind in types:
            rows = db.execute("SELECT * FROM products WHERE product_type=? AND "
                              "json_type(raw_json,'$.item_dimensions')='object' ORDER BY id", (kind,))
            candidates = sorted(rows, key=lambda row: sha256((version + row["listing_key"]).encode()).digest())
            for row in candidates:
                if row["family_key"] in excluded:
                    continue
                fields = defaultdict(set)
                for key, value in evidence_fields(json.loads(row["raw_json"])):
                    value = value.strip()
                    if key in FIELDS and value and value.isascii() and len(value) < 55:
                        fields[key].add(value)
                fixed = {key: next(iter(values)) for key, values in fields.items() if len(values) == 1}
                numeric = {key: fixed[key] for key in NUMERIC if key in fixed and
                           re.fullmatch(r"\d+(?:\.\d+)? [A-Za-z]+", fixed[key])}
                if not numeric or "brand" not in fixed or "model_number" not in fixed:
                    continue
                if len(choices[kind]) >= 24:
                    break
                choices[kind].append((dict(row), fixed, numeric))
                excluded.add(row["family_key"])
        pool = [items[i] for i in range(24) for items in choices.values() if i < len(items)]
        if len(pool) < count:
            raise ValueError(f"Only {len(pool)} eligible disjoint product families")
        return iter(pool[:count])
    finally:
        db.close()


def make_card(product, fields, data=DATA, version=VERSION, chart_field=None):
    from reportlab.pdfgen.canvas import Canvas

    path = data / "cards" / (sha256(product["listing_key"].encode()).hexdigest()[:24] + ".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=(595, 595), invariant=1)
    doc = path.stem
    for page, group in enumerate((tuple(k for k in FIELDS if k not in NUMERIC), NUMERIC), 1):
        canvas.setFont("Helvetica-Bold", 21)
        canvas.drawString(45, 540, "ABO product evidence card")
        canvas.setFont("Helvetica", 12)
        canvas.drawString(45, 512, product["listing_key"])
        canvas.drawString(45, 494, "Category: " + product["product_type"])
        if page == 2 and chart_field:
            value, unit = fields[chart_field].split()
            canvas.setFont("Helvetica-Bold", 16)
            canvas.drawString(65, 427, "Reported " + chart_field)
            canvas.setFillColorRGB(0.16, 0.38, 0.63)
            end = 85 + (300 if float(value) else 0)
            canvas.rect(85, 285, end - 85, 34, stroke=0, fill=1)
            canvas.setFillColorRGB(0, 0, 0)
            canvas.setFont("Helvetica", 14)
            canvas.drawString(end + 10, 296, value)
            canvas.line(85, 280, 465, 280)
            canvas.drawString(80, 260, "0")
            canvas.drawString(85, 223, "Unit: " + unit)
        else:
            canvas.setFont("Helvetica-Bold", 12)
            canvas.drawString(65, 449, "Field")
            canvas.drawString(225, 449, "Original value and unit")
            y = 425
            for field in group:
                if field not in fields:
                    continue
                canvas.line(55, y - 10, 535, y - 10)
                canvas.setFont("Helvetica", 13)
                canvas.drawString(65, y, field)
                canvas.drawString(225, y, fields[field])
                y -= 38
        canvas.setFont("Helvetica", 9)
        layout = "chart" if page == 2 and chart_field else "table"
        canvas.drawString(45, 55, f"ABO derived {layout}, CC BY 4.0 | " + product["source_member"] + " | " + str(product["source_line"]))
        canvas.showPage()
    canvas.save()
    return [{"pdf": path.relative_to(WORKSPACE).as_posix(), "pdf_sha256": file_hash(path), "doc_id": doc,
             "page": i, "page_id": f"{doc}:p{i:04d}", "version": version, "sku": product["sku"],
             "listing_key": product["listing_key"], "domain": product["domain"],
             "text_source": "ABO_source_derived_chart" if i == 2 and chart_field else "ABO_source_derived_table",
             "score": 1.0} for i in (1, 2)]


def freeze(data=DATA, version=VERSION, splits=SPLITS, excluded=None, charts=False, id_prefix=None, category_limit=36):
    if data.exists():
        raise FileExistsError(f"Frozen directory already exists: {data}")
    id_prefix = id_prefix or ("C8" if version == VERSION else "C8S")
    rows = []
    need = sum(sum(group) + group[1] + group[3] // 2 for group in splits.values())
    products = choose_products(need, historical_families() | set(excluded or ()), version, category_limit)
    for split, counts_by_group in splits.items():
        for suite, quantity in zip(GROUPS, counts_by_group):
            for index in range(quantity):
                first = next(products)
                group = [first]
                if suite == "double" or (suite == "evidence" and index < quantity // 2):
                    group.append(next(products))
                ids = [p["listing_key"] for p, _, _ in group]
                skus = [p["sku"] for p, _, _ in group]
                one, attrs, numeric = first
                key = tuple(numeric)[index % len(numeric)]
                field = FIELDS[key]
                chart = key if charts and suite == "table" else None
                cards = [make_card(p, f, data, version, chart) for p, f, _ in group]
                pages = cards[0]
                if suite == "single":
                    keys = tuple(k for k in ("brand", "color", "model_number", "material") if k in attrs)
                    key = keys[index % len(keys)]
                    question = f"资料登记 {ids[0]} 的{FIELDS[key]}是什么？只抄原值。"
                    answer, gold = attrs[key], [cards[0][0]]
                elif suite == "double":
                    other = group[1][2]
                    key2 = tuple(other)[index % len(other)]
                    question = (f"核对 {ids[0]} 的{field}和 {ids[1]} 的{FIELDS[key2]}。"
                                "保留原值和单位，按顺序输出 SKU: 值 | SKU: 值。")
                    answer = f"{skus[0]}: {numeric[key]} | {skus[1]}: {other[key2]}"
                    pages = gold = [cards[0][1], cards[1][1]]
                elif suite == "table":
                    if chart:
                        question = f"图中 {ids[0]} 的 {key} 条形终点标了什么值？同时抄写横轴单位。"
                    else:
                        question = f"规格表的 {key} 行对 {ids[0]} 给出了什么？数值和原单位一起填。"
                    answer, gold = numeric[key], [cards[0][1]]
                elif suite == "evidence":
                    if len(group) == 2:
                        question = f"两份卡中请只查 {ids[0]} 的{field}。原单位照抄并引用该商品的规格页。"
                        answer, gold = numeric[key], [cards[0][1]]
                        pages = [cards[1][1], cards[0][1]]
                    else:
                        question = f"请分别核实 {ids[0]} 的品牌和{field}。输出 品牌原值 | 规格原值，引用支持各项的两页。"
                        answer, gold = f"{attrs['brand']} | {numeric[key]}", cards[0]
                else:
                    unknown = ("出厂序列号", "单品出厂检测报告编号", "制造日期", "授权维修点电话", "未开封包装的精确重量")
                    question = f"证据页有没有记录 {ids[0]} 的{unknown[index % len(unknown)]}？请只按页面作答。"
                    answer, gold = "NOT_FOUND", []
                if index % 2 and len(pages) == 2:
                    pages = list(reversed(pages))
                rows.append({"question_id": f"{id_prefix}-{split}-{suite}-{index + 1:03d}", "question": question,
                             "request": {"skus": skus, "store_id": one["domain"] if len({p["domain"] for p, _, _ in group}) == 1 else "all_ABO_domains"},
                             "split": split, "suite": suite, "family_keys": [p["family_key"] for p, _, _ in group],
                             "product_type": one["product_type"], "listing_keys": ids,
                             "reference_answer": answer, "reference_pages": [p["page_id"] for p in gold],
                             "expected_action": "answer" if gold else "not_found", "retrieval_seconds": 0.0,
                             "retrieved_pages": [{**page, "rank": rank} for rank, page in enumerate(pages, 1)]})
    write_jsonl(data / "questions.jsonl", rows)
    write_json(data / "protocol.json", {"version": version, "groups": GROUPS, "splits": splits,
                "questions_sha256": file_hash(data / "questions.jsonl"),
                "scope": "disjoint families; fixed evidence; source-derived ABO tables and charts" if charts else
                         "new disjoint families; fixed evidence, derived ABO specification tables, no genuine manufacturer manual/graph",
                "max_pixels": 460800, "max_tokens": 128, "seed": 9026, "epochs": 1, "accumulation": 2,
                "learning_rate": 0.00015})
    load(data, splits)


def load(data=DATA, splits=SPLITS):
    protocol = json.loads((data / "protocol.json").read_text("utf-8"))
    file = data / "questions.jsonl"
    if file_hash(file) != protocol["questions_sha256"]:
        raise ValueError("Frozen question SHA differs")
    rows = read_jsonl(file)
    families = defaultdict(set)
    paths = {}
    for row in rows:
        families[row["split"]].update(row["family_keys"])
        for page in row["retrieved_pages"]:
            page["pdf"] = PureWindowsPath(page["pdf"]).as_posix()
            paths[page["pdf"]] = page["pdf_sha256"]
    if sum(map(len, families.values())) != len(set().union(*families.values())):
        raise ValueError("Family leakage")
    if any(file_hash(WORKSPACE / path) != sha for path, sha in paths.items()):
        raise ValueError("Frozen evidence SHA differs")
    counts = Counter((r["split"], r["suite"]) for r in rows)
    if counts != Counter({(split, key): n for split, ns in splits.items() for key, n in zip(GROUPS, ns) if n}):
        raise ValueError("Frozen stratum count differs")
    return rows, protocol


if __name__ == "__main__":
    freeze()
    print(f"Frozen {sum(map(sum, SPLITS.values()))} cases at {DATA}", flush=True)

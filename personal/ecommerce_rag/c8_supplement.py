"""Freeze the additive 8B test and its separately reviewed multi-view photo cases."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path

import c8_photos
import c8_protocol
from common import WORKSPACE, file_hash, read_jsonl, write_json, write_jsonl
from full_cards import photo
from full_catalog import ROOT, connect


VERSION = "abo-8b-supplement-v1"
DATA = ROOT.parent / VERSION
QA = DATA / "qa"
PHOTOS = DATA / "photos"
SPLITS = {"test": (20, 5, 5, 0, 0)}


def freeze_qa():
    baseline, _ = c8_protocol.load()
    photos, _ = c8_photos.load()
    excluded = {key for r in baseline for key in r["family_keys"]} | {r["family_key"] for r in photos}
    c8_protocol.freeze(QA, VERSION, SPLITS, excluded=excluded, charts=True)


def stage_photos():
    from PIL import Image, ImageDraw, ImageOps

    original, _ = c8_protocol.load()
    new, _ = c8_protocol.load(QA, SPLITS)
    photos, _ = c8_photos.load()
    excluded = (c8_protocol.historical_families() | {key for r in original + new for key in r["family_keys"]}
                | {r["family_key"] for r in photos})
    image_hashes = {r["image_sha256"] for r in photos}
    candidates = []
    seen_images = set()
    with closing(connect()) as db:
        for kind, code in c8_photos.TYPES.items():
            rows = db.execute("SELECT * FROM products WHERE product_type=? AND "
                              "json_array_length(raw_json,'$.other_image_id')>=2", (kind,))
            products = sorted(rows, key=lambda row: sha256((VERSION + row["listing_key"]).encode()).digest())
            if kind == "HEADPHONES":
                products = sorted(products, key=lambda row: "packaging" not in row["raw_json"].casefold())
            groups = 0
            for row in products:
                if row["family_key"] in excluded:
                    continue
                groups += 1
                excluded.add(row["family_key"])
                alternate = json.loads(row["raw_json"])["other_image_id"][:3]
                for index, image_id in enumerate([row["main_image_id"], *alternate], 1):
                    if not image_id or image_id in seen_images:
                        continue
                    seen_images.add(image_id)
                    candidates.append((dict(row, main_image_id=image_id), f"S{code}{groups}{index}", index))
                if groups == 4:
                    break
    sources = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        downloaded = pool.map(photo, (p for p, _, _ in candidates))
        for (row, name, index), (path, record) in zip(candidates, downloaded, strict=True):
            if path is None or record["sha256"] in image_hashes:
                continue
            image_hashes.add(record["sha256"])
            sources.append({"id": name, "view_index": index, "product_type": row["product_type"],
                            "listing_key": row["listing_key"], "family_key": row["family_key"], "sku": row["sku"],
                            "domain": row["domain"], "image": path.relative_to(WORKSPACE).as_posix(),
                            "image_sha256": record["sha256"], "url": record["url"]})
    for kind, code in c8_photos.TYPES.items():
        items = [r for r in sources if r["product_type"] == kind]
        if len(items) < 2:
            raise RuntimeError(f"Too few official photo candidates for {kind}")
        sheet = Image.new("RGB", (1000, 265 * ((len(items) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(items):
            left, top = 250 * (i % 4), 265 * (i // 4)
            with Image.open(WORKSPACE / row["image"]) as image:
                view = ImageOps.contain(image.convert("RGB"), (238, 235))
            sheet.paste(view, (left + (250 - view.width) // 2, top + 22))
            draw.text((left + 7, top + 3), row["id"] + " " + row["sku"], fill="black")
        sheet_dir = PHOTOS / "contact-sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_dir / f"{code}.jpg", quality=92)
    write_jsonl(PHOTOS / "candidates.jsonl", sources)
    print(Counter(r["product_type"] for r in sources), flush=True)


def freeze_photos():
    labels = PHOTOS / "annotations.jsonl"
    chosen = {row["id"] for row in read_jsonl(labels)}
    sources = [row for row in read_jsonl(PHOTOS / "candidates.jsonl") if row["id"] in chosen]
    if Counter(r["product_type"] for r in sources) != Counter({kind: 2 for kind in c8_photos.TYPES}):
        raise ValueError("Two inspected photos per category are required")
    if (PHOTOS / "protocol.json").is_file():
        raise FileExistsError("Supplemental photo protocol is frozen")
    write_jsonl(PHOTOS / "sources.jsonl", sources)
    c8_photos.freeze(labels, PHOTOS, VERSION)
    load()


def load():
    qa, qa_protocol = c8_protocol.load(QA, SPLITS)
    photos, photo_protocol = c8_photos.load(PHOTOS, 2)
    old_qa, _ = c8_protocol.load()
    old_photos, _ = c8_photos.load()
    old_families = {k for r in old_qa for k in r["family_keys"]} | {r["family_key"] for r in old_photos}
    qa_families = {k for r in qa for k in r["family_keys"]}
    photo_families = {r["family_key"] for r in photos}
    image_hashes = {r["image_sha256"] for r in photos + old_photos}
    if (old_families & (qa_families | photo_families) or qa_families & photo_families
            or len(image_hashes) != len(photos + old_photos)):
        raise ValueError("Overlapping case families or source photo duplicates")
    if file_hash(PHOTOS / "annotations.jsonl") != photo_protocol["annotations_sha256"]:
        raise ValueError("Photo annotations changed after freezing")
    return qa + photos, {"qa": qa_protocol, "photos": photo_protocol}


def combine(parent, supplement, output):
    from audit import format_question
    from stage_c import report

    parent_qa, _ = c8_protocol.load()
    parent_photos, _ = c8_photos.load()
    extra, _ = load()
    parent_cases = [r for r in parent_qa if r["split"] != "train"] + parent_photos
    variants = ("C0", "C1", "C2")
    model_sha = file_hash(parent / "adapter/adapter_model.safetensors")
    parent_sha = file_hash(parent / "config.json")
    extra_config = json.loads((supplement / "config.json").read_text("utf-8"))
    training = json.loads((parent / "training-summary.json").read_text("utf-8"))
    if (extra_config["parent_config_sha256"] != parent_sha or extra_config["parent_adapter_sha256"] != model_sha
            or training["adapter_sha256"] != model_sha):
        raise ValueError("Parent run provenance does not match")
    records = {}
    origins = {}
    for folder, cases in ((parent, parent_cases), (supplement, extra)):
        expected = {r["question_id"]: r for r in cases}
        config_sha = file_hash(folder / "config.json")
        rows = read_jsonl(folder / "results.jsonl")
        keys = {(r["variant"], r["question_id"]) for r in rows}
        if len(rows) != len(keys) or keys != {(variant, qid) for variant in variants for qid in expected}:
            raise ValueError(f"Missing or duplicate same-case predictions in {folder.name}")
        if records.keys() & keys:
            raise ValueError("Parent and supplement case IDs overlap")
        origins[folder.name] = {"config_sha256": config_sha, "results_sha256": file_hash(folder / "results.jsonl"),
                                "predictions": len(rows)}
        for row in rows:
            key = row["variant"], row["question_id"]
            frozen = expected[row["question_id"]]
            if any(row[field] != frozen[field] for field in ("question", "split", "suite", "reference_answer",
                                                            "reference_pages", "expected_action")):
                raise ValueError(f"Frozen labels differ for {key}")
            if [p["page_id"] for p in row["retrieved_pages"]] != [p["page_id"] for p in frozen["retrieved_pages"]]:
                raise ValueError(f"Frozen evidence order differs for {key}")
            if (row["config_sha256"] != config_sha or row["adapter_sha256"] != (model_sha if key[0] == "C2" else None)):
                raise ValueError(f"Unexpected configuration or adapter for {key}")
            records[key] = {**row, "cloud_run_directory": row["run_directory"], "run_directory": str(folder)}
    output.mkdir(parents=True)
    write_json(output / "config.json", {"sources": origins, "adapter_sha256": model_sha,
                                       "scope": "210 frozen QA + 100 photos; dev reported separately"})
    write_jsonl(output / "results.jsonl", records.values())
    (output / "逐题结果.txt").write_text("合并逐题记录，保留每题的实际来源运行。\n\n" +
                                       "".join(map(format_question, records.values())), "utf-8")
    report(output, records)
    lines = ["8B 三组配对验收", "C0 原提示 · C1 改进提示 · C2 同提示加 LoRA",
             "开发题、正式问答、照片分别计分。照片含同一商品的两个视角，汇总按题计数。", ""]
    comparison = {}
    for split in ("test", "photo", "dev"):
        groups = {v: [r for (variant, _), r in records.items() if variant == v and r["split"] == split] for v in variants}
        comparison[split] = {}
        for v, rows in groups.items():
            unknown = [r for r in rows if r["expected_action"] == "not_found"]
            answerable = len(rows) - len(unknown)
            strict = sum(r["metrics"]["strict_pass"] for r in rows)
            refusals = sum(r["metrics"]["strict_pass"] for r in unknown)
            false_refusals = sum(r["metrics"]["false_refusal"] for r in rows)
            comparison[split][v] = {"n": len(rows), "strict": strict, "correct_refusal": refusals,
                                    "unanswerable": len(unknown), "false_refusal": false_refusals, "answerable": answerable}
            lines.append(f"{split} {v} 严格 {strict}/{len(rows)}，正确拒答 {refusals}/{len(unknown)}，"
                         f"误拒答 {false_refusals}/{answerable}")
        paired = {}
        for first, second in (("C0", "C1"), ("C1", "C2")):
            counts = Counter((r["metrics"]["strict_pass"], records[second, r["question_id"]]["metrics"]["strict_pass"])
                             for r in groups[first])
            key = f"{first}→{second}"
            paired[key] = {"both_pass": counts[True, True], "both_fail": counts[False, False],
                           "improved": counts[False, True], "regressed": counts[True, False]}
            lines.append(f"{split} {key} 改善 {counts[False, True]}，退化 {counts[True, False]}，"
                         f"均通过 {counts[True, True]}，均失败 {counts[False, False]}")
        comparison[split]["paired"] = paired
        lines.append("")
    write_json(output / "paired-summary.json", comparison)
    (output / "合并验收.txt").write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("freeze-qa", "stage-photos", "freeze-photos", "combine"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task == "combine":
        if None in (args.parent, args.supplement, args.output):
            parser.error("combine requires --parent, --supplement, and --output")
        combine(args.parent, args.supplement, args.output)
    else:
        {"freeze-qa": freeze_qa, "stage-photos": stage_photos, "freeze-photos": freeze_photos}[args.task]()

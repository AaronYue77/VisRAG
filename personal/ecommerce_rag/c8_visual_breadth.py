"""Freeze a separately reported 15-category photo breadth set and compare both adapters."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path

import c8_photos
import c8_protocol
from c8_upgrade import DATA as UPGRADE, families, load as load_upgrade, read_json, summarize
from common import WORKSPACE, file_hash, read_jsonl, write_json, write_jsonl
from full_cards import photo
from full_catalog import connect


VERSION = "abo-8b-photo-breadth-v1"
DATA = UPGRADE.parent / VERSION
RUN_NAME = "stage-c-8b-photo-breadth-20261110"
CATEGORIES = {
    "CU": "DRINKING_CUP", "CL": "LAUNDRY_DETERGENT", "AU": "AUTO_OIL", "TO": "WRENCH",
    "SP": "RECREATION_BALL", "BE": "SHAMPOO", "TY": "TOY_FIGURE", "BA": "BABY_PRODUCT",
    "OF": "FILE_FOLDER", "PE": "PET_SUPPLIES", "LI": "LIGHT_BULB", "GA": "PLANTER",
    "FO": "COFFEE", "MU": "INSTRUMENT_PARTS_AND_ACCESSORIES", "JE": "FINEEARRING",
}
SIZE = 5


def stage():
    from PIL import Image, ImageDraw, ImageOps

    train, dev, test, _, _ = load_upgrade()
    excluded = families(train + dev + test) | c8_protocol.historical_families()
    existing = {r["image_sha256"] for r in train + dev + test if "image_sha256" in r}
    queued = []
    with closing(connect()) as db:
        for short, kind in CATEGORIES.items():
            matches = db.execute("SELECT p.* FROM products p JOIN images i ON p.main_image_id=i.image_id "
                                 "WHERE p.product_type=? AND min(i.width,i.height)>=300", (kind,))
            products = sorted(matches, key=lambda row: sha256((VERSION + row["listing_key"]).encode()).digest())
            count = 0
            for row in products:
                if row["family_key"] in excluded:
                    continue
                count += 1
                excluded.add(row["family_key"])
                queued.append((short, count, dict(row)))
                if count == 8:
                    break
    sources = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        photos = pool.map(photo, (row for _, _, row in queued))
        for (short, number, row), (path, asset) in zip(queued, photos, strict=True):
            if path is None or asset["sha256"] in existing:
                continue
            existing.add(asset["sha256"])
            sources.append({"id": f"X{short}{number}", "product_type": row["product_type"], "listing_key": row["listing_key"],
                            "family_key": row["family_key"], "sku": row["sku"], "domain": row["domain"],
                            "image": path.relative_to(WORKSPACE).as_posix(), "image_sha256": asset["sha256"], "url": asset["url"]})
    for short, kind in CATEGORIES.items():
        rows = [r for r in sources if r["product_type"] == kind]
        if len(rows) < SIZE:
            raise RuntimeError(f"Only {len(rows)} distinct photographs for {kind}")
        sheet = Image.new("RGB", (1200, 590), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            left, top = 300 * (i % 4), 295 * (i // 4)
            with Image.open(WORKSPACE / row["image"]) as original:
                view = ImageOps.contain(original.convert("RGB"), (285, 265))
            sheet.paste(view, (left + (300 - view.width) // 2, top + 25))
            draw.text((left + 7, top + 5), row["id"] + " " + row["sku"], fill="black")
        (DATA / "contact-sheets").mkdir(parents=True, exist_ok=True)
        sheet.save(DATA / "contact-sheets" / (short + ".jpg"), quality=94)
    write_jsonl(DATA / "candidates.jsonl", sources)
    print(Counter(r["product_type"] for r in sources), flush=True)


def freeze():
    labels_path = DATA / "annotations.jsonl"
    labels = {r["id"] for r in read_jsonl(labels_path)}
    sources = [r for r in read_jsonl(DATA / "candidates.jsonl") if r["id"] in labels]
    if len(labels) != len(sources) or Counter(r["product_type"] for r in sources) != Counter({
            key: SIZE for key in CATEGORIES.values()}):
        raise ValueError("Need five reviewed sources per category")
    if (DATA / "protocol.json").exists():
        raise FileExistsError("Photo breadth protocol is already frozen")
    write_jsonl(DATA / "sources.jsonl", sources)
    c8_photos.freeze(labels_path, DATA, VERSION)
    load()


def load():
    rows, protocol = c8_photos.load(DATA, SIZE, CATEGORIES.values())
    if file_hash(DATA / "annotations.jsonl") != protocol["annotations_sha256"]:
        raise ValueError("Breadth annotations changed")
    train, dev, test, cfg, _ = load_upgrade()
    if families(rows) & families(train + dev + test) or len(families(rows)) != len(rows):
        raise ValueError("Breadth families overlap")
    hashes = {r["image_sha256"] for r in train + dev + test if "image_sha256" in r}
    if len({r["image_sha256"] for r in rows} - hashes) != len(rows):
        raise ValueError("Breadth photos overlap")
    return rows, cfg


def run(model_dir, prior_dir, candidate_dir, output):
    import peft
    import torch

    from c8_experiment import evaluate_cases
    from qwen8_runtime import MODEL_ID, REVISION, Qwen8Runtime
    from stage_a import Reader, snapshot
    from stage_c import PROMPT_C1

    rows, settings = load()
    baseline = read_json(UPGRADE / "baseline.json")
    old_sha = file_hash(prior_dir / "adapter/adapter_model.safetensors")
    new_sha = file_hash(candidate_dir / "adapter/adapter_model.safetensors")
    if old_sha != baseline["adapter_sha256"] or new_sha != read_json(candidate_dir / "training-summary.json")["adapter_sha256"]:
        raise ValueError("Adapter artifact mismatch")
    expected = {"model": MODEL_ID, "revision": REVISION, "settings": settings, "prompt_addition": PROMPT_C1,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "quantization": "nf4"}
    config = read_json(candidate_dir / "config.json")
    if (any(config[k] != value for k, value in expected.items()) or config["baseline_sha256"] !=
            file_hash(UPGRADE / "baseline.json")):
        raise ValueError("Incompatible candidate for photo breadth run")
    output.mkdir(parents=True)
    source_sha = snapshot(output)
    reader = Reader(max_tokens=settings["max_tokens"], vlm=Qwen8Runtime(model_dir))
    if str(reader.vlm.dtype) != config["compute_dtype"]:
        raise ValueError("Compute dtype differs from candidate")
    write_json(output / "config.json", {**expected, "source_sha256": source_sha, "compute_dtype": str(reader.vlm.dtype),
               "protocol_sha256": file_hash(DATA / "protocol.json"), "adapter_sha256": [old_sha, new_sha]})
    config_sha = file_hash(output / "config.json")
    model = peft.PeftModel.from_pretrained(reader.vlm.model, prior_dir / "adapter", adapter_name="prior").eval()
    reader.vlm.model = model
    evaluate_cases(reader, rows, output, settings, config_sha, source_sha, "prior", PROMPT_C1, old_sha)
    model.load_adapter(candidate_dir / "adapter", adapter_name="candidate")
    model.set_adapter("candidate")
    evaluate_cases(reader, rows, output, settings, config_sha, source_sha, "candidate", PROMPT_C1, new_sha)
    print("Photo breadth complete: 150 same-case predictions", flush=True)


def compare(output, candidate_sha):
    rows, _ = load()
    config = read_json(output / "config.json")
    baseline = read_json(UPGRADE / "baseline.json")
    hashes = [baseline["adapter_sha256"], candidate_sha]
    if config["protocol_sha256"] != file_hash(DATA / "protocol.json") or config["adapter_sha256"] != hashes:
        raise ValueError("Breadth protocol or adapter mismatch")
    config_sha = file_hash(output / "config.json")
    actual = read_jsonl(output / "results.jsonl")
    by_key = {(r["variant"], r["question_id"]): r for r in actual}
    variants = ("prior", "candidate")
    if len(by_key) != len(actual) or by_key.keys() != {(v, r["question_id"]) for v in variants for r in rows}:
        raise ValueError("Missing or duplicate breadth predictions")
    pairs = defaultdict(list)
    changes = []
    for case in rows:
        qid = case["question_id"]
        pair = tuple(by_key[v, qid] for v in variants)
        for row, sha in zip(pair, hashes, strict=True):
            if (row["adapter_sha256"] != sha or row["config_sha256"] != config_sha or row["status"] != "ok"
                    or any(row[k] != case[k] for k in ("question", "reference_answer", "reference_pages", "expected_action"))
                    or [p["page_id"] for p in row["retrieved_pages"]] != [p["page_id"] for p in case["retrieved_pages"]]):
                raise ValueError(f"Invalid breadth row: {qid}")
        pairs["all"].append(pair)
        pairs[case["suite"]].append(pair)
        before, after = pair
        if before["metrics"]["strict_pass"] != after["metrics"]["strict_pass"]:
            changes.append({"id": qid, "suite": case["suite"], "old": before["raw_output"], "new": after["raw_output"],
                            "old_pass": before["metrics"]["strict_pass"], "new_pass": after["metrics"]["strict_pass"]})
    metrics = {key: summarize(value) for key, value in pairs.items()}
    all_rows = metrics["all"]
    passed = all_rows["new"] >= all_rows["old"] and all(all_rows[key][1] <= all_rows[key][0]
                                                       for key in ("false_refusal", "unsupported_answer"))
    write_json(output / "breadth-metrics.json", {"passes": passed, "by_category": metrics})
    (output / "跨类目照片验收.txt").write_text(
        "跨类目照片增补：15 类，各 5 张。与原 100 道保持独立统计。\n"
        + ("增补通过切换保护。\n" if passed else "增补未通过切换保护。\n")
        + "\n".join(f"{key} | {v['old']}/{v['n']} → {v['new']}/{v['n']} | 改善 {v['improved']}，退化 {v['regressed']}"
                    for key, v in metrics.items()) + "\n\n逐题变化\n" + json.dumps(changes, ensure_ascii=False, indent=2) + "\n",
        "utf-8")
    return passed, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("stage", "freeze", "run"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    if args.task == "stage":
        stage()
    elif args.task == "freeze":
        freeze()
    else:
        if not all((args.model_dir, args.prior_dir, args.candidate_dir)):
            parser.error("run requires model, prior, and candidate directories")
        run(args.model_dir, args.prior_dir, args.candidate_dir, WORKSPACE / "reports/ecommerce_rag" / RUN_NAME)

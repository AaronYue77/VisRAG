"""Stage source images for visual annotation and freeze the manually labelled cases."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from c8_protocol import DATA as QA, historical_families, load as load_qa
from common import WORKSPACE, file_hash, read_jsonl, write_json, write_jsonl
from full_cards import photo
from full_catalog import ROOT, connect


DATA = ROOT.parent / "abo-8b-photos-v1"
TYPES = {"CHAIR": "CH", "HEADPHONES": "HP", "TABLE": "TB", "SHOES": "SH", "BACKPACK": "BP"}
SIZE = 18


def prepare():
    from PIL import Image, ImageDraw, ImageOps

    qa, _ = load_qa()
    used = historical_families() | {key for r in qa for key in r["family_keys"]}
    data = []
    db = connect()
    try:
        for kind, code in TYPES.items():
            candidates = sorted(db.execute(
                "SELECT p.* FROM products p JOIN images i ON p.main_image_id=i.image_id "
                "WHERE p.product_type=? AND min(i.width,i.height)>=380", (kind,)),
                key=lambda row: sha256(("8b-photo-v1" + row["listing_key"]).encode()).digest())
            selected = []
            for product in candidates:
                if product["family_key"] in used:
                    continue
                selected.append(dict(product))
                used.add(product["family_key"])
                if len(selected) == SIZE + 4:
                    break
            with ThreadPoolExecutor(max_workers=8) as pool:
                downloaded = list(pool.map(photo, selected))
            for row, (image_path, record) in zip(selected, downloaded):
                if not image_path:
                    continue
                index = 1 + sum(r["product_type"] == kind for r in data)
                data.append({"id": f"{code}{index:02d}", "product_type": kind, "listing_key": row["listing_key"],
                             "family_key": row["family_key"], "sku": row["sku"], "domain": row["domain"],
                             "title": row["title"], "image": image_path.relative_to(WORKSPACE).as_posix(),
                             "image_sha256": record["sha256"], "url": record["url"]})
                if index == SIZE:
                    break
            print(kind, sum(r["product_type"] == kind for r in data), flush=True)
    finally:
        db.close()
    if Counter(r["product_type"] for r in data) != Counter({kind: SIZE for kind in TYPES}):
        raise RuntimeError("Insufficient distinct photos; retain downloads and retry")
    for kind, code in TYPES.items():
        subset = [r for r in data if r["product_type"] == kind]
        sheet = Image.new("RGB", (1500, 840), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(subset):
            left, top = (i % 6) * 250, (i // 6) * 280
            with Image.open(WORKSPACE / row["image"]) as image:
                preview = ImageOps.contain(image.convert("RGB"), (240, 245))
            sheet.paste(preview, (left + (250 - preview.width) // 2, top + 18))
            draw.text((left + 7, top + 3), row["id"], fill="black")
        (DATA / "contact-sheets").mkdir(parents=True, exist_ok=True)
        sheet.save(DATA / "contact-sheets" / f"{code}.jpg", quality=93)
    write_jsonl(DATA / "sources.jsonl", data)


def freeze(labels, data=DATA, version="abo-8b-photos-v1"):
    from PIL import Image, ImageFilter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    protocol_path = data / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Photo protocol is already frozen")
    annotations = {row["id"]: row for row in read_jsonl(labels)}
    sources = read_jsonl(data / "sources.jsonl")
    if set(annotations) != {row["id"] for row in sources}:
        raise ValueError("Every photo needs a human-reviewed label")
    rows = []
    for item in sources:
        label = annotations[item["id"]]
        radius = label.get("blur_radius")
        image_path = WORKSPACE / item["image"]
        if file_hash(image_path) != item["image_sha256"]:
            raise ValueError("Source photo changed")
        doc = "c8photo-" + item["id"]
        path = data / "cards" / (doc + ".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas = Canvas(str(path), pagesize=(595, 595), invariant=1)
        canvas.drawString(25, 565, "ABO photo | " + item["listing_key"])
        with Image.open(image_path) as image:
            view = image.convert("RGB")
            view.thumbnail((1000, 1000))
            if radius:
                view = view.filter(ImageFilter.GaussianBlur(radius))
            scale = min(515 / view.width, 495 / view.height)
            encoded = BytesIO()
            view.save(encoded, format="JPEG", quality=88)
            encoded.seek(0)
            canvas.drawImage(ImageReader(encoded), (595 - view.width * scale) / 2, 48,
                             width=view.width * scale, height=view.height * scale)
        canvas.setFont("Helvetica", 8)
        treatment = "downscaled with simulated blur" if radius else "downscaled for evidence presentation"
        canvas.drawString(25, 22, f"ABO original photo, {treatment} | CC BY 4.0")
        canvas.showPage()
        canvas.save()
        page = {"doc_id": doc, "page": 1, "page_id": doc + ":p0001", "version": version,
                "pdf": path.relative_to(WORKSPACE).as_posix(), "pdf_sha256": file_hash(path),
                "sku": item["sku"], "listing_key": item["listing_key"], "rank": 1, "score": 1.0,
                "text_source": "original_ABO_photo"}
        answer = label["answer"]
        rows.append({**item, "question_id": "C8P-" + item["id"], "question": label["question"],
                     "split": "photo", "suite": item["product_type"], "request": {"skus": [item["sku"]],
                     "store_id": item["domain"]}, "reference_answer": answer,
                     "reference_pages": [] if answer == "NOT_FOUND" else [page["page_id"]],
                     "expected_action": "not_found" if answer == "NOT_FOUND" else "answer",
                     "annotation": label["condition"], "answer_aliases": label.get("aliases", []),
                     **({"blur_radius": radius} if radius else {}),
                     "retrieved_pages": [page], "retrieval_seconds": 0.0})
    write_jsonl(data / "questions.jsonl", rows)
    write_json(protocol_path, {"version": version, "questions_sha256": file_hash(data / "questions.jsonl"),
                              "annotations_sha256": file_hash(labels), "scope": "visual inspection of actual ABO images",
                              "categories": dict(Counter(row["product_type"] for row in rows))})


def load(data=DATA, size=SIZE, categories=TYPES):
    protocol = json.loads((data / "protocol.json").read_text("utf-8"))
    if file_hash(data / "questions.jsonl") != protocol["questions_sha256"]:
        raise ValueError("Frozen photo questions changed")
    rows = read_jsonl(data / "questions.jsonl")
    if Counter(row["product_type"] for row in rows) != Counter({key: size for key in categories}):
        raise ValueError("Frozen photo category count differs")
    for row in rows:
        page, = row["retrieved_pages"]
        if file_hash(WORKSPACE / page["pdf"]) != page["pdf_sha256"]:
            raise ValueError("Frozen photo PDF changed")
    return rows, protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("prepare", "freeze"))
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    if args.task == "prepare":
        prepare()
    else:
        freeze(args.labels)

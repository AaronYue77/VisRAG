from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

from common import WORKSPACE, read_jsonl, write_json, write_jsonl, file_hash

STOP = set("a an the is are was were be been being of for to in on at by with and or from as that this which what how many much does do did used use using it its their we our than".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text.lower())
    result = []
    for token in re.findall(r"[a-z_][a-z_0-9]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text):
        if re.search(r"[\u4e00-\u9fff]", token):
            import jieba
            result.extend(t.strip() for t in jieba.cut(token) if t.strip())
        elif token not in STOP:
            result.append(token)
    return result


def page_id(doc_id: str, page: int) -> str:
    if page < 1:
        raise ValueError("PDF page numbers must be one-based")
    return f"{doc_id}:p{page:04d}"


def render_page(pdf: Path, number: int, max_pixels: int = 786432):
    import pymupdf
    from PIL import Image
    with pymupdf.open(pdf) as document:
        page = document[number - 1]
        scale = math.sqrt(max_pixels / (page.rect.width * page.rect.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(32, int(image.width * factor)), max(32, int(image.height * factor))))
    return image


def build_pages(documents: list[dict], output: Path, ocr_fallback: bool = True) -> list[dict]:
    import pymupdf
    rows = []
    ocr = None
    for document in documents:
        pdf = WORKSPACE / document["pdf"]
        digest = file_hash(pdf)
        if document.get("sha256") and digest != document["sha256"]:
            raise ValueError(f"PDF changed: {document['doc_id']}")
        with pymupdf.open(pdf) as source:
            for number, page in enumerate(source, start=1):
                text = page.get_text(sort=True).strip()
                origin = "pdf_text"
                if len(text) < 20 and ocr_fallback:
                    if ocr is None:
                        from rapidocr import RapidOCR
                        ocr = RapidOCR(config_path=str(WORKSPACE / "configs/ocr.yaml"))
                    import numpy as np
                    result = ocr(np.asarray(render_page(pdf, number)))
                    text = "\n".join(result.txts or [])
                    origin = "ocr"
                rows.append({
                    "page_id": page_id(document["doc_id"], number), "doc_id": document["doc_id"],
                    "title": document["title"], "pdf": document["pdf"], "page": number,
                    "pdf_sha256": digest, "text": text, "text_source": origin,
                    "width_pt": page.rect.width, "height_pt": page.rect.height,
                })
        print(f"Indexed {document['doc_id']}: {number} pages", flush=True)
    write_jsonl(output, rows)
    return rows


class TextRetriever:
    def __init__(self, pages: list[dict]):
        from rank_bm25 import BM25Okapi
        if not pages:
            raise ValueError("The page corpus is empty")
        self.pages = pages
        self.bm25 = BM25Okapi([tokenize(p["text"]) or ["__empty__"] for p in pages], k1=1.5, b=0.75)

    def search(self, question: str, top_k: int) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.pages[i]["page_id"]))[:top_k]
        return [{**self.pages[i], "rank": rank, "score": float(scores[i])} for rank, i in enumerate(order, start=1)]


def normalize(text: str) -> str:
    text = text.lower().replace(",", "")
    text = re.sub(r"(?<=\d)\.0\b", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(re.findall(r"[\w]+(?:\.\d+)?", text, re.UNICODE))


def answer_metrics(answer: str, aliases: list[str]) -> dict:
    candidate = normalize(answer)
    em = float(any(candidate == normalize(gold) for gold in aliases))
    best = 0.0
    for gold in aliases:
        predicted, expected = candidate.split(), normalize(gold).split()
        common = sum((Counter(predicted) & Counter(expected)).values())
        if not predicted or not expected:
            score = float(predicted == expected)
        elif common:
            precision, recall = common / len(predicted), common / len(expected)
            score = 2 * precision * recall / (precision + recall)
        else:
            score = 0.0
        best = max(best, score)
    return {"normalized_em": em, "token_f1": best}


def retrieval_metrics(retrieved: list[str], gold: list[str]) -> dict:
    relevant = set(gold)
    if not relevant:
        raise ValueError("Gold evidence pages cannot be empty for these answerable suites")
    found = set(retrieved) & relevant
    first = next((rank for rank, value in enumerate(retrieved, start=1) if value in relevant), None)
    return {"hit_at_k": float(bool(found)), "recall_at_k": len(found) / len(relevant), "reciprocal_rank": 1 / first if first else 0.0}


def verify_questions(questions: list[dict], pages: list[dict]) -> None:
    ids = set()
    corpus = {row["page_id"] for row in pages}
    groups = {}
    for question in questions:
        if question["question_id"] in ids:
            raise ValueError("Duplicate question_id")
        ids.add(question["question_id"])
        if not question["reference_answer"].strip() or not question["reference_pages"]:
            raise ValueError("Missing gold answer/evidence")
        if set(question["reference_pages"]) - corpus:
            raise ValueError(f"Unknown gold page: {question['question_id']}")
        for document in question["document_ids"]:
            previous = groups.setdefault(document, question["split"])
            if previous != question["split"]:
                raise ValueError(f"Document split leakage: {document}")


def build_messages(question: str, page_records: list[dict], image_paths: list[Path]) -> list[dict]:
    """Inference accepts only the user question and retrieved pages, never reference labels."""
    content = [{"type": "text", "text": (
        "Answer the question using only the supplied document page images. "
        "Return a JSON object with keys answer and citations. Keep answer short. "
        "citations must contain supplied page labels such as P1, P2. "
        'Use valid JSON, for example {"answer":"42","citations":["P1"]}. '
        'The citations value must be an array of quoted strings, never a single string. '
        'If evidence is insufficient, return {"answer":"NOT_FOUND","citations":[]}. '
        "Do not use outside knowledge. Do not include reasoning or markdown."
    )}]
    for index, (page, path) in enumerate(zip(page_records, image_paths), start=1):
        content.extend([{"type": "text", "text": f"P{index}: {page['title']}; PDF page {page['page']}"}, {"type": "image", "image": str(path)}])
    content.append({"type": "text", "text": f"Question: {question}"})
    return [{"role": "user", "content": content}]


def parse_answer(text: str, pages: list[dict]) -> dict:
    parsed = None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict) and isinstance(value.get("answer"), str) and isinstance(value.get("citations"), list):
                parsed = value
                break
        except ValueError:
            pass
    if parsed is None:
        return {"answer": text.strip(), "citations": [], "citation_labels": [], "invalid_citations": [], "parse_ok": False}
    mapping = {f"P{i}": row["page_id"] for i, row in enumerate(pages, start=1)}
    labels = [str(value) for value in parsed["citations"]]
    return {"answer": parsed["answer"].strip(), "citation_labels": labels, "citations": [mapping[value] for value in labels if value in mapping], "invalid_citations": [value for value in labels if value not in mapping], "parse_ok": True}

"""A stage: PDF/OCR, incremental page cache, rule routing and BM25 only."""
from datetime import date
import json
import hashlib
import importlib.metadata
import math
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "rag_common"))
from common import WORKSPACE, file_hash, read_jsonl, write_jsonl, write_json, operation
from core import page_id, render_page, tokenize

DEFAULT_CORPUS = WORKSPACE / "data/ecommerce_rag/abo-cards-v1"
EXTRACTOR_VERSION = "a-pdf-blocks-ocr-v2"


def route(question, request, documents):
    """Only public request context; reference labels cannot enter this signature."""
    if not isinstance(question, str) or not question.strip() or len(question) > 3000:
        raise ValueError("问题必须为 1-3000 字符")
    if not isinstance(request, dict) or not isinstance(request.get("skus", []), list):
        raise ValueError("request.skus must be an array")
    snapshot = documents[0]["snapshot_date"]
    as_of = request.get("as_of", snapshot)
    date.fromisoformat(as_of)
    store = request.get("store_id", "amazon.com")
    if re.search(r"订单|物流|快递|包裹|多少钱|售价|实时价格|库存|有货|下单|退款|取消订单", question):
        return {"action": "requires_tool", "reason": "live_business", "answer": "当前仅查询商品静态资料，未接入订单、价格、库存或交易工具"}
    if re.search(r"天气|旅游|写代码|股票|笑话", question):
        return {"action": "unsupported", "reason": "out_of_scope", "answer": "当前仅支持商品资料问答"}
    known = {d["sku"].upper() for d in documents}
    mentions = {sku for sku in known if re.search(r"(?<![A-Z0-9])" + re.escape(sku) + r"(?![A-Z0-9])", question.upper())}
    # ASIN-like IDs outside this corpus must not silently resolve to another item.
    mentions.update(re.findall(r"\bB0[A-Z0-9]{8}\b", question.upper()))
    supplied = {str(s).upper() for s in request.get("skus", [])}
    if mentions and supplied and not mentions <= supplied:
        return {"action": "clarify", "reason": "context_conflict", "answer": "问题中的商品编号与当前商品上下文不一致，请确认商品"}
    skus = sorted(supplied or mentions)
    if re.search(r"店铺规则|退货政策|换货政策|售后政策", question):
        return {"action": "not_found", "reason": "no_policy_source", "answer": "此官方商品数据未提供当前店铺售后政策"}
    if not skus or len(skus) > 2:
        return {"action": "clarify", "reason": "missing_or_many_products", "answer": "请指定一到两个商品编号"}
    if not set(skus) <= known:
        return {"action": "not_found", "reason": "unknown_product", "answer": "知识库没有该商品编号"}
    if as_of != snapshot:
        return {"action": "not_found", "reason": "no_historical_snapshot", "answer": "当前仅有下载日登记的静态数据快照，没有请求日期的历史资料"}
    if any(not any(d["sku"].upper() == sku and d["store_id"] == store for d in documents) for sku in skus):
        return {"action": "not_found", "reason": "store_scope", "answer": "没有该店铺范围的商品资料"}
    return {"action": "retrieve", "reason": "static_product", "skus": skus, "store_id": store, "as_of": as_of}


def extract(corpus=DEFAULT_CORPUS, ocr_enabled=True, cache_dir=None):
    import pymupdf
    import numpy as np
    documents = json.loads((corpus / "documents.json").read_text("utf-8"))
    cache = cache_dir or corpus / "page_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ocr_sha = file_hash(WORKSPACE / "configs/ocr.yaml")
    signature={"extractor":EXTRACTOR_VERSION,"pymupdf":importlib.metadata.version("PyMuPDF"),
               "ocr_package":importlib.metadata.version("rapidocr"),"ocr_config_sha256":ocr_sha,"ocr_enabled":ocr_enabled,
               "ocr_weights":{p.name:file_hash(p) for p in sorted((WORKSPACE/"models/ocr").glob("*.onnx"))}}
    signature_sha=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    ocr, rows, reused, rebuilt = None, [], 0, 0
    started = time.perf_counter()
    for doc in documents:
        pdf = WORKSPACE / doc["pdf"]
        digest = file_hash(pdf)
        if digest != doc["sha256"]:
            raise ValueError("PDF differs from source manifest: " + doc["doc_id"])
        key = f"{digest}-{signature_sha[:16]}"
        cached = cache / (key + ".json")
        if cached.exists():
            extracted = json.loads(cached.read_text("utf-8"))
            reused += 1
        else:
            extracted = []
            with pymupdf.open(pdf) as source:
                for number, page in enumerate(source, 1):
                    started_page = time.perf_counter()
                    blocks = [{"bbox_pt": list(b[:4]), "text": b[4].strip(), "confidence": None}
                              for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
                    text = "\n".join(b["text"] for b in blocks)
                    origin = "pdf_text"
                    if len(re.sub(r"\s", "", text)) < 20 and ocr_enabled:
                        if ocr is None:
                            from rapidocr import RapidOCR
                            ocr = RapidOCR(config_path=str(WORKSPACE / "configs/ocr.yaml"))
                        image = render_page(pdf, number, 1500000)
                        result = ocr(np.asarray(image))
                        blocks = []
                        if result.txts is not None:
                            for i, value in enumerate(result.txts):
                                box = result.boxes[i]
                                blocks.append({"text": value, "confidence": float(result.scores[i]),
                                               "bbox_pt": [float(box[:, 0].min()) * page.rect.width / image.width,
                                                           float(box[:, 1].min()) * page.rect.height / image.height,
                                                           float(box[:, 0].max()) * page.rect.width / image.width,
                                                           float(box[:, 1].max()) * page.rect.height / image.height]})
                        text = "\n".join(b["text"] for b in blocks)
                        origin = "ocr" if text else "ocr_empty"
                    for i, block in enumerate(blocks):
                        block["block_id"] = f"b{i+1:03d}"
                    extracted.append({"page": number, "pdf_sha256": digest, "text": text,
                                      "text_source": origin, "blocks": blocks, "width_pt": page.rect.width,
                                      "height_pt": page.rect.height, "extraction_seconds": time.perf_counter()-started_page})
            write_json(cached, extracted)
            rebuilt += 1
        for page in extracted:
            rows.append({**doc, **page, "page_id": page_id(doc["doc_id"], page["page"]), "cache_key": key})
    write_jsonl(corpus / "pages.jsonl", rows)
    summary = {"documents": len(documents), "pages": len(rows), "cache_reused_documents": reused,
               "rebuilt_documents": rebuilt, "ocr_pages": sum(p["text_source"] == "ocr" for p in rows),
               "empty_pages": sum(not p["text"].strip() for p in rows), "seconds": time.perf_counter()-started,
               "extractor_version": EXTRACTOR_VERSION, "ocr_config_sha256": ocr_sha, "ocr_enabled": ocr_enabled,
               "cache_signature":signature,"cache_signature_sha256":signature_sha,
               "documents_sha256": file_hash(corpus / "documents.json"), "pages_sha256": file_hash(corpus / "pages.jsonl")}
    write_json(corpus / "index-manifest.json", summary)
    operation("A 阶段页面文本/OCR 与增量缓存", "PyMuPDF 按物理页保留文字块和坐标；少于20个非空字符时启用本地 RapidOCR；PDF SHA、解析版本和 OCR 配置共同作为缓存键。",
              json.dumps(summary, ensure_ascii=False), [corpus / "pages.jsonl", corpus / "index-manifest.json"], "改动文档只重算对应缓存；输出页面仍保留原图和来源字段。不是通用表格结构识别器。")
    return summary


def chunks(pages, strategy="page", overlap=0):
    result = []
    if not 0 <= overlap <= 1:
        raise ValueError("block overlap must be 0 or 1")
    for page in pages:
        if strategy == "page":
            pieces = [page["text"]]
        elif strategy == "blocks":
            blocks = [b["text"] for b in page["blocks"]]
            pieces = ["\n".join(blocks[i:i+4]) for i in range(0, len(blocks), 4-overlap)]
        else:
            raise ValueError("Unknown chunk strategy")
        for i, piece in enumerate(pieces):
            result.append({"chunk_id": page["page_id"] + f":c{i+1:03d}", "page_id": page["page_id"], "text": piece})
    return result


class BM25Retriever:
    def __init__(self, corpus=DEFAULT_CORPUS, strategy="page", overlap=0, filters=True):
        from rank_bm25 import BM25Okapi
        self.corpus, self.strategy, self.overlap, self.filters = corpus, strategy, overlap, filters
        self.documents = json.loads((corpus / "documents.json").read_text("utf-8"))
        for doc in self.documents:
            if file_hash(WORKSPACE / doc["pdf"]) != doc["sha256"]:
                raise ValueError("Source PDF changed; re-register and re-index: " + doc["doc_id"])
        self.pages = read_jsonl(corpus / "pages.jsonl")
        manifest = json.loads((corpus / "index-manifest.json").read_text("utf-8"))
        if manifest["pages_sha256"] != file_hash(corpus / "pages.jsonl") or manifest["documents_sha256"] != file_hash(corpus / "documents.json"):
            raise ValueError("Stale or modified index; re-index before querying")
        self.by_id = {p["page_id"]: p for p in self.pages}
        self.chunks = chunks(self.pages, strategy, overlap)
        self.bm25 = BM25Okapi([tokenize(c["text"]) or ["__empty__"] for c in self.chunks], k1=1.5, b=.75)

    def search(self, question, request, top_k=2):
        if not 1 <= top_k <= 4:
            raise ValueError("top_k must be 1..4 for the 8GB image budget")
        started = time.perf_counter()
        decision = route(question, request, self.documents)
        trace = {"mode": "bm25", "chunk_strategy": self.strategy, "overlap_blocks": self.overlap, "filters": self.filters}
        if decision["action"] != "retrieve":
            return {"plan": decision, "pages": [], "trace": trace, "seconds": time.perf_counter()-started}
        scores = self.bm25.get_scores(tokenize(question))
        allowed = {p["page_id"] for p in self.pages if not self.filters or (p["sku"] in decision["skus"] and p["store_id"] == decision["store_id"])}
        ordering = sorted((i for i, c in enumerate(self.chunks) if c["page_id"] in allowed),
                          key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]))
        unique = []
        for i in ordering:
            key = self.chunks[i]["page_id"]
            if key not in [self.chunks[j]["page_id"] for j in unique]:
                unique.append(i)
        # A multi-product request reserves one candidate per product within the explicit budget.
        if self.filters and len(decision["skus"]) > 1 and top_k >= len(decision["skus"]):
            first = [next(i for i in unique if self.by_id[self.chunks[i]["page_id"]]["sku"] == sku) for sku in decision["skus"]]
            unique = first + [i for i in unique if i not in first]
        selected = unique[:top_k]
        rows = [{**self.by_id[self.chunks[i]["page_id"]], "rank": rank, "score": float(scores[i]),
                 "bm25_score": float(scores[i]), "dense_score": None, "best_chunk_id": self.chunks[i]["chunk_id"]}
                for rank, i in enumerate(selected, 1)]
        trace.update(eligible_pages=len(allowed), excluded_pages=len(self.pages)-len(allowed), candidate_chunks=len(ordering),
                     page_ranking=[self.chunks[i]["page_id"] for i in unique],
                     zero_keyword_scores=all(p["score"] <= 0 for p in rows))
        return {"plan": decision, "pages": rows, "trace": trace, "seconds": time.perf_counter()-started}

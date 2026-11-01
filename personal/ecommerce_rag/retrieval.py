"""RAG-only routing, effective-document filters, BM25/dense search and RRF."""
from datetime import date
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "a_baseline"))
from common import WORKSPACE, read_jsonl, write_jsonl, write_json, file_hash, operation
from core import build_pages, tokenize

EMBED_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def active(page, store_id, as_of):
    value = date.fromisoformat(as_of)
    return (page["store_id"] == store_id and date.fromisoformat(page["valid_from"]) <= value
            and (page.get("valid_to") is None or value <= date.fromisoformat(page["valid_to"])))


def plan(question, request, documents):
    """Task inputs only: never accept expected_action, reference_answer or gold pages."""
    store = request.get("store_id", "demo_store")
    as_of = request.get("as_of", "2026-11-01")
    date.fromisoformat(as_of)
    if re.search(r"订单|物流|快递|包裹", question):
        return {"action": "requires_tool", "answer": "当前 RAG 未接入订单与物流查询服务", "required_tool": "order_or_logistics_lookup"}
    if re.search(r"多少钱|售价|实时价格|库存|有货|卖.*钱", question):
        return {"action": "requires_tool", "answer": "当前 RAG 未接入实时价格与库存查询服务", "required_tool": "price_or_inventory_lookup"}
    mentioned = sorted(set(re.findall(r"DN-R\d+", question.upper())))
    supplied = sorted(set(str(x).upper() for x in request.get("skus", [])))
    if supplied and mentioned and not set(mentioned) <= set(supplied):
        return {"action": "clarify", "answer": "问题中的型号与当前商品上下文不一致，请确认型号"}
    skus = supplied or mentioned
    policy = bool(re.search(r"退货|换货|店铺规则|服务规则", question))
    if not skus and not policy:
        return {"action": "clarify", "answer": "请提供具体商品型号"}
    available = {d["sku"] for d in documents if d["sku"] != "*" and active(d, store, as_of)}
    if skus and not set(skus) <= available:
        return {"action": "not_found", "answer": "知识库没有该商品型号的有效资料"}
    return {"action": "retrieve", "skus": skus, "policy": policy, "store_id": store, "as_of": as_of}


def eligible(page, planned):
    if not active(page, planned["store_id"], planned["as_of"]):
        return False
    if planned["policy"]:
        return page["sku"] == "*" and page["doc_type"] == "store_policy"
    return page["sku"] in planned["skus"] and page["doc_type"] == "product_manual"


def rrf(rankings, c=60):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), scores


class DenseEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        path = WORKSPACE / "models/bge-small-zh-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to("cpu").eval()

    def encode(self, values, query=False):
        import numpy as np
        import torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(values), 4):
                texts = values[start:start+4]
                if query:
                    texts = [QUERY_INSTRUCTION + text for text in texts]
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                hidden = self.model(**inputs).last_hidden_state[:, 0]
                outputs.append(torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(outputs)


def make_index():
    import numpy as np
    root = WORKSPACE / "data/ecommerce_rag/demo-v1"
    docs = json.loads((root / "documents.json").read_text("utf-8"))
    pages = build_pages(docs, root / "pages.jsonl", ocr_fallback=False)
    metadata = {d["doc_id"]: d for d in docs}
    for row in pages:
        row.update({key: value for key, value in metadata[row["doc_id"]].items() if key not in ("pdf", "sha256", "title")})
    write_jsonl(root / "pages.jsonl", pages)
    encoder = DenseEncoder()
    vectors = encoder.encode([p["text"] for p in pages])
    assert vectors.shape == (len(pages), 512)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    np.save(root / "dense.npy", vectors, allow_pickle=False)
    manifest = {"pages": len(pages), "shape": list(vectors.shape), "embedding_model": "BAAI/bge-small-zh-v1.5", "embedding_revision": EMBED_REVISION,
                "pooling": "CLS + L2 normalization", "device": "cpu", "max_length": 512,
                "pages_sha256": file_hash(root / "pages.jsonl"), "vectors_sha256": file_hash(root / "dense.npy"),
                "documents_sha256": file_hash(root / "documents.json"), "questions_sha256": file_hash(root / "questions.jsonl"),
                "source_type": "synthetic_demo", "chunk_strategy": "one PDF page per chunk in this functional fixture; section/parent-child chunking remains future work"}
    write_json(root / "index-manifest.json", manifest)
    operation("建立电商图文 RAG 功能索引", "读取模拟 PDF 原生文本，绑定型号、店铺、资料类型和有效期；BGE-small-zh 在 CPU 编码 CLS 并 L2 归一化；保留页面图片供 VLM 读取。", f"{len(pages)} 页、{vectors.shape} 稠密索引已保存；BM25 在查询时使用同一文本。", [root / "pages.jsonl", root / "dense.npy", root / "index-manifest.json"], "向量有限且归一化；这是明确标注的模拟功能语料，未声称业务泛化能力。")


class Retriever:
    def __init__(self, mode="hybrid"):
        import numpy as np
        from rank_bm25 import BM25Okapi
        self.root = WORKSPACE / "data/ecommerce_rag/demo-v1"
        self.pages = read_jsonl(self.root / "pages.jsonl")
        self.documents = json.loads((self.root / "documents.json").read_text("utf-8"))
        self.manifest = json.loads((self.root / "index-manifest.json").read_text("utf-8"))
        if file_hash(self.root / "pages.jsonl") != self.manifest["pages_sha256"] or file_hash(self.root / "dense.npy") != self.manifest["vectors_sha256"]:
            raise ValueError("Index integrity check failed")
        self.mode = mode
        self.bm25 = BM25Okapi([tokenize(p["text"]) or ["__empty__"] for p in self.pages], k1=1.5, b=.75)
        self.vectors = np.load(self.root / "dense.npy", allow_pickle=False)
        self.encoder = DenseEncoder() if mode == "hybrid" else None

    def search(self, question, request, top_k=2):
        started = time.perf_counter()
        planned = plan(question, request, self.documents)
        if planned["action"] != "retrieve":
            return {"plan": planned, "pages": [], "trace": {"mode": self.mode, "model_called": False}, "seconds": time.perf_counter() - started}
        allowed = [i for i, page in enumerate(self.pages) if eligible(page, planned)]
        if not allowed:
            return {"plan": {"action": "not_found", "answer": "没有符合店铺、型号和日期的有效资料"}, "pages": [], "trace": {"mode": self.mode, "eligible_count": 0}, "seconds": time.perf_counter() - started}
        sparse = self.bm25.get_scores(tokenize(question))
        sparse_order = sorted(allowed, key=lambda i: (-float(sparse[i]), self.pages[i]["page_id"]))
        sparse_ids = [self.pages[i]["page_id"] for i in sparse_order]
        dense = None
        if self.mode == "hybrid":
            query = self.encoder.encode([question], query=True)
            dense = (query @ self.vectors.T)[0]
            dense_order = sorted(allowed, key=lambda i: (-float(dense[i]), self.pages[i]["page_id"]))
            dense_ids = [self.pages[i]["page_id"] for i in dense_order]
            ordering, scores = rrf([sparse_ids, dense_ids])
        else:
            dense_ids = []
            ordering, scores = sparse_ids, {self.pages[i]["page_id"]: float(sparse[i]) for i in sparse_order}
        by_id = {p["page_id"]: (i, p) for i, p in enumerate(self.pages)}
        selected = []
        # Preserve model diversity for an explicit multi-SKU comparison, never use gold pages.
        if len(planned["skus"]) > 1:
            for sku in planned["skus"]:
                candidate = next((key for key in ordering if by_id[key][1]["sku"] == sku), None)
                if candidate:
                    selected.append(candidate)
        selected.extend(key for key in ordering if key not in selected)
        selected = selected[:max(top_k, len(planned["skus"]))]
        rows = []
        for rank, key in enumerate(selected, start=1):
            i, page = by_id[key]
            rows.append({**page, "rank": rank, "score": scores[key], "bm25_score": float(sparse[i]), "dense_score": float(dense[i]) if dense is not None else None})
        return {"plan": planned, "pages": rows, "trace": {"mode": self.mode, "eligible_count": len(allowed), "excluded_count": len(self.pages)-len(allowed), "sparse_ranking": sparse_ids, "dense_ranking": dense_ids, "rrf_c": 60 if dense is not None else None}, "seconds": time.perf_counter()-started}

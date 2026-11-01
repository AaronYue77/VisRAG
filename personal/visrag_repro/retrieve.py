"""Original VisRAG-Ret image/query encoding and cached visual Top-k retrieval.

Runs only in .venv-visrag. Generation is a separate process/environment.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "a_baseline"))
from common import WORKSPACE, REPO, read_jsonl, write_json, write_jsonl, file_hash, timestamp, operation, issue
from compat import load_tokenizer

MODEL = WORKSPACE / "models/VisRAG-Ret"
MODEL_REVISION = "95ef596df871b606167cb7e4b7215caf1bfdf761"
INSTRUCTION = "Represent this query for retrieving relevant document: "
UPSTREAM_UTILS = REPO / "visrag_scripts/demo/visrag_pipeline/utils.py"


def safe_name(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("Names must use ASCII letters, digits, hyphens or underscores")
    return name


def snapshot(directory):
    target = directory / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    sources = {p.name: p for p in HERE.glob("*.py")}
    sources["a_common.py"] = HERE.parent / "a_baseline/common.py"
    sources["upstream_utils.py"] = UPSTREAM_UTILS
    digest = hashlib.sha256()
    for name, source in sorted(sources.items()):
        data = source.read_bytes()
        (target / name).write_bytes(data)
        digest.update(name.encode())
        digest.update(data)
    return digest.hexdigest()


def model_runtime():
    import torch
    from transformers import AutoModel, AutoTokenizer
    started = time.perf_counter()
    if torch.__version__ != "2.1.2+cu121":
        raise RuntimeError("Use the separate .venv-visrag with torch 2.1.2+cu121")
    manifest = json.loads((WORKSPACE / "reports/visrag-repro/model-manifest.json").read_text("utf-8"))
    if manifest["revision"] != MODEL_REVISION:
        raise ValueError("Unexpected retrieval checkpoint revision")
    # Verify reviewed remote Python sources before trust_remote_code local import.
    for entry in manifest["files"]:
        if entry["name"].endswith(".py") or entry["name"] == "config.json":
            if file_hash(MODEL / entry["name"]) != entry["sha256"]:
                raise ValueError(f"Reviewed model code changed: {entry['name']}")
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(MODEL)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True,
                                     torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                     low_cpu_mem_usage=True, device_map={"": "cuda:0"}).eval()
    # The upstream VisRAG_Ret.forward calls llm.model(), never llm.lm_head().
    # Move this unused generation head to CPU; do not quantize or change any weight.
    head_bytes = sum(p.numel() * p.element_size() for p in model.llm.lm_head.parameters())
    model.llm.lm_head.to("cpu")
    torch.cuda.empty_cache()
    module_spec = importlib.util.spec_from_file_location("original_visrag_embedding_utils", UPSTREAM_UTILS)
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    runtime = {"load_seconds": time.perf_counter() - started, "unused_lm_head_cpu_gib": head_bytes / 2**30,
               "process_id": os.getpid(), "python_executable": sys.executable,
               "gpu": torch.cuda.get_device_name(0), "weights_dtype": "bfloat16",
               "allocated_after_load_gib": torch.cuda.memory_allocated() / 2**30,
               "upstream_encode_sha256": file_hash(UPSTREAM_UTILS),
               "model_revision": MODEL_REVISION, "model_config_sha256": file_hash(MODEL / "config.json")}
    print(json.dumps(runtime, indent=2), flush=True)
    return model, tokenizer, original.encode, runtime


def call_encoder(model, tokenizer, encode, inputs):
    import torch
    import numpy as np
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = encode(model, tokenizer, inputs)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    if values.ndim != 2 or values.shape[0] != len(inputs) or values.shape[1] != 2304:
        raise ValueError(f"Unexpected embedding shape: {values.shape}")
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5):
        raise ValueError("Embedding is non-finite or not L2 normalized")
    return values.astype("float32"), duration, torch.cuda.max_memory_allocated() / 2**30


def index(args):
    import numpy as np
    import fitz
    from PIL import Image
    directory = WORKSPACE / "data/visrag_repro" / safe_name(args.index_name)
    if (directory / "index-manifest.json").exists():
        raise ValueError("A completed index already exists; preserve it and choose a new index-name")
    source = WORKSPACE / "data/a_baseline/smoke"
    docs = json.loads((source / "documents.json").read_text("utf-8"))
    config = {"schema_version": "1.0", "corpus": "same two technical PDFs as BM25 smoke suite", "dpi": args.dpi,
              "model": "openbmb/VisRAG-Ret", "model_revision": MODEL_REVISION,
              "upstream_encode_sha256": file_hash(UPSTREAM_UTILS), "documents": docs,
              "dtype": "bfloat16", "attention": "sdpa", "page_batch_size": 1,
              "max_slice_nums": 9, "unused_generation_head_cpu": True,
              "weights_quantized": False, "pooling": "unchanged upstream weighted_mean_pooling then L2 normalize"}
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text("utf-8")) != config:
        raise ValueError("Existing index has a different configuration; choose a new index-name")
    if not config_path.exists():
        write_json(config_path, config)
    source_sha = snapshot(directory)
    (directory / "pages").mkdir(exist_ok=True)
    (directory / "page_vectors").mkdir(exist_ok=True)
    model, tokenizer, encode, runtime = model_runtime()
    rows, arrays = [], []
    for document in docs:
        pdf = WORKSPACE / document["pdf"]
        if file_hash(pdf) != document["sha256"]:
            raise ValueError(f"PDF hash changed: {pdf}")
        with fitz.open(pdf) as pages:
            for number, page in enumerate(pages, start=1):
                started = time.perf_counter()
                image_path = directory / "pages" / f"{document['doc_id']}-{number:04d}.png"
                pix = page.get_pixmap(dpi=args.dpi, colorspace=fitz.csRGB, alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                if not image_path.exists():
                    image.save(image_path)
                rendering_seconds = time.perf_counter() - started
                vector_path = directory / "page_vectors" / f"{document['doc_id']}-{number:04d}.npy"
                # Recompute on a run, retaining failures and avoiding accidental mixed-code caches.
                vector, seconds, peak = call_encoder(model, tokenizer, encode, [image])
                np.save(vector_path, vector, allow_pickle=False)
                row = {"page_id": f"{document['doc_id']}:p{number:04d}", "doc_id": document["doc_id"],
                       "title": document["title"], "pdf": document["pdf"], "page": number,
                       "pdf_sha256": document["sha256"], "text_source": "page_image_visrag_ret",
                       "image_path": str(image_path.relative_to(WORKSPACE)), "image_sha256": file_hash(image_path),
                       "image_size": list(image.size), "vector_row": len(rows),
                       "vector_path": str(vector_path.relative_to(WORKSPACE)), "vector_sha256": file_hash(vector_path),
                       "render_seconds": rendering_seconds, "encoding_seconds": seconds, "peak_cuda_gib": peak}
                rows.append(row)
                arrays.append(vector)
                write_jsonl(directory / "page-map.in-progress.jsonl", rows)
                print(f"Encoded {row['page_id']} {vector.shape} {seconds:.2f}s peak={peak:.2f} GiB", flush=True)
    matrix = np.concatenate(arrays, axis=0)
    np.save(directory / "reps.npy", matrix, allow_pickle=False)
    write_jsonl(directory / "page-map.jsonl", rows)
    (directory / "index2img_filename.txt").write_text("\n".join(Path(r["image_path"]).name for r in rows) + "\n", "utf-8", newline="\n")
    report = {"created_at": timestamp(), "pages": len(rows), "embedding_shape": list(matrix.shape),
              "reps_sha256": file_hash(directory / "reps.npy"), "page_map_sha256": file_hash(directory / "page-map.jsonl"),
              "config_sha256": file_hash(config_path), "source_sha256": source_sha,
              "runtime": runtime, "encoding_seconds_total": sum(r["encoding_seconds"] for r in rows),
              "peak_cuda_gib": max(r["peak_cuda_gib"] for r in rows),
              "all_vectors_finite_and_normalized": True, "gold_labels_used_for_indexing": False,
              "upstream_environment": json.loads((WORKSPACE / "reports/visrag-repro/environment.json").read_text("utf-8"))}
    write_json(directory / "index-manifest.json", report)
    operation("原版 VisRAG-Ret 离线编码页面", "原版 200 DPI RGB 渲染；直接调用未改动上游 utils.encode，weighted mean pooling + L2；单页批次；仅把未参与前向的 lm_head 放 CPU。", f"{len(rows)} 页编码完成，向量矩阵 {matrix.shape}；保留逐页向量、页 ID/图像映射、哈希与耗时。", [directory / "reps.npy", directory / "page-map.jsonl", directory / "index-manifest.json"], "所有向量有限且单位归一化；页面不使用问题或参考答案；检索进程结束后释放 GPU。")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def retrieve(args):
    import torch
    import numpy as np
    directory = WORKSPACE / "data/visrag_repro" / safe_name(args.index_name)
    output = WORKSPACE / "data/visrag_repro" / safe_name(args.retrieval_name)
    if (output / "retrieval-manifest.json").exists():
        raise ValueError("A completed retrieval already exists; choose a new retrieval-name")
    output.mkdir(parents=True, exist_ok=True)
    index_manifest = json.loads((directory / "index-manifest.json").read_text("utf-8"))
    if file_hash(directory / "reps.npy") != index_manifest["reps_sha256"] or file_hash(directory / "page-map.jsonl") != index_manifest["page_map_sha256"]:
        raise ValueError("Index integrity check failed")
    pages = read_jsonl(directory / "page-map.jsonl")
    questions_path = WORKSPACE / "data/a_baseline/smoke/questions.jsonl"
    questions = read_jsonl(questions_path)
    if len(questions) != 10 or any(q["split"] != "smoke_dev" for q in questions):
        raise ValueError("This reproduction accepts only the original ten smoke_dev questions")
    matrix = np.load(directory / "reps.npy", allow_pickle=False)
    if matrix.shape != (len(pages), 2304):
        raise ValueError("Page matrix and mapping disagree")
    config = {"model": "openbmb/VisRAG-Ret", "model_revision": MODEL_REVISION, "top_k": args.top_k,
              "instruction": INSTRUCTION, "instruction_source": "upstream demo answer.py retrieve()",
              "index_manifest_sha256": file_hash(directory / "index-manifest.json"), "questions_sha256": file_hash(questions_path),
              "score": "torch.matmul(normalized_query, normalized_page.T); torch.topk", "query_batch_size": 1}
    write_json(output / "config.json", config)
    source_sha = snapshot(output)
    model, tokenizer, encode, runtime = model_runtime()
    device_matrix = torch.from_numpy(matrix).cuda()
    results, queries, scores = [], [], []
    for question in questions:
        # Only this question text is passed into the model, not its gold pages/answer.
        query, encoding_seconds, peak = call_encoder(model, tokenizer, encode, [INSTRUCTION + question["question"]])
        torch.cuda.synchronize()
        started = time.perf_counter()
        similarities = torch.matmul(torch.from_numpy(query).cuda(), device_matrix.T)
        values, ids = torch.topk(similarities, k=args.top_k)
        torch.cuda.synchronize()
        ranking_seconds = time.perf_counter() - started
        ordered = [{**pages[index], "rank": rank, "score": score} for rank, (index, score) in enumerate(zip(ids[0].cpu().tolist(), values[0].cpu().tolist()), start=1)]
        row = {**question, "retrieved_pages": ordered, "retrieval_seconds": encoding_seconds + ranking_seconds,
               "query_encoding_seconds": encoding_seconds, "ranking_seconds": ranking_seconds,
               "retriever_model": "openbmb/VisRAG-Ret", "retriever_revision": MODEL_REVISION,
               "retriever_config_sha256": file_hash(output / "config.json"), "retriever_source_sha256": source_sha,
               "retrieval_peak_cuda_gib": peak}
        results.append(row)
        queries.append(query)
        scores.append(similarities.float().cpu().numpy())
        write_jsonl(output / "retrieval.jsonl", results)
        print(f"Retrieved {question['question_id']}: {[p['page_id'] for p in ordered]} {row['retrieval_seconds']:.3f}s", flush=True)
    np.save(output / "query_reps.npy", np.concatenate(queries), allow_pickle=False)
    np.save(output / "similarities.npy", np.concatenate(scores), allow_pickle=False)
    write_jsonl(output / "query-map.jsonl", [{"question_id": q["question_id"], "vector_row": i, "model_input": INSTRUCTION + q["question"]} for i, q in enumerate(questions)])
    report = {"created_at": timestamp(), "questions": len(results), "runtime": runtime,
              "retrieval_sha256": file_hash(output / "retrieval.jsonl"), "query_reps_sha256": file_hash(output / "query_reps.npy"),
              "similarities_sha256": file_hash(output / "similarities.npy"), "source_sha256": source_sha,
              "index_directory": str(directory.relative_to(WORKSPACE)), "config_sha256": file_hash(output / "config.json")}
    write_json(output / "retrieval-manifest.json", report)
    operation("原版问题编码与视觉 Top-k 检索", "用上游 demo 原样 instruction 编码 10 个问题；读取缓存页面向量，torch 点积后 Top-k；不使用 OCR 文本或 gold 参与排序。", f"{len(results)} 题视觉召回已保存；问题向量、得分矩阵与映射可离线复查。", [output / "retrieval.jsonl", output / "query_reps.npy", output / "similarities.npy"], "页面向量与映射哈希匹配；只有 smoke_dev 参与；生成将在此进程退出后由另一个环境运行。")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("index")
    p.add_argument("--index-name", default="smoke-dpi200-v1")
    p.add_argument("--dpi", type=int, default=200)
    p.set_defaults(function=index)
    p = commands.add_parser("retrieve")
    p.add_argument("--index-name", default="smoke-dpi200-v1")
    p.add_argument("--retrieval-name", default="smoke-top2-v1")
    p.add_argument("--top-k", type=int, default=2)
    p.set_defaults(function=retrieve)
    args = parser.parse_args()
    operation(f"开始 VisRAG-Ret {args.command}", f"独立进程 {sys.executable} retrieve.py {' '.join(sys.argv[1:])}", "开始执行，后续记录真实结果。", status="进行中")
    try:
        args.function(args)
    except Exception as error:
        key = issue(f"原版 VisRAG-Ret {args.command} 失败：{type(error).__name__}: {error}", "检查具体异常、依赖、显存或索引配置；已保存中间文件。", "只修改独立复现环境或新适配模块；保留模型源码及旧结果，修复后以明确配置重跑。")
        operation(f"VisRAG-Ret {args.command} 失败", "原版编码与索引/检索流程", str(error), status="失败，待修复", related_issues=key)
        raise


if __name__ == "__main__":
    main()

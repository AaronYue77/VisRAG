"""Ask an unlabelled question; reuse the same RAG without a gold answer or Agent."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "rag_common"))
from common import WORKSPACE, timestamp, operation, issue, file_hash, write_json, write_jsonl
from audit import append_question, export_existing_runs
from retrieval import Retriever, EMBED_REVISION
from run import EcommerceReader


def main(args):
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        raise ValueError("Use an ASCII run-name")
    output = WORKSPACE / "reports/ecommerce_rag" / args.run_name
    if output.exists():
        raise ValueError("Preserve existing runs; use a new run-name")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    digest = hashlib.sha256()
    sources = {p.name: p for p in HERE.glob("*.py")}
    sources.update({"shared_" + p.name: p for p in (HERE.parent / "rag_common").glob("*.py")})
    for name, path in sorted(sources.items()):
        data = path.read_bytes()
        (snapshot / name).write_bytes(data)
        digest.update(name.encode()); digest.update(data)
    request = {"store_id": args.store, "as_of": args.as_of, "skus": args.sku}
    config = {"schema_version": "1.0", "suite": "ecommerce_interactive", "mode": args.mode, "top_k": 2,
              "source_sha256": digest.hexdigest(), "request": request,
              "embedding_revision": EMBED_REVISION, "index_manifest_sha256": file_hash(WORKSPACE / "data/ecommerce_rag/demo-v1/index-manifest.json"),
              "model": "Qwen/Qwen3.5-2B", "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
              "max_new_tokens": 160, "max_pixels_per_page": 786432, "agent_implemented": False,
              "live_business_tools_connected": False, "evaluation": "unlabelled; not scored"}
    write_json(output / "config.json", config)
    search = Retriever(args.mode).search(args.question, request)
    row = {"question_id": "interactive-01", "question": args.question, "request": request,
           "reference_answer": "未标注（交互问答，不计分）", "reference_pages": [], "answer_aliases": [],
           "split": "unlabelled", "run_name": args.run_name, "run_directory": output.relative_to(WORKSPACE).as_posix(),
           "timestamp": timestamp(), "status": "ok", "source_sha256": config["source_sha256"],
           "config_sha256": file_hash(output / "config.json"), "retrieved_pages": search["pages"],
           "retrieval_trace": search["trace"], "retrieval_seconds": search["seconds"],
           "model": None, "model_revision": None, "generation_seconds": 0.0, "metrics": {}}
    if search["plan"]["action"] == "retrieve":
        row.update(model=config["model"], model_revision=config["model_revision"])
        row.update(EcommerceReader().read(args.question, request, search["pages"], output / "images"))
    else:
        row.update(action=search["plan"]["action"], answer=search["plan"]["answer"], citations=[],
                   answer_source="rule_router", required_tool=search["plan"].get("required_tool"))
    write_jsonl(output / "results.jsonl", [row])
    append_question(output, row)
    export_existing_runs()
    operation("电商 RAG 交互问答", "同一 Retriever 与 EcommerceReader；请求只含问题、型号、店铺和日期；没有输入参考答案或正确证据页。",
              f"动作={row['action']}；答案={row['answer']}；未标注，不计准确率。", [output / "逐题结果.txt", output / "results.jsonl"], "逐题字段与配置、代码快照已保存；没有连接真实业务接口。")
    print(json.dumps({"action": row["action"], "answer": row["answer"], "citations": row["citations"], "record": str(output / "逐题结果.txt")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--sku", action="append", default=[])
    p.add_argument("--store", default="demo_store")
    p.add_argument("--as-of", default="2026-11-01")
    p.add_argument("--mode", choices=["bm25", "hybrid"], default="hybrid")
    p.add_argument("--run-name", required=True)
    try:
        main(p.parse_args())
    except Exception as error:
        key = issue(f"电商交互问答失败：{type(error).__name__}: {error}", "查看输入、索引和本地模型错误。", "保留失败目录，修复后使用新运行名。")
        operation("电商交互问答失败", "ask.py", str(error), status="失败", related_issues=key)
        raise

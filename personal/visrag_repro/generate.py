"""Read cached visual retrieval and answer with Qwen in the separate A environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "a_baseline"
sys.path.insert(0, str(BASE))
from common import WORKSPACE, REPO, read_jsonl, write_json, file_hash, timestamp, operation, issue
from core import VLMReader, retrieval_metrics, answer_metrics
from audit import append_question
from run import summarize


def snapshot(output):
    target = output / "source_snapshot"
    target.mkdir()
    sources = {p.name: p for p in BASE.glob("*.py")}
    sources.update({"repro_" + p.name: p for p in HERE.glob("*.py")})
    digest = hashlib.sha256()
    for name, path in sorted(sources.items()):
        data = path.read_bytes()
        (target / name).write_bytes(data)
        digest.update(name.encode())
        digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-name", default="smoke-top2-v1")
    parser.add_argument("--run-name", default="smoke10-visrag-ret-v1")
    args = parser.parse_args()
    for value in (args.retrieval_name, args.run_name):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
            raise ValueError("Names must use ASCII letters, digits, hyphens or underscores")
    source = WORKSPACE / "data/visrag_repro" / args.retrieval_name
    manifest = json.loads((source / "retrieval-manifest.json").read_text("utf-8"))
    if file_hash(source / "retrieval.jsonl") != manifest["retrieval_sha256"]:
        raise ValueError("Cached visual retrieval changed")
    records = read_jsonl(source / "retrieval.jsonl")
    canonical = read_jsonl(WORKSPACE / "data/a_baseline/smoke/questions.jsonl")
    for q, r in zip(canonical, records):
        for field in ("question_id", "question", "reference_answer", "reference_pages", "answer_aliases"):
            if q[field] != r[field]:
                raise ValueError(f"Question or gold label changed: {field}")
    if len(records) != 10 or len(canonical) != 10:
        raise ValueError("Must use the exact original ten questions")
    output = WORKSPACE / "reports/a_baseline" / args.run_name
    if output.exists():
        raise ValueError("Run directory exists; preserve old output and choose a new run-name")
    output.mkdir(parents=True)
    source_sha = snapshot(output)
    retrieval_config = json.loads((source / "config.json").read_text("utf-8"))
    names = ["torch", "transformers", "PyMuPDF", "Pillow"]
    config = {"schema_version": "1.0", "suite": "smoke", "split": "smoke_dev", "planned_questions": 10,
              "question_ids": [r["question_id"] for r in records], "top_k": retrieval_config["top_k"], "retrieval_only": False,
              "max_pixels_per_page": 786432, "max_new_tokens": 96,
              "model": "Qwen/Qwen3.5-2B", "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
              "model_config_sha256": file_hash(WORKSPACE / "models/Qwen3.5-2B/config.json"),
              "generation": {"do_sample": False, "enable_thinking": False, "dtype": "bfloat16", "attention": "sdpa"},
              "questions_sha256": file_hash(WORKSPACE / "data/a_baseline/smoke/questions.jsonl"),
              "source_sha256": source_sha, "packages": {name: importlib.metadata.version(name) for name in names},
              "retriever": {"display_name": "VisRAG-Ret 视觉向量检索", **retrieval_config},
              "retrieval_result_sha256": manifest["retrieval_sha256"], "retrieval_directory": str(source.relative_to(WORKSPACE)),
              "reader_input_policy": "same original A VLMReader, prompt, model, pixel and decoding settings; only retrieved pages differ"}
    write_json(output / "config.json", config)
    gpu_before = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip()
    write_json(output / "provenance.json", {"started_at": timestamp(), "python_executable": sys.executable,
               "process_id": os.getpid(), "git_head_before_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
               "source_sha256": source_sha, "config_sha256": file_hash(output / "config.json"),
               "gpu_memory_used_mib_before_reader": gpu_before,
               "retrieval_process_id": manifest["runtime"]["process_id"],
               "process_policy": "Retrieval program completed and exited before launching this generation program."})
    operation("开始视觉检索结果的独立 VLM 问答", "读取已结束检索进程保存的 Top-k；在原 Qwen 环境加载同一 Reader，保持原题、提示、输入预算和解码设置。", f"生成进程 {os.getpid()}，加载前 GPU 已用 {gpu_before} MiB。", [output], "检索文件哈希及 10 道题所有 gold 字段与原版对照一致。", status="进行中")
    reader = VLMReader(WORKSPACE / "models/Qwen3.5-2B", max_new_tokens=96)
    with (output / "results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for number, record in enumerate(records, start=1):
            row = {**record, "run_name": args.run_name, "timestamp": timestamp(), "schema_version": "1.0", "status": "ok",
                   "model": config["model"], "model_revision": config["model_revision"],
                   "config_sha256": file_hash(output / "config.json"), "source_sha256": source_sha,
                   "metrics": retrieval_metrics([p["page_id"] for p in record["retrieved_pages"]], record["reference_pages"])}
            try:
                row.update(reader.read(record["question"], record["retrieved_pages"], 786432, output / "images"))
                row["metrics"].update(answer_metrics(row["answer"], row["answer_aliases"]))
                row["metrics"]["citation_gold_hit"] = float(bool(set(row["citations"]) & set(row["reference_pages"])))
                row["metrics"]["json_parse_ok"] = float(row["parse_ok"])
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                row["issue_id"] = issue(f"视觉检索后的问题 {row['question_id']} 生成失败：{row['error']}", "保留该题和已缓存页面，检查本地 VLM 输入与显存。", "修复后使用新 run-name，不覆盖本次原始输出。")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            append_question(output, row)
            print(f"{number}/10 {row['question_id']} hit={row['metrics']['hit_at_k']} answer={row.get('answer')} gen={row.get('generation_seconds', 0):.2f}s", flush=True)
    summarize(output)
    operation("完成同批 10 题视觉检索与本地 VLM 问答", "独立旧版环境完成 VisRAG-Ret 编码/检索，退出后由 Qwen 环境生成；统一指标和 TXT/JSONL 保存。", (output / "summary.json").read_text("utf-8"), [output / "逐题结果.txt", output / "results.jsonl", output / "运行报告.md"], "全部题目保留，原 BM25 运行和封存测试集未覆盖；结果需与基线逐题对照。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        key = issue(f"视觉检索结果生成流程失败：{type(error).__name__}: {error}", "检查配置、检索缓存、模型或 GPU 异常。", "保留旧输出，修复后使用独立的新 run-name。")
        operation("视觉检索结果生成流程失败", "generate.py 缓存视觉检索 -> 本地 Qwen", str(error), status="失败，待修复", related_issues=key)
        raise

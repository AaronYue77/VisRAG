"""A-stage CLI: build page corpus, retrieve, generate, persist, evaluate."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import shutil
import time

from common import WORKSPACE, CODE, REPO, read_jsonl, write_json, file_hash, timestamp, issue, operation
from audit import append_question, export_existing_runs
from core import build_pages, verify_questions, TextRetriever, VLMReader, retrieval_metrics, answer_metrics


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(CODE.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def suite_root(suite):
    return WORKSPACE / "data/a_baseline" / suite


def index(args):
    root = suite_root(args.suite)
    documents = json.loads((root / "documents.json").read_text("utf-8"))
    pages = build_pages(documents, root / "pages.jsonl", ocr_fallback=not args.no_ocr)
    questions = read_jsonl(root / "questions.jsonl")
    verify_questions(questions, pages)
    report = {"created_at": timestamp(), "documents": len(documents), "pages": len(pages),
              "questions": len(questions), "text_source_counts": dict(Counter(x["text_source"] for x in pages)),
              "empty_pages": sum(not x["text"].strip() for x in pages),
              "split_counts": dict(Counter(x["split"] for x in questions)),
              "page_number_validation": "Every labeled evidence page exists in actual PDFs.",
              "document_split_validation": "No document appears in more than one question split.",
              "pages_sha256": file_hash(root / "pages.jsonl"), "questions_sha256": file_hash(root / "questions.jsonl")}
    write_json(root / "index-report.json", report)
    operation(f"构建 {args.suite} 页面语料", "校验 PDF SHA256；逐页原生文本抽取，不足 20 字符时 OCR 回退；检查证据页存在与文档分区。", json.dumps(report, ensure_ascii=False), [root / "pages.jsonl", root / "index-report.json"], "参考页均存在且同一文档不会跨问题分区；空文本页仍完整保留。")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def summarize(run_dir: Path):
    records = read_jsonl(run_dir / "results.jsonl")
    config = json.loads((run_dir / "config.json").read_text("utf-8"))
    completed = [x for x in records if x["status"] == "ok"]
    values = {key: statistics.mean(x["metrics"][key] for x in records) for key in ("hit_at_k", "recall_at_k", "reciprocal_rank")} if records else {}
    if not config["retrieval_only"]:
        # All attempted cases form the denominator; errors count as zero answer score.
        for key in ("normalized_em", "token_f1", "citation_gold_hit", "json_parse_ok"):
            values[key] = statistics.mean(x.get("metrics", {}).get(key, 0.0) for x in records) if records else 0.0
    elapsed = [x.get("generation_seconds", 0.0) for x in completed]
    summary = {"attempted": len(records), "completed": len(completed), "errors": len(records) - len(completed),
               "total_planned": config["planned_questions"], "metrics": values,
               "mean_generation_seconds": statistics.mean(elapsed) if elapsed else None,
               "peak_cuda_gib": max((x.get("peak_cuda_gib", 0.0) for x in records), default=0),
               "score_note": "Lexical EM/F1 and citation-page overlap are diagnostics, not verified semantic correctness or proof of groundedness.",
               "run_config_sha256": file_hash(run_dir / "config.json")}
    write_json(run_dir / "summary.json", summary)
    retriever_label = config.get("retriever", {}).get("display_name", "BM25")
    lines = ["# A 阶段运行检查", "", f"问题数：{len(records)}；成功执行：{len(completed)}；异常：{len(records)-len(completed)}。", "",
             "本报告保留检索失败和回答失败。EM/F1 是文本匹配指标；引用命中只表示页号重合，不证明答案被证据支持。", "",
             f"模式：{'仅检索' if config['retrieval_only'] else retriever_label + ' → 页面图像 → 本地 VLM'}；套件：{config['suite']}；分区：{config['split']}。", "",
             "| ID | 参考答案 | 模型答案 | 参考证据 | 召回页面 | 检索命中 | EM | 生成秒 |", "|---|---|---|---|---|---|---|---|"]
    clean = lambda value: str(value).replace("|", " / ").replace("\n", " ")
    for row in records:
        lines.append("| " + " | ".join(map(clean, [row["question_id"], row["reference_answer"], row.get("answer", row.get("error", "未生成")),
                     ", ".join(row["reference_pages"]), ", ".join(x["page_id"] for x in row["retrieved_pages"]),
                     row["metrics"]["hit_at_k"], row["metrics"].get("normalized_em", "—"), round(row.get("generation_seconds", 0), 2)])) + " |")
    lines.extend(["", "## 如何检查每道题", "", "完整字段在 results.jsonl；输入页面图片在 images/。先看证据页是否召回，再核对图片是否支持模型答案。", "",
                  "检索未命中只能说明标注证据页未进入 Top-k；其他页面可能重复含有答案。检索命中但 EM=0 也可能只是措辞差异，需人工复核。", ""])
    for row in records:
        lines.extend([f"### {row['question_id']}", "", row.get("question_zh", row["question"]), "", f"原问题：{row['question']}", "",
                      f"输出：`{clean(row.get('raw_output', '仅检索'))}`", "", f"输出引用：{row.get('citations', [])}", ""])
        for path in row.get("image_paths", []):
            import os
            link = Path(os.path.relpath(WORKSPACE / path, run_dir)).as_posix()
            lines.append(f"[输入页面：{Path(path).name}]({link})  ")
        lines.append("")
    (run_dir / "运行报告.md").write_text("\n".join(lines), "utf-8")
    export_existing_runs()
    operation(f"汇总运行 {run_dir.name}", "读取逐题 JSONL，宏平均检索/答案指标；导出逐题 TXT、汇总 TXT 和可阅读报告。", json.dumps(summary, ensure_ascii=False), [run_dir / "逐题结果.txt", run_dir / "summary.json", run_dir / "运行报告.md"], "记录数量与计划题数对照；执行错误保留在分母中，不删除失败题。")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run(args):
    root = suite_root(args.suite)
    pages = read_jsonl(root / "pages.jsonl")
    all_questions = read_jsonl(root / "questions.jsonl")
    verify_questions(all_questions, pages)
    if args.split == "test" and not args.final_test:
        raise ValueError("Final test is locked: freeze A/B/C configuration first, then pass --final-test explicitly.")
    questions = [x for x in all_questions if x["split"] == args.split]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("Selected split is empty")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        raise ValueError("run-name must contain only ASCII letters, digits, hyphens or underscores")
    output = WORKSPACE / "reports/a_baseline" / args.run_name
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "PyMuPDF", "rank-bm25", "Pillow")}
    config = {"schema_version": "1.0", "suite": args.suite, "split": args.split, "planned_questions": len(questions),
              "question_ids": [x["question_id"] for x in questions], "top_k": args.top_k, "bm25": {"k1": 1.5, "b": .75},
              "retrieval_only": args.retrieval_only, "max_pixels_per_page": args.max_pixels, "max_new_tokens": args.max_new_tokens,
              "model": "Qwen/Qwen3.5-2B", "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
              "model_config_sha256": file_hash(WORKSPACE / "models/Qwen3.5-2B/config.json"),
              "generation": {"do_sample": False, "enable_thinking": False, "dtype": "bfloat16", "attention": "sdpa"},
              "questions_sha256": file_hash(root / "questions.jsonl"), "pages_sha256": file_hash(root / "pages.jsonl"),
              "source_sha256": source_hash(), "packages": packages}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text("utf-8")) != config:
        raise ValueError("Existing run has a different configuration/source hash; choose a new --run-name.")
    if not config_path.exists():
        write_json(config_path, config)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for path in CODE.glob("*.py"):
            shutil.copyfile(path, snapshot / path.name)
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()
        write_json(output / "provenance.json", {"started_at": timestamp(), "git_head_before_run": git_head, "source_sha256": config["source_sha256"], "config_sha256": file_hash(config_path)})
    result_path = output / "results.jsonl"
    previous = read_jsonl(result_path) if result_path.exists() else []
    done = {x["question_id"] for x in previous}
    pending = [x for x in questions if x["question_id"] not in done]
    if not pending:
        summarize(output)
        return
    retriever = TextRetriever(pages)
    reader = None if args.retrieval_only else VLMReader(WORKSPACE / "models/Qwen3.5-2B", args.max_new_tokens)
    if reader:
        print(f"Model loaded in {reader.load_seconds:.2f}s", flush=True)
    with result_path.open("a", encoding="utf-8", newline="\n") as stream:
        for position, question in enumerate(pending, start=1):
            started = time.perf_counter()
            retrieved = retriever.search(question["question"], args.top_k)
            retrieve_seconds = time.perf_counter() - started
            evidence = [{k: page[k] for k in ("page_id", "doc_id", "page", "rank", "score", "text_source", "pdf_sha256")} for page in retrieved]
            metrics = retrieval_metrics([x["page_id"] for x in retrieved], question["reference_pages"])
            row = {**question, "run_name": args.run_name, "timestamp": timestamp(), "schema_version": "1.0", "status": "ok",
                   "retrieved_pages": evidence, "retrieval_seconds": retrieve_seconds, "metrics": metrics,
                   "model": config["model"] if reader else None, "model_revision": config["model_revision"] if reader else None,
                   "config_sha256": file_hash(config_path), "source_sha256": config["source_sha256"]}
            try:
                if reader:
                    row.update(reader.read(question["question"], retrieved, args.max_pixels, output / "images"))
                    metrics.update(answer_metrics(row["answer"], question["answer_aliases"]))
                    metrics["citation_gold_hit"] = float(bool(set(row["citations"]) & set(question["reference_pages"])))
                    metrics["json_parse_ok"] = float(row["parse_ok"])
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                row["issue_id"] = issue(f"问题 {question['question_id']} 推理失败：{row['error']}", "见 results.jsonl 与终端异常。", "保留失败记录；检查显存、输入尺寸与模型配置，以新 run-name 重跑修正版。")
                if reader:
                    import torch
                    torch.cuda.empty_cache()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            append_question(output, row)
            print(f"{position}/{len(pending)} {row['question_id']} status={row['status']} hit={metrics['hit_at_k']} answer={row.get('answer', 'N/A')} gen={row.get('generation_seconds', 0):.2f}s", flush=True)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("index")
    p.add_argument("--suite", choices=["smoke", "vidoseek"], default="smoke")
    p.add_argument("--no-ocr", action="store_true")
    p.set_defaults(function=index)
    p = commands.add_parser("run")
    p.add_argument("--suite", choices=["smoke", "vidoseek"], default="smoke")
    p.add_argument("--split", choices=["smoke_dev", "dev", "test"], default="smoke_dev")
    p.add_argument("--run-name", required=True)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=786432)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--limit", type=int)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--final-test", action="store_true")
    p.set_defaults(function=run)
    args = parser.parse_args()
    method = f"python run.py {' '.join(__import__('sys').argv[1:])}；具体算法、模型、参数见运行 config.json。"
    operation(f"开始 {args.command}", method, "开始执行，最终结果见后续完成或失败条目。", status="进行中")
    try:
        args.function(args)
        operation(f"完成 {args.command}", method, "命令正常返回；指标、计数与数据校验见产物。", [suite_root(args.suite)], "命令执行无异常；run 的逐题与汇总结果另行记录。")
    except Exception as error:
        key = issue(f"命令 {args.command} 失败：{type(error).__name__}: {error}", "根据异常检查数据、参数、依赖或资源。", "修复后重跑；不同配置使用新 run-name，保留旧记录。")
        operation(f"失败 {args.command}", method, f"{type(error).__name__}: {error}", status="失败，待修复", related_issues=key)
        raise


if __name__ == "__main__":
    main()

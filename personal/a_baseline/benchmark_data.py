"""Pinned ViDoSeek subset; document-disjoint development/final-test partitions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from common import WORKSPACE, CODE, write_json, write_jsonl, file_hash, issue, operation
from download_verified import download_verified

SEED = "visrag-personal-a-v1-20261031"


def key(value: str) -> str:
    return hashlib.sha256((SEED + value).encode()).hexdigest()


def main():
    root = WORKSPACE / "data/a_baseline/vidoseek"
    info = json.loads((root / "upstream.json").read_text("utf-8"))
    archive_info = next(x for x in info["siblings"] if x["rfilename"] == "vidoseek_pdf_document.zip")
    archive = WORKSPACE / "downloads/vidoseek_pdf_document.zip"
    canonical = f"https://huggingface.co/datasets/{info['id']}/resolve/{info['sha']}/{archive_info['rfilename']}"
    download_url = canonical.replace("https://huggingface.co", "https://hf-mirror.com")
    download_verified(download_url, archive, archive_info["size"], archive_info["lfs"]["sha256"], WORKSPACE / "downloads/vidoseek-parts")
    examples = json.loads((root / "vidoseek.json").read_text("utf-8"))["examples"]
    docs = sorted({x["meta_info"]["file_name"] for x in examples}, key=key)
    doc_split = {name: ("dev" if i % 2 == 0 else "test") for i, name in enumerate(docs)}
    selected = []
    for split in ("dev", "test"):
        for kind, quota in {"2d_layout": 14, "table": 12, "chart": 12, "text": 12}.items():
            candidates = [x for x in examples if doc_split[x["meta_info"]["file_name"]] == split and x["meta_info"]["source_type"] == kind]
            candidates.sort(key=lambda x: key(str(x["uid"])))
            if len(candidates) < quota:
                raise ValueError(f"Insufficient {split}/{kind} examples")
            selected.extend(candidates[:quota])
    needed = sorted({x["meta_info"]["file_name"] for x in selected})
    destination = root / "documents"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as source:
        by_name = {}
        for name in source.namelist():
            basename = Path(name).name
            if basename in needed:
                if basename in by_name:
                    raise ValueError(f"Ambiguous PDF basename: {basename}")
                by_name[basename] = name
        for name in needed:
            path = destination / name
            content = source.read(by_name[name])  # zipfile checks this member's CRC
            if not content.startswith(b"%PDF"):
                raise ValueError(f"Not a PDF: {name}")
            path.write_bytes(content)  # explicitly selected basenames, no extractall/zip-slip
            records.append({"doc_id": Path(name).stem, "title": name,
                            "pdf": str(path.relative_to(WORKSPACE)), "sha256": file_hash(path),
                            "source_archive": archive_info["rfilename"]})
    rows = []
    for x in selected:
        meta = x["meta_info"]
        doc = Path(meta["file_name"]).stem
        rows.append({"question_id": f"vidoseek-{x['uid']}", "question": x["query"],
                     "reference_answer": str(x["reference_answer"]), "answer_aliases": [str(x["reference_answer"])],
                     "reference_pages": [f"{doc}:p{int(p):04d}" for p in meta["reference_page"]],
                     "document_ids": [doc], "split": doc_split[meta["file_name"]],
                     "question_type": meta["source_type"], "query_type": meta["query_type"],
                     "gold_origin": "ViDoSeek author-provided annotation; not all manually audited",
                     "page_numbering": "one_based_pdf_page"})
    rows.sort(key=lambda x: (x["split"], x["question_id"]))
    write_jsonl(root / "questions.jsonl", rows)
    write_json(root / "documents.json", records)
    protocol = {"seed": SEED, "dataset_repo": info["id"], "revision": info["sha"],
                "annotation_sha256": file_hash(root / "vidoseek.json"), "archive_sha256": file_hash(archive),
                "archive_url": canonical, "download_url": download_url,
                "question_count": len(rows), "document_count": len(records),
                "split_counts": dict(Counter(x["split"] for x in rows)),
                "type_counts": dict(Counter(x["question_type"] for x in rows)),
                "split_rule": "Stable SHA256 ordering of all document names; alternate dev/test, then deterministic stratified question selection.",
                "corpus_scope": "Only documents appearing in the selected 100 questions, shared retrieval corpus. Not the full ViDoSeek corpus or official leaderboard.",
                "test_policy": "Do not inspect test predictions or tune on test; run final test only after A/B/C configuration is fixed.",
                "future_training_policy": "Future SFT documents must exclude every dev/test document ID; this suite has no training split.",
                "page_numbering": "One-based physical PDF page; validated against actual PDF page counts by index command."}
    write_json(root / "protocol.json", protocol)
    # Publish reproducible selection, not redistributed PDFs or all source annotations.
    write_json(CODE / "data/vidoseek-selection.json", {"protocol": protocol, "selected": [{"question_id": x["question_id"], "split": x["split"]} for x in rows]})
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    operation("准备固定公开评测子集", "分块下载并校验压缩包 SHA256；安全读取选中 ZIP 成员；按文档哈希分区后按题型固定抽样。", f"{len(rows)} 题、{len(records)} 份文档，dev/test 各 50 题。", [root / "questions.jsonl", root / "documents.json", root / "protocol.json"], "压缩包哈希与成员 CRC 通过；页号和跨文档分区由 index 命令继续校验。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        key = issue(f"ViDoSeek 准备失败：{type(error).__name__}: {error}", "按错误检查下载、压缩包和标注。", "保留分块文件，修复后重跑 benchmark_data.py。")
        operation("公开评测子集准备失败", "benchmark_data.py 下载、抽样与解压", str(error), status="失败，待修复", related_issues=key)
        raise

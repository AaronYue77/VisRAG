"""Small technical documents and authoritative public annotation download."""
import json
import hashlib
from common import WORKSPACE, fetch, file_hash, write_json, issue


def main() -> None:
    target = WORKSPACE / "data/a_baseline"
    docs = target / "smoke/documents"
    docs.mkdir(parents=True, exist_ok=True)
    sources = [
        ("attention", "Attention Is All You Need", "https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf", "attention.pdf"),
        ("clip", "Learning Transferable Visual Models From Natural Language Supervision", "https://proceedings.mlr.press/v139/radford21a/radford21a.pdf", "clip.pdf"),
    ]
    records = []
    for doc_id, title, url, name in sources:
        path = docs / name
        if not path.exists():
            content = fetch(url)
            if not content.startswith(b"%PDF"):
                raise RuntimeError(f"Not a PDF: {url}")
            path.write_bytes(content)
        records.append({"doc_id": doc_id, "title": title, "pdf": str(path.relative_to(WORKSPACE)), "source_url": url, "sha256": file_hash(path)})
        print(f"Document ready: {doc_id}", flush=True)
    write_json(target / "smoke/documents.json", records)
    metadata = target / "vidoseek/upstream.json"
    if metadata.exists():
        # Preserve the initially selected revision instead of silently updating the benchmark.
        info = json.loads(metadata.read_text("utf-8"))
    else:
        info = json.loads(fetch("https://huggingface.co/api/datasets/autumncc/ViDoSeek?blobs=true"))
        write_json(metadata, info)
    revision = info["sha"]
    url = f"https://huggingface.co/datasets/autumncc/ViDoSeek/resolve/{revision}/vidoseek.json"
    path = target / "vidoseek/vidoseek.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            content = fetch(url, retries=2)
        except Exception:
            content = fetch(url.replace("https://huggingface.co", "https://hf-mirror.com"))
        json.loads(content)
        path.write_bytes(content)
    expected = next(item for item in info["siblings"] if item["rfilename"] == "vidoseek.json")
    if expected.get("lfs") and file_hash(path) != expected["lfs"]["sha256"]:
        raise RuntimeError("ViDoSeek annotation SHA256 mismatch")
    if not expected.get("lfs"):
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != expected["blobId"]:
            raise RuntimeError("ViDoSeek annotation Git blob hash mismatch")
    write_json(target / "sources.json", {"documents": records, "annotation": {"repo": info["id"], "revision": revision, "url": url, "sha256": file_hash(path)}})
    print("ViDoSeek annotation ready", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        issue(f"数据准备失败：{type(error).__name__}: {error}", "查看具体网络响应或数据校验异常。", "保留已下载文件，修正原因后重新运行 download_data.py。")
        raise

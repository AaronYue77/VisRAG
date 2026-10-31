from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import ssl
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("DOC_VQA_WORKSPACE", REPO.parent.parent)).resolve()
CODE = Path(__file__).resolve().parent
ISSUES = CODE / "docs/问题与解决记录.txt"


def configure_environment() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    paths = {
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_XET_CACHE": "cache/huggingface/xet", "TORCH_HOME": "cache/torch",
        "TORCH_EXTENSIONS_DIR": "cache/torch-extensions", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "TRITON_CACHE_DIR": "cache/triton", "CUDA_CACHE_PATH": "cache/cuda",
        "TEMP": "cache/tmp", "TMP": "cache/tmp",
    }
    for key, value in paths.items():
        os.environ[key] = str(WORKSPACE / value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    (WORKSPACE / "cache/tmp").mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    # PDF text can contain U+2028/U+0085 inside JSON strings. str.splitlines()
    # treats them as line breaks even though a JSONL boundary is a physical LF.
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def issue(symptom: str, cause: str, solution: str, verification: str = "待验证", status: str = "待解决") -> str:
    text = ISSUES.read_text("utf-8") if ISSUES.exists() else ""
    numbers = [int(value) for value in re.findall(r"ISSUE-(\d+)", text)]
    key = f"ISSUE-{max(numbers, default=0) + 1:03d}"
    ISSUES.parent.mkdir(parents=True, exist_ok=True)
    with ISSUES.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n[{timestamp()}] {key}\n现象：{symptom}\n原因：{cause}\n解决：{solution}\n验证：{verification}\n状态：{status}\n")
    return key


def fetch(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "personal-visrag-a/1.0"})
            context = ssl.create_default_context()
            if attempt >= 2:
                context.set_ecdh_curve("prime256v1")
            if attempt % 2:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
            else:
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
            with opener.open(request, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


configure_environment()

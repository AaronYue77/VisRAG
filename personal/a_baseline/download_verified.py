"""Resumable ranged download with a final SHA256 check; temporary files stay on E:."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import ssl
import time
import urllib.request
import urllib.parse


def open_url(request, timeout=90):
    # Keep certificate validation; retry a direct connection if the system proxy resets TLS.
    context = ssl.create_default_context()
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    host = urllib.parse.urlsplit(url).hostname or ""
    if host.endswith("modelscope.cn"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
        return opener.open(request, timeout=timeout)
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    except (OSError, urllib.error.URLError):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
        return opener.open(request, timeout=timeout)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified(url: str, target: Path, size: int, expected_sha256: str, parts_root: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size and sha256(target) == expected_sha256:
        print(f"Already verified: {target.name}", flush=True)
        return
    parts_root.mkdir(parents=True, exist_ok=True)
    chunk = 32 * 1024 * 1024
    parts = [(number, start, min(start + chunk, size) - 1) for number, start in enumerate(range(0, size, chunk))]
    started = time.monotonic()

    def fetch_part(item):
        number, start, end = item
        part = parts_root / f"{number:05d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return expected
        for attempt in range(5):
            try:
                separator = "&" if "?" in url else "?"
                request = urllib.request.Request(
                    f"{url}{separator}download_part={number}",
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "doc-vqa-setup"},
                )
                temporary = part.with_suffix(".incomplete")
                with open_url(request, timeout=90) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Unexpected range: {response.status} {response.headers.get('Content-Range')}")
                    with temporary.open("wb") as output:
                        while data := response.read(1024 * 1024):
                            output.write(data)
                if temporary.stat().st_size != expected:
                    raise RuntimeError("Incomplete range")
                os.replace(temporary, part)
                return expected
            except Exception as error:
                print(f"Retry part {number}, attempt {attempt + 1}: {type(error).__name__}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_part, item) for item in parts]
        for count, future in enumerate(as_completed(futures), start=1):
            completed += future.result()
            if count == 1 or count % 8 == 0 or count == len(parts):
                elapsed = max(time.monotonic() - started, 0.1)
                print(f"{target.name}: {completed / size:.1%}, {completed / 2**20:.0f}/{size / 2**20:.0f} MiB, {completed / 2**20 / elapsed:.1f} MiB/s", flush=True)
    assembled = target.with_suffix(target.suffix + ".assembling")
    with assembled.open("wb") as output:
        for number, _, _ in parts:
            with (parts_root / f"{number:05d}.part").open("rb") as source:
                while data := source.read(4 * 1024 * 1024):
                    output.write(data)
    if assembled.stat().st_size != size or sha256(assembled) != expected_sha256:
        raise RuntimeError(f"SHA256 verification failed: {target.name}")
    os.replace(assembled, target)
    # Delete only the verified chunks this downloader created, inside its explicit project folder.
    checked_root = parts_root.resolve()
    for number, _, _ in parts:
        part = parts_root / f"{number:05d}.part"
        if part.resolve().parent != checked_root:
            raise RuntimeError("Unexpected temporary path")
        part.unlink()
    print(f"SHA256 verified: {target.name}", flush=True)

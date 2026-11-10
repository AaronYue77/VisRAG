"""Bootstrap the full ABO A/B application and the frozen C experiment on Kaggle."""
from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copytree
import subprocess
import sys
import traceback
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path("/kaggle/working/abo-rag")
CATALOG = "https://amazon-berkeley-objects.s3.amazonaws.com/"
FILES = {
    "archives/abo-listings.tar": "b7f7ceacb328fa5ab6e143b88e1f948443a877cfc95b67ff09c8ebabd50644e3",
    "images/metadata/images.csv.gz": "915263ef354d7a5d3503e899e610082f510c0011cccd5b4dc4f4c5efa86ef539",
    "README.md": None,
    "LICENSE-CC-BY-4.0.txt": None,
}
EXPANDED_LISTINGS_SHA = (
    "aa2c1884f7622600afcbbc70a53f6b3b6fe3f5d0306a2453424dcdd5785ecdd1",
    "286d62b963b166228d83bf32b654372a96966a340635e44d504bbb16953d7245",
    "fdd27bf555bc97052ba7c865ee48567c2f8a8c82e0454e0a8e9ecb84dbffccf0",
    "eccb27ea4032e435b1e4fdac33c24aca538ed874dea4a0f26d87e7013e2fa980",
    "bfc0c5874d4092a87685cba47900e8c5cbb1129e9a182dcb9ef7e73f204cd6c0",
    "c0ccc337b2ac49d5f13eb25e0ed25833e5bfb35bc9f2d76692458479011ffcb5",
    "9234be49ff2d93cd0c72aca97898d77e8b415c3e4c407d28c44f2bd70ecc9bc3",
    "1773f799f9936a2294394b79741daf4ec228d13672994d4c789a4a216ad28a42",
    "f0c304839cce4dd8fcd5eb4546062a28e3cface61a95d9c3674b3f5451931b5e",
    "8c408aef0b94df12aeb41bb8a7eb7e91e20f2ab9528b4af45371980ef412c4bf",
    "8af565ee5523ad99de8b59ce15bc90f8e0cd3052d5cdf297f80b117954bbf510",
    "cde2f40ce67c388ad53808a9949c7cc5eb46baa83c88e1fc14d510cab92bc6c1",
    "862bbd8ad7cd5112e93d68051038809936a28f8fe7611cbf0f0a202efab71629",
    "71c1583fa0a7354f8321c341eebcc3eda6a85ab9a50148ec15209fb76fad9d15",
    "6839d01c8ea125300f5473d4f7effbecb987d3f819e8d16a886a60bc1611ec6c",
    "329454b450ae6b5c0f6e6d4b92c084205f1e1b9fed3bceaa04987bfc6b0349f8",
)
MODEL = Path("/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1")
MODEL_REVISION = "kaggle:qwen-lm/qwen-3-vl/transformers/8b-instruct/1"
CODE = ROOT / "upstream/VisRAG/personal/ecommerce_rag"


def command(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        from shutil import copyfileobj

        partial = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            copyfileobj(response, stream)
        partial.replace(path)
    if expected:
        verify(path, expected)


def verify(path, expected):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError(f"ABO source checksum mismatch: {path}")


def setup_workspace():
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = Path("/kaggle/input")
    markers = list(inputs.rglob("ABO_8B_FORMAL.txt", recurse_symlinks=True))
    if markers:
        marker, = markers
        copytree(marker.parent, ROOT, dirs_exist_ok=True)
    else:
        bundle, = inputs.rglob("abo-8b-formal.zip", recurse_symlinks=True)
        with ZipFile(bundle) as archive:
            archive.extractall(ROOT)
    os.environ.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HOME="/tmp/abo-rag-hf",
                      ABO_QWEN8_MODEL_DIR=str(MODEL), ABO_QWEN8_REVISION=MODEL_REVISION,
                      DOC_VQA_WORKSPACE=str(ROOT))


def setup():
    setup_workspace()
    command("-c", "import torch; print(torch.__version__, torch.version.cuda, "
            "torch.cuda.get_device_name()); print((torch.ones(2, device='cuda') + 1).sum().item())")
    if not (MODEL / "config.json").is_file():
        raise FileNotFoundError("Attach official Kaggle model " + MODEL_REVISION)
    wheels = sorted(Path("/kaggle/input").rglob("*.whl", recurse_symlinks=True))
    if wheels:
        command("-m", "pip", "install", "--no-deps", "--no-index", *wheels)
    command("-c", "import bitsandbytes,pymupdf,peft,jieba,reportlab,transformers; "
            "print('offline packages available', transformers.__version__, peft.__version__)")


def setup_catalog():
    inputs = Path("/kaggle/input")
    sources = list(inputs.rglob("ABO_CLOUD_SOURCE.txt", recurse_symlinks=True))
    source = ROOT / "data/ecommerce_rag/abo-source"
    if sources:
        item, = sources
        source = item.parent / "data/ecommerce_rag/abo-source"
        images = Path("data/ecommerce_rag/abo-full-v1/images")
        copytree(item.parent / images, ROOT / images, dirs_exist_ok=True)
        copytree(item.parent / "fonts", ROOT / "fonts", dirs_exist_ok=True)
    if sources and not (source / "archives/abo-listings.tar").is_file():
        expanded = source / "archives/abo-listings/listings/metadata"
        for digit, expected in zip("0123456789abcdef", EXPANDED_LISTINGS_SHA, strict=True):
            verify(expanded / f"listings_{digit}.json", expected)
        verify(source / "images/metadata/images.csv",
               "f7d46958198e14de35f4e1a525b5ce978468587e8fcf7f0acc2a4974ed82e233")
        print("Verified 17 expanded Kaggle inputs against the official ABO source.", flush=True)
    else:
        for name, expected in FILES.items():
            download(CATALOG + name, source / name, expected)
    os.environ["ABO_SOURCE_DIR"] = str(source)
    standard = ROOT / "fonts/NotoSansSC.ttf"
    if not standard.exists():
        standard = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if not standard.exists():
        font_base = "https://raw.githubusercontent.com/google/fonts/2894aab31764f10f29c421bdfd2340d3b382d384/ofl/notosanssc/"
        standard = Path("/tmp/abo-fonts/NotoSansSC.ttf")
        download(font_base + "NotoSansSC%5Bwght%5D.ttf", standard)
        download(font_base + "OFL.txt", standard.with_name("OFL.txt"))
    os.environ["ABO_FONT_PATH"] = str(standard)
    command(CODE / "full_catalog.py")


def app_check():
    setup_catalog()
    sys.path.insert(0, str(CODE))
    from full_app import FullApplication

    report = ROOT / "reports/ecommerce_rag/cloud-b-8b"
    app = FullApplication("B", mode="bm25", report_root=report, model="8b")
    session = app.session()
    checks = []
    scenarios = [
        ("你好", "chat"),
        ("搜索 黑色双肩包", "search_results"),
        ("第一个", "selected"),
        ("品牌是什么？", "answer"),
        ("B076QBF37R@amazon.in 的照片上，正面拉链的亮色部分是什么颜色？", "answer"),
        ("换个话题，用一句话说说月亮为什么有圆缺", "chat"),
        ("能再举个例子吗？", "chat"),
        ("B076QBF37R@amazon.in 的高度是多少？", "answer"),
        ("Amazon.com 的退货政策是什么？", "answer"),
        ("帮我办理退款", "requires_tool"),
    ]
    try:
        for question, expected in scenarios:
            row = session.respond(question)
            passed = row["status"] == "ok" and row["action"] == expected
            if expected == "chat":
                passed = passed and not row["citations"] and not row["retrieved_pages"]
            if "拉链" in question:
                passed = passed and any(color in row["answer"].casefold() for color in ("蓝", "青", "blue"))
            if "高度" in question:
                passed = passed and "480 millimeters" in row["answer"]
            checks.append({"question": question, "expected": expected, "pass": passed, **row})
            print("B8", expected, row["action"], passed, row["strategy"], flush=True)
        (report / "验收.txt").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", "utf-8")
        return sum(r["pass"] for r in checks) == len(checks)
    finally:
        app.close()


def main():
    if "--app-check" in sys.argv:
        sys.exit(0 if app_check() else 2)
    if "--catalog-check" in sys.argv:
        setup_workspace()
        setup_catalog()
        command(CODE / "full_app.py", "check", "--stage", "B", "--mode", "bm25")
        command(CODE / "full_app.py", "search", "--question", "black backpack", "--mode", "bm25")
        command(CODE / "full_app.py", "ask", "--question", "B076QBF37R@amazon.in 的高度是多少？")
        return
    setup()
    status = {}
    try:
        # Keep the B model in a separate process so its CUDA context and cache exit before C.
        result = subprocess.run([sys.executable, __file__, "--app-check"], cwd=ROOT)
        status["b_full"] = {0: "pass", 2: "quality_failure"}.get(result.returncode, f"exit_{result.returncode}")
    except Exception as error:
        traceback.print_exc()
        status["b_full"] = type(error).__name__ + ": " + str(error)
    (ROOT / "cloud-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), "utf-8")
    command(CODE / "c8_experiment.py", "--model-dir", MODEL, "--run-name", "stage-c-8b-180-20261109", "--photos")
    status["c_experiment"] = "complete"
    (ROOT / "cloud-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), "utf-8")
    if (ROOT / "data/ecommerce_rag/abo-8b-supplement-v1/qa/protocol.json").is_file():
        previous = ROOT / "reports/ecommerce_rag/stage-c-8b-180-20261109"
        command(CODE / "c8_experiment.py", "--model-dir", MODEL, "--run-name", "stage-c-8b-supplement-20261110",
                "--parent-run", previous)
        status["c_supplement"] = "complete"
        (ROOT / "cloud-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()

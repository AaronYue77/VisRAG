"""Export only portable source, public ABO evidence and frozen evaluation files."""
import argparse
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PERSONAL = ROOT / "upstream/VisRAG/personal"
sys.path.insert(0, str(PERSONAL / "ecommerce_rag"))

from c8_protocol import DATA, load
from c8_photos import DATA as PHOTOS, load as load_photos


def export(destination, include_supplement=False):
    rows, _ = load()
    photos, _ = load_photos()
    files = {DATA / name for name in ("protocol.json", "questions.jsonl")}
    files.update(PHOTOS / name for name in ("protocol.json", "questions.jsonl", "annotations.jsonl", "sources.jsonl"))
    files.update(ROOT / page["pdf"] for row in rows + photos for page in row["retrieved_pages"])
    files.update({ROOT / "upstream/VisRAG/LICENSE", ROOT / "data/ecommerce_rag/abo-full-v1/ATTRIBUTION.txt",
                  ROOT / "data/ecommerce_rag/abo-full-v1/SOURCE-LICENSE-CC-BY-4.0.txt",
                  ROOT / "data/ecommerce_rag/policies-us-v1/clauses.json", ROOT / "scripts/kaggle_8b_project.py",
                  ROOT / "scripts/stage_c_8b.py"})
    for directory in (PERSONAL / "ecommerce_rag", PERSONAL / "rag_common"):
        files.update(path for path in directory.glob("*.py") if not path.name.startswith("test_"))
    supplement_count = 0
    if include_supplement:
        from c8_supplement import PHOTOS as EXTRA_PHOTOS, QA as EXTRA_QA, load as load_supplement

        extra, _ = load_supplement()
        supplement_count = len(extra)
        files.update(EXTRA_QA / name for name in ("questions.jsonl", "protocol.json"))
        files.update(EXTRA_PHOTOS / name for name in ("questions.jsonl", "protocol.json", "annotations.jsonl", "sources.jsonl"))
        files.update(ROOT / page["pdf"] for row in extra for page in row["retrieved_pages"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("ABO_8B_FORMAL.txt", "Private ABO fixture: 60 train, 25 dev, 180 test and 90 annotated photos.\n"
                        f"Frozen supplemental test: {supplement_count} cases.\n"
                        "ABO-derived evidence CC BY 4.0. Source license and attribution included.\n")
        for path in sorted(files):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Corrupt portable bundle")
    print(f"{destination}: {len(rows)} frozen QA + {len(photos)} photos + {supplement_count} supplement, "
          f"{destination.stat().st_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--with-supplement", action="store_true")
    args = parser.parse_args()
    export(args.destination, args.with_supplement)

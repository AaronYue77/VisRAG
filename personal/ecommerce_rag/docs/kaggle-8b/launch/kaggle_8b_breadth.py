"""Evaluate two pinned adapters on the 75 extra frozen official ABO photos."""
from pathlib import Path
import runpy
from zipfile import ZipFile


def main():
    inputs = Path("/kaggle/input")
    markers = list(inputs.rglob("ABO_8B_FORMAL.txt", recurse_symlinks=True))
    if markers:
        marker, = markers
        entry = marker.parent / "scripts/kaggle_8b_project.py"
    else:
        bundle, = inputs.rglob("abo-8b-formal.zip", recurse_symlinks=True)
        with ZipFile(bundle) as archive:
            entry = Path(archive.extract("scripts/kaggle_8b_project.py", "/tmp/abo-rag-bootstrap"))
    bootstrap = runpy.run_path(str(entry))
    bootstrap["setup"]()
    root = bootstrap["ROOT"]
    bootstrap["command"](bootstrap["CODE"] / "c8_visual_breadth.py", "run", "--model-dir", bootstrap["MODEL"],
                         "--prior-dir", root / "accepted-c60", "--candidate-dir", root / "candidate-c120")


if __name__ == "__main__":
    main()

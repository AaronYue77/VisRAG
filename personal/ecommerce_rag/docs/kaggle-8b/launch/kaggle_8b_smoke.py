"""Verify the current 8B code with the private frozen fixture on Kaggle."""
from pathlib import Path
import runpy


def main():
    inputs = Path("/kaggle/input")
    markers = list(inputs.rglob("ABO_8B_FORMAL.txt", recurse_symlinks=True))
    if markers:
        marker, = markers
        bootstrap = marker.parent / "scripts/kaggle_8b_project.py"
    else:
        from zipfile import ZipFile

        bundle, = inputs.rglob("abo-8b-formal.zip", recurse_symlinks=True)
        bootstrap = Path("/kaggle/working/kaggle_8b_project.py")
        with ZipFile(bundle) as archive:
            bootstrap.write_bytes(archive.read("scripts/kaggle_8b_project.py"))

    project = runpy.run_path(str(bootstrap))
    project["setup"]()
    command = project["command"]
    code = project["CODE"]
    root = project["ROOT"]
    command("-c", f"""
import sys
sys.path.insert(0, {str(code)!r})
from full_app import EvidenceReader, FullConversation
reader = EvidenceReader()
trace = []
answer = FullConversation(reader).reply("What is 2 plus 3? Reply with the digit only.", [], trace)
if answer.strip() != "5" or reader.load_count != 1 or not trace[-1]["model_executed"]:
    raise RuntimeError(f"Unexpected 8B chat smoke result: {{answer!r}}, {{trace!r}}")
from common import write_json
output = {str(root / "smoke-chat.json")!r}
from pathlib import Path
write_json(Path(output), {{"answer": answer, "trace": trace, "model_loads": reader.load_count}})
""")
    runner = "scripts/stage_c_8b.py"
    command(runner, "check-data")
    command(runner, "infer", "--nf4", "--suite", "double", "--limit", "1", "--run-name", "smoke-8b-infer")
    command(runner, "qlora-step", "--nf4", "--run-name", "smoke-8b-step")
    command(runner, "infer", "--nf4", "--suite", "double", "--limit", "1", "--run-name", "smoke-8b-reload",
            "--adapter", "reports/ecommerce_rag/smoke-8b-step/adapter")
    (root / "smoke-complete.txt").write_text("Open dialogue, two-page inference, QLoRA step and reload completed.\n",
                                            encoding="utf-8")


if __name__ == "__main__":
    main()

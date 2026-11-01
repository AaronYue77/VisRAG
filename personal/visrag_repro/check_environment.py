"""Verify the original-core inference environment without loading large weights."""
import sys
from pathlib import Path
import json
import importlib.metadata
import platform
import subprocess

BASE = Path(__file__).resolve().parents[1] / "rag_common"
sys.path.insert(0, str(BASE))
from common import WORKSPACE, REPO, file_hash, write_json, operation, issue, timestamp
from compat import load_tokenizer


def main():
    import torch
    import torchvision
    import transformers
    import timm
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer
    assert torch.__version__ == "2.1.2+cu121", torch.__version__
    assert transformers.__version__ == "4.40.2", transformers.__version__
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    tensor = torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(4, 4)
    matrix = (tensor @ tensor.T).float().cpu().numpy()
    assert np.isfinite(matrix).all()
    # Exercise the timm model symbol used by MiniCPMV, without allocating its weights.
    assert timm.is_model("vit_so400m_patch14_siglip_384.webli")
    model_dir = WORKSPACE / "models/VisRAG-Ret"
    tokenizer_test = None
    if (model_dir / "tokenizer.model").exists():
        tokenizer = load_tokenizer(model_dir)
        ids = tokenizer.encode("Represent this query for retrieving relevant document: What is shown?")
        assert ids and isinstance(tokenizer.im_start_id, int)
        tokenizer_test = {"tokens": len(ids), "image_start_id": tokenizer.im_start_id,
                          "io_compatibility": "Read verified SentencePiece protobuf bytes through Python; unchanged tokenizer rules."}
    names = ["torch", "torchvision", "transformers", "accelerate", "numpy", "Pillow", "sentencepiece", "protobuf", "tokenizers", "huggingface-hub", "safetensors", "timm", "einops", "PyMuPDF"]
    packages = {name: importlib.metadata.version(name) for name in names}
    original = REPO / "requirements.txt"
    report = {"checked_at": timestamp(), "python_executable": sys.executable, "python": platform.python_version(),
              "environment": str(WORKSPACE / ".venv-visrag"), "packages": packages,
              "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
              "bf16_supported": True, "gpu_matrix_test_finite": True, "tokenizer_test": tokenizer_test,
              "upstream_requirements_sha256": file_hash(original),
              "scope": "Original-core VisRAG-Ret image/query inference; generation stays in the existing Qwen environment.",
              "excluded_optional_dependencies": {"deepspeed": "training only", "pytrec_eval": "not used by this fixed smoke evaluation", "decord": "video decoder not used for PDF page images"}}
    target = WORKSPACE / "reports/visrag-repro"
    write_json(target / "environment.json", report)
    freeze = subprocess.check_output([str(WORKSPACE / ".tools/uv/uv.exe"), "pip", "freeze", "--python", sys.executable]).decode("utf-8")
    (target / "requirements.freeze.txt").write_text(freeze, "utf-8", newline="\n")
    (Path(__file__).parent / "requirements.freeze.txt").write_text(freeze, "utf-8", newline="\n")
    (target / "upstream-requirements.original.txt").write_bytes(original.read_bytes())
    operation("验证独立原版核心推理环境", "独立 Python 3.11；锁定原版 torch/torchvision/transformers 等核心版本，执行 CUDA BF16 矩阵、timm 模型注册与可用时的本地 tokenizer 检查；导出完整依赖锁。", json.dumps(report, ensure_ascii=False), [target / "environment.json", target / "requirements.freeze.txt"], "版本断言与 GPU 计算通过；原有 Qwen .venv 不变。仅省略与 PDF 图像推理无关的训练/视频/额外评测扩展。")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        key = issue(f"原版环境验证失败：{type(error).__name__}: {error}", "查看具体版本、导入或 CUDA 错误。", "只在 .venv-visrag 修正依赖，保留当前 .venv。")
        operation("原版环境验证失败", "check_environment.py 版本与 GPU 验证", str(error), status="失败，待修复", related_issues=key)
        raise

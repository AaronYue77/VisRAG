"""One pinned Qwen3-VL model shared by evidence reading and open dialogue."""
import os
from pathlib import Path
import time

from common import WORKSPACE


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = os.environ.get("ABO_QWEN8_REVISION", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")


class Qwen8Runtime:
    def __init__(self, model_dir=None, *, nf4=True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        directory = Path(model_dir or os.environ.get("ABO_QWEN8_MODEL_DIR") or
                         WORKSPACE / "models/Qwen3-VL-8B-Instruct")
        if not (directory / "config.json").is_file():
            raise FileNotFoundError("Qwen3-VL-8B snapshot missing. Configure ABO_QWEN8_MODEL_DIR or use the Kaggle launch script.")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL-8B requires an enabled CUDA accelerator")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        self.nf4 = nf4
        options = {"local_files_only": True, "dtype": self.dtype, "device_map": {"": 0},
                   "attn_implementation": "sdpa"}
        if nf4:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(directory, **options).eval()
        self.load_seconds = time.perf_counter() - started

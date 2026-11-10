"""Run an 8B fixed-evidence inference or one-step QLoRA check on a CUDA host."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/VisRAG/personal/ecommerce_rag"))

from c_protocol import DATA, load
from common import file_hash, timestamp, write_json


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = os.environ.get("ABO_QWEN8_REVISION", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")


def run(args):
    rows, protocol = load()
    if args.task == "check-data":
        print(f"{len(rows)} frozen questions, {MODEL_ID} revision {REVISION}")
        return
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    from audit import append_question
    from stage_a import Reader, snapshot
    from stage_c import PROMPT_C1, score, training_input
    if args.extended:
        from c_extended import load_extended

        rows, protocol = load_extended()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError("Stage the pinned 8B model snapshot before running")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if args.nf4 else None
    torch.manual_seed(protocol["settings"]["seed"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, dtype=dtype, device_map={"": 0},
        attn_implementation="sdpa", quantization_config=quantization).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    reader = Reader.__new__(Reader)
    reader.vlm = SimpleNamespace(model=model, processor=processor)
    cfg = protocol["settings"]
    reader.max_tokens = cfg["max_tokens"]
    out = ROOT / "reports/ecommerce_rag" / args.run_name
    out.mkdir(parents=True)
    source_sha = snapshot(out)
    adapter_sha = file_hash(args.adapter / "adapter_model.safetensors") if args.adapter else None
    data = DATA.parent / "abo-stage-c-extended-v1" if args.extended else DATA
    write_json(out / "config.json", {"model": MODEL_ID, "expected_revision": REVISION,
               "protocol_sha256": file_hash(data / "protocol.json"), "source_sha256": source_sha,
               "runner_sha256": file_hash(Path(__file__)), "dtype": str(dtype), "nf4": args.nf4,
               "gpu": torch.cuda.get_device_name(), "settings": cfg, "task": args.task,
               "adapter_sha256": adapter_sha})
    if args.task == "qlora-step":
        import peft

        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                      gradient_checkpointing_kwargs={"use_reentrant": False})
        q_names = [name for name, _ in model.named_modules()
                   if "language_model" in name and name.endswith(".self_attn.q_proj")]
        if len(q_names) < 2:
            raise RuntimeError("Expected Qwen3-VL text attention layers were not found")
        targets = [name.removesuffix("q_proj") + proj for name in q_names[-2:] for proj in ("q_proj", "v_proj")]
        model = peft.get_peft_model(model, peft.LoraConfig(r=8, lora_alpha=16, target_modules=targets, bias="none"))
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"])
        row = next(r for r in rows if r["split"] == "train" and r["suite"] == "two_pages")
        torch.cuda.reset_peak_memory_stats()
        inputs, targets, positions, info = training_input(reader, row, out / "images", cfg)
        logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.save_pretrained(out / "adapter", save_embedding_layers=False)
        summary = {"question_id": row["question_id"], "loss": loss.item(), "visual_tokens": info["visual_tokens_from_grid"],
                   "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "adapter_sha256": file_hash(out / "adapter/adapter_model.safetensors")}
        write_json(out / "training-summary.json", summary)
        (out / "训练记录.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        print(summary, flush=True)
        return
    if args.adapter:
        import peft
        reader.vlm.model = peft.PeftModel.from_pretrained(model, args.adapter).eval()
    split = "test" if args.extended else args.split
    selected = [row for row in rows if row["split"] == split][:args.limit]
    config_sha = file_hash(out / "config.json")
    for row in selected:
        pages = row["retrieved_pages"]
        result = reader.read(row["question"], row["request"], pages, out / "images", cfg["max_pixels"], PROMPT_C1)
        record = {**{key: value for key, value in row.items() if key != "gold_pages"}, **result,
                  "model": MODEL_ID, "model_revision": REVISION, "source_sha256": source_sha,
                  "config_sha256": config_sha, "run_name": args.run_name, "run_directory": str(out),
                  "status": "ok", "timestamp": timestamp(), "metrics": score(row, result, pages),
                  "routing": {"reason": "cloud_fixed_evidence", "quantized": args.nf4}}
        with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        append_question(out, record)
        print(row["question_id"], record["metrics"]["strict_pass"], result["peak_cuda_gib"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("check-data", "infer", "qlora-step"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--run-name", default="stage-c-qwen3vl8b-smoke")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--extended", action="store_true", help="Use the frozen 80-case held-out evaluation")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name) or args.limit < 1:
        parser.error("Invalid run name or case count")
    if args.adapter and args.task != "infer":
        parser.error("--adapter is only for inference")
    if args.extended and args.task != "infer":
        parser.error("--extended is only for held-out inference")
    if args.task == "qlora-step" and not args.nf4:
        parser.error("Select --nf4 for the QLoRA check")
    run(args)

"""Run the pinned 8B NF4 C0/C1/C2 experiment on fixed, family-disjoint evidence."""
import argparse
import json
from pathlib import Path
import random
import re
import time

from c8_protocol import DATA, load
from audit import append_question
from common import WORKSPACE, file_hash, timestamp, write_json
from qwen8_runtime import MODEL_ID, REVISION, Qwen8Runtime
from stage_a import Reader, snapshot
from stage_c import PROMPT_C1, compact_answer, report, score, training_input


def train(reader, rows, protocol, output):
    import peft
    import torch

    model = peft.prepare_model_for_kbit_training(reader.vlm.model, use_gradient_checkpointing=True,
                                                 gradient_checkpointing_kwargs={"use_reentrant": False})
    layers = [name for name, _ in model.named_modules()
              if "language_model" in name and name.endswith(".self_attn.q_proj")]
    if len(layers) < 2:
        raise RuntimeError("Expected Qwen3-VL text attention layers")
    targets = [layer.removesuffix("q_proj") + proj for layer in layers[-2:] for proj in ("q_proj", "v_proj")]
    model = peft.get_peft_model(model, peft.LoraConfig(r=8, lora_alpha=16, target_modules=targets, bias="none"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=protocol["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256, enabled=reader.vlm.dtype == torch.float16)
    batch = protocol["accumulation"]
    rng = random.Random(protocol["seed"])
    examples = list(rows)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    steps = 0
    for epoch in range(protocol["epochs"]):
        rng.shuffle(examples)
        for start in range(0, len(examples), batch):
            optimizer.zero_grad(set_to_none=True)
            items = examples[start:start + batch]
            for row in items:
                inputs, target, positions, info = training_input(reader, row, output / "images", protocol)
                logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits
                loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), target.flatten())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss on {row['question_id']}")
                scaler.scale(loss / len(items)).backward()
                losses.append(loss.item())
                with (output / "training-loss.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"epoch": epoch + 1, "id": row["question_id"], "loss": loss.item(),
                                             "visual_tokens": info["visual_tokens_from_grid"]}) + "\n")
                del logits, loss, inputs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=not scaler.is_enabled())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            steps += scaler.get_scale() >= scale_before
            if len(losses) % 10 == 0:
                print(f"8B training {len(losses)}/{len(examples) * protocol['epochs']} | loss={losses[-1]:.5f}", flush=True)
    if not steps:
        raise RuntimeError("No finite QLoRA optimizer step")
    model.save_pretrained(output / "adapter", save_embedding_layers=False)
    summary = {"examples": len(losses), "optimizer_updates": steps, "mean_loss": sum(losses) / len(losses),
               "seconds": time.perf_counter() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
               "trainable_parameters": sum(p.numel() for p in params), "targets": targets,
               "adapter_sha256": file_hash(output / "adapter/adapter_model.safetensors")}
    write_json(output / "training-summary.json", summary)
    (output / "训练记录.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return summary


def evaluate_cases(reader, selection, output, protocol, config_sha, source_sha, variant, prompt, adapter_sha=None):
    results = {}
    for index, row in enumerate(selection, 1):
        result = reader.read(row["question"], row["request"], row["retrieved_pages"], output / "images",
                             protocol["max_pixels"], prompt)
        scoring = row
        if row.get("answer_aliases") and compact_answer(result["answer"]) in {
                compact_answer(alias) for alias in row["answer_aliases"]}:
            scoring = {**row, "reference_answer": result["answer"]}
        record = {**row, **result, "metrics": score(scoring, result, row["retrieved_pages"]),
                  "variant": variant, "model": MODEL_ID, "model_revision": REVISION,
                  "adapter_sha256": adapter_sha, "source_sha256": source_sha, "config_sha256": config_sha,
                  "run_name": output.name, "run_directory": str(output), "status": "ok",
                  "timestamp": timestamp(), "routing": {"reason": "frozen_8b_nf4", "variant": variant}}
        with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        append_question(output, record)
        results[(variant, row["question_id"])] = record
        if index == 1 or index % 15 == 0 or index == len(selection):
            print(f"{variant} {row['split']} {index}/{len(selection)}", flush=True)
    return results


def run(model_dir, name, include_photos, parent_run=None):
    import gc

    import peft
    import torch

    rows, protocol = load()
    photos, photo_protocol = [], None
    supplement = None
    parent_config_sha = None
    adapter_sha = None
    if parent_run:
        from c8_supplement import load as load_supplement

        rows, supplement = load_supplement()
        parent_config = json.loads((parent_run / "config.json").read_text("utf-8"))
        summary = json.loads((parent_run / "training-summary.json").read_text("utf-8"))
        adapter_sha = file_hash(parent_run / "adapter/adapter_model.safetensors")
        expected = {"model": MODEL_ID, "revision": REVISION, "quantization": "nf4",
                    "settings": protocol, "protocol_sha256": file_hash(DATA / "protocol.json"),
                    "prompt_addition": PROMPT_C1, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
        if any(parent_config[key] != value for key, value in expected.items()):
            raise ValueError("Supplement must use the same model, accelerator, settings, and prompt as the parent")
        if adapter_sha != summary["adapter_sha256"]:
            raise ValueError("Parent adapter SHA mismatch")
        parent_config_sha = file_hash(parent_run / "config.json")
    elif include_photos:
        from c8_photos import load as load_photos

        photos, photo_protocol = load_photos()
        if {family for r in rows for family in r["family_keys"]} & {r["family_key"] for r in photos}:
            raise ValueError("Photo families overlap C cases")
    output = WORKSPACE / "reports/ecommerce_rag" / name
    output.mkdir(parents=True)
    source_sha = snapshot(output)
    torch.manual_seed(protocol["seed"])
    reader = Reader(max_tokens=protocol["max_tokens"], vlm=Qwen8Runtime(model_dir))
    config = {"model": MODEL_ID, "revision": REVISION, "quantization": "nf4",
                                        "compute_dtype": str(reader.vlm.dtype), "source_sha256": source_sha,
                                        "protocol_sha256": file_hash(DATA / "protocol.json"), "settings": protocol,
                                        "photos": photo_protocol, "prompt_addition": PROMPT_C1,
                                        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    if supplement:
        if config["compute_dtype"] != parent_config["compute_dtype"]:
            raise ValueError("Parent compute dtype changed")
        config.update(supplement=supplement, parent_config_sha256=parent_config_sha, parent_adapter_sha256=adapter_sha)
    write_json(output / "config.json", config)
    config_sha = file_hash(output / "config.json")
    results = {}

    def evaluate(variant, selection, prompt, adapter_sha=None):
        results.update(evaluate_cases(reader, selection, output, protocol, config_sha, source_sha, variant, prompt, adapter_sha))

    if parent_run:
        for variant, prompt in (("C0", ""), ("C1", PROMPT_C1)):
            evaluate(variant, rows, prompt)
        reader.vlm.model = peft.PeftModel.from_pretrained(reader.vlm.model, parent_run / "adapter").eval()
        evaluate("C2", rows, PROMPT_C1, adapter_sha)
        report(output, results)
        print(f"Frozen 8B supplement complete: {len(results)} rows", flush=True)
        return
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    for variant, prompt in (("C0", ""), ("C1", PROMPT_C1)):
        evaluate(variant, dev, prompt)
    # Fit only training families. Keep the test labels out of every generation call.
    torch.cuda.empty_cache()
    training = train(reader, [r for r in rows if r["split"] == "train"], protocol, output)
    reader.vlm.model = None
    gc.collect()
    torch.cuda.empty_cache()
    reader = Reader(max_tokens=protocol["max_tokens"], vlm=Qwen8Runtime(model_dir))
    for variant, prompt in (("C0", ""), ("C1", PROMPT_C1)):
        evaluate(variant, test + photos, prompt)
    reader.vlm.model = peft.PeftModel.from_pretrained(reader.vlm.model, output / "adapter").eval()
    evaluate("C2", dev, PROMPT_C1, training["adapter_sha256"])
    evaluate("C2", test + photos, PROMPT_C1, training["adapter_sha256"])
    report(output, results)
    print(f"Frozen 8B experiment complete: {len(results)} rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--photos", action="store_true")
    parser.add_argument("--parent-run", type=Path, help="Evaluate the frozen supplement with this completed run's adapter")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    if args.parent_run and args.photos:
        parser.error("The supplement includes photos by protocol")
    run(args.model_dir, args.run_name, args.photos, args.parent_run)

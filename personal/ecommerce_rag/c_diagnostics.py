"""Paired resolution and correct-page diagnostics on the frozen C protocol."""
import argparse
import json
import re
import statistics

import torch

from c_protocol import DATA, load
from audit import append_question
from common import WORKSPACE, file_hash, operation, write_json
from stage_a import MODEL_REVISION, Reader, snapshot
from stage_c import PROMPT_C1, score


LOW_PIXELS = 230400


def run(run_name):
    rows, protocol = load()
    settings = protocol["settings"]
    dev = [row for row in rows if row["split"] == "dev" and row["suite"] == "unit"]
    historical = [row for row in rows if row["split"] == "test" and row["suite"] == "two_pages"]
    if len(dev) != 8 or len(historical) != 8:
        raise ValueError("Unexpected frozen diagnostic groups")
    out = WORKSPACE / "reports/ecommerce_rag" / run_name
    out.mkdir(parents=True)
    source_sha = snapshot(out)
    standard = settings["max_pixels"]
    config = {"protocol_sha256": file_hash(DATA / "protocol.json"), "source_sha256": source_sha,
              "model_revision": MODEL_REVISION, "prompt": PROMPT_C1,
              "pixels": {"standard": standard, "half_pixels": LOW_PIXELS, "oracle_dev": standard},
              "dev_ids": [row["question_id"] for row in dev],
              "historical_double_ids": [row["question_id"] for row in historical],
              "gpu": torch.cuda.get_device_name(), "max_tokens": settings["max_tokens"]}
    write_json(out / "config.json", config)
    config_sha = file_hash(out / "config.json")
    reader = Reader(settings["max_tokens"])
    # The warm-up is excluded from measurement.
    first = dev[0]
    reader.read(first["question"], first["request"], first["retrieved_pages"], out / "images", standard, PROMPT_C1)
    records = []
    for index, row in enumerate(dev + historical):
        variants = ["standard", "half_pixels"]
        if index % 2:
            variants.reverse()
        if row["split"] == "dev":
            variants.append("oracle_dev")
        for variant in variants:
            oracle = variant == "oracle_dev"
            pages = row["gold_pages"] if oracle else row["retrieved_pages"]
            result = reader.read(row["question"], row["request"], pages, out / "images",
                                 config["pixels"][variant], PROMPT_C1)
            record = {**{key: value for key, value in row.items() if key != "gold_pages"}, **result,
                      "status": "ok", "run_name": run_name, "run_directory": str(out), "variant": variant,
                      "model": "Qwen/Qwen3.5-2B", "model_revision": MODEL_REVISION,
                      "source_sha256": source_sha, "config_sha256": config_sha,
                      "retrieved_pages": pages, "retrieval_seconds": 0.0 if oracle else row["retrieval_seconds"],
                      "diagnostic_oracle": oracle, "metrics": score(row, result, pages),
                      "routing": {"reason": variant, "max_pixels": config["pixels"][variant]}}
            with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            append_question(out, record)
            records.append(record)
        print(f"C diagnostics {index + 1}/{len(dev + historical)}: {row['question_id']}", flush=True)
    summary = []
    for split in ("dev", "test"):
        for variant in config["pixels"]:
            group = [row for row in records if row["split"] == split and row["variant"] == variant]
            if not group:
                continue
            summary.append({"split": split, "variant": variant, "n": len(group),
                            "strict": sum(row["metrics"]["strict_pass"] for row in group),
                            "value_and_unit": sum(row["metrics"]["exact_value_unit"] for row in group),
                            "citations": sum(row["metrics"]["correct_evidence"] for row in group),
                            "median_visual_tokens": statistics.median(row["visual_tokens_from_grid"] for row in group),
                            "median_prepare_s": statistics.median(row["prepare_seconds"] for row in group),
                            "median_generation_s": statistics.median(row["generation_seconds"] for row in group),
                            "max_cuda_gib": max(row["peak_cuda_gib"] for row in group)})
    write_json(out / "summary.json", summary)
    (out / "分辨率与正确页对照.txt").write_text(
        "固定模型与 C1 提示。standard/half_pixels 只改变像素上限；oracle_dev 使用人工标记正确页作诊断。"
        "历史测试双商品组仅用于诊断，不能用于参数选择。\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    operation("C 阶段分辨率与正确页诊断", "固定模型、提示、候选与解码做像素配对，并独立加入开发集正确页诊断。",
              f"完成 {len(records)} 条逐题记录", [out], "查看逐题结果及分辨率与正确页对照。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("Invalid run name")
    run(args.run_name)

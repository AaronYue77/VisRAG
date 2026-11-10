"""Build and compare a 120-example 8B adapter without replacing the accepted run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from shutil import copy2

import c8_photos
import c8_protocol
import c8_supplement
from common import WORKSPACE, file_hash, read_jsonl, write_json, write_jsonl


VERSION = "abo-8b-120-v1"
DATA = c8_protocol.DATA.parent / VERSION
QA = DATA / "qa"
PHOTOS = DATA / "photos"
SPLITS = {"train": (0, 16, 14, 6, 4), "dev": (0, 5, 5, 3, 2)}
REPORTS = WORKSPACE / "reports/ecommerce_rag"
OLD = REPORTS / "stage-c-8b-180-20261109"
OLD_COMBINED = REPORTS / "stage-c-8b-310-20261110"
ACTIVE = DATA.parent / "stage-c-active.json"
RUN_NAME = "stage-c-8b-120-20261110"
KERNEL = "aaronyue77/abo-rag-8b-120-example-candidate"
RULE = {
    "frozen_qa_min": 199, "frozen_photo_min": 74, "at_least_one_strict_gain": True,
    "no_stratum_loss": True, "no_increase_in_false_refusal_or_unsupported_answer": True,
    "new_dev_no_net_loss": True,
}


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def families(rows):
    return {key for row in rows for key in (row.get("family_keys") or [row["family_key"]])}


def original_cases():
    qa, protocol = c8_protocol.load()
    photos, _ = c8_photos.load()
    extra, _ = c8_supplement.load()
    test = [r for r in qa if r["split"] == "test"] + photos + extra
    return qa, test, protocol


def lock_baseline():
    if (DATA / "baseline.json").exists():
        return read_json(DATA / "baseline.json")
    old_summary = read_json(OLD / "training-summary.json")
    adapter = OLD / "adapter/adapter_model.safetensors"
    if file_hash(adapter) != old_summary["adapter_sha256"]:
        raise ValueError("Accepted 60-example adapter SHA changed")
    paired = read_json(OLD_COMBINED / "paired-summary.json")
    if (paired["test"]["C2"]["strict"], paired["photo"]["C2"]["strict"]) != (199, 74):
        raise ValueError("Unexpected accepted scores")
    paths = [adapter, OLD / "adapter/adapter_config.json", OLD / "config.json", OLD / "training-summary.json",
             OLD_COMBINED / "results.jsonl", OLD_COMBINED / "paired-summary.json"]
    for folder in (c8_protocol.DATA, c8_photos.DATA, c8_supplement.QA, c8_supplement.PHOTOS):
        paths.extend((folder / "questions.jsonl", folder / "protocol.json"))
    baseline = {"files": {p.relative_to(WORKSPACE).as_posix(): file_hash(p) for p in paths},
                "adapter_sha256": old_summary["adapter_sha256"], "config_sha256": file_hash(OLD / "config.json"),
                "train": 60, "dev": 25, "gate": RULE}
    write_json(DATA / "baseline.json", baseline)
    if not ACTIVE.exists():
        write_json(ACTIVE, {"kernel": "aaronyue77/abo-rag-qwen3-vl-8b-formal",
                            "run": OLD.name, "train": 60, "dev": 25,
                            "adapter": adapter.relative_to(WORKSPACE).as_posix(),
                            "adapter_sha256": baseline["adapter_sha256"], "config_sha256": baseline["config_sha256"]})
    return baseline


def prepare():
    qa, test, _ = original_cases()
    baseline = lock_baseline()
    verify_baseline(baseline)
    labels = read_jsonl(PHOTOS / "annotations.jsonl")
    if len(labels) != 30 or len({r["source_id"] for r in labels}) != 30:
        raise ValueError("Exactly 30 separately reviewed official assets are required")
    pool = {r["id"]: r for r in read_jsonl(c8_supplement.PHOTOS / "candidates.jsonl")}
    sources = [{**pool[r["source_id"]], "id": r["id"], "source_id": r["source_id"]} for r in labels]
    if families(sources) & families(qa + test):
        raise ValueError("Review photos overlap original train/development/test families")
    if not (PHOTOS / "protocol.json").exists():
        write_jsonl(PHOTOS / "sources.jsonl", sources)
        c8_photos.freeze(PHOTOS / "annotations.jsonl", PHOTOS, VERSION)
    if not QA.exists():
        c8_protocol.freeze(QA, VERSION, SPLITS, excluded=families(qa + test + sources),
                           charts=True, id_prefix="C8U", category_limit=96)
    train, dev, _, _, _ = load()
    print("Frozen train/dev:", len(train), len(dev), "classes:",
          len({r["product_type"] for r in train}), len({r["product_type"] for r in dev}), flush=True)


def load():
    original, test, protocol = original_cases()
    qa, _ = c8_protocol.load(QA, SPLITS)
    photos, photo_protocol = c8_photos.load(PHOTOS, 6)
    labels_path = PHOTOS / "annotations.jsonl"
    if file_hash(labels_path) != photo_protocol["annotations_sha256"]:
        raise ValueError("Frozen labels changed")
    labels = {r["id"]: r["split"] for r in read_jsonl(labels_path)}
    photos = [{**r, "split": labels[r["id"]]} for r in photos]
    if Counter((r["split"], r["suite"]) for r in photos) != Counter({
            (split, kind): n for split, n in (("train", 4), ("dev", 2)) for kind in c8_photos.TYPES}):
        raise ValueError("New photo train/development class counts differ")
    if families(qa + photos) & families(original + test) or families(qa) & families(photos):
        raise ValueError("New train/development cases overlap an existing family")
    image_hashes = [r["image_sha256"] for r in test + photos if "image_sha256" in r]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError("Reused source image")
    train = [r for r in original + qa + photos if r["split"] == "train"]
    dev = [r for r in original + qa + photos if r["split"] == "dev"]
    if families(train) & families(dev) or families(train + dev) & families(test):
        raise ValueError("Train/development/test family leakage")
    if (len(train), len(dev), len(test)) != (120, 50, 310):
        raise ValueError("Incorrect upgrade split size")
    ids = [r["question_id"] for r in train + dev + test]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated question IDs")
    return train, dev, test, protocol, {r["question_id"] for r in qa + photos if r["split"] == "dev"}


def verify_baseline(baseline):
    if baseline["gate"] != RULE:
        raise ValueError("Frozen activation gate differs")
    for name, expected in baseline["files"].items():
        if file_hash(WORKSPACE / name) != expected:
            raise ValueError(f"Accepted baseline file changed: {name}")


def run(model_dir, baseline_dir, name):
    import gc
    import peft
    import torch

    from c8_experiment import evaluate_cases, train
    from qwen8_runtime import MODEL_ID, REVISION, Qwen8Runtime
    from stage_a import Reader, snapshot
    from stage_c import PROMPT_C1

    train_rows, dev, test, protocol, _ = load()
    baseline = read_json(DATA / "baseline.json")
    if baseline["gate"] != RULE:
        raise ValueError("Frozen activation gate differs")
    if file_hash(baseline_dir / "config.json") != baseline["config_sha256"]:
        raise ValueError("Wrong accepted model configuration")
    old_sha = file_hash(baseline_dir / "adapter/adapter_model.safetensors")
    if old_sha != baseline["adapter_sha256"]:
        raise ValueError("Wrong accepted adapter")
    parent_config = read_json(baseline_dir / "config.json")
    env = {"model": MODEL_ID, "revision": REVISION, "quantization": "nf4",
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "prompt_addition": PROMPT_C1,
           "settings": protocol}
    if any(parent_config[k] != v for k, v in env.items()):
        raise ValueError("Evaluation settings or hardware differ from accepted run")
    output = REPORTS / name
    output.mkdir(parents=True)
    source_sha = snapshot(output)
    torch.manual_seed(protocol["seed"])
    reader = Reader(max_tokens=protocol["max_tokens"], vlm=Qwen8Runtime(model_dir))
    if str(reader.vlm.dtype) != parent_config["compute_dtype"]:
        raise ValueError("Compute dtype differs from accepted run")
    config = {**env, "compute_dtype": str(reader.vlm.dtype), "source_sha256": source_sha,
              "baseline_sha256": file_hash(DATA / "baseline.json"), "sizes": [120, 50, 310],
              "new_qa_sha256": file_hash(QA / "protocol.json"), "new_photo_sha256": file_hash(PHOTOS / "protocol.json")}
    write_json(output / "config.json", config)
    config_sha = file_hash(output / "config.json")
    training = train(reader, train_rows, protocol, output)
    reader.vlm.model = None
    gc.collect()
    torch.cuda.empty_cache()
    reader = Reader(max_tokens=protocol["max_tokens"], vlm=Qwen8Runtime(model_dir))
    model = peft.PeftModel.from_pretrained(reader.vlm.model, baseline_dir / "adapter", adapter_name="prior").eval()
    reader.vlm.model = model
    evaluate_cases(reader, dev, output, protocol, config_sha, source_sha, "prior", PROMPT_C1, old_sha)
    model.load_adapter(output / "adapter", adapter_name="candidate")
    model.set_adapter("candidate")
    evaluate_cases(reader, dev, output, protocol, config_sha, source_sha, "candidate", PROMPT_C1,
                   training["adapter_sha256"])
    evaluate_cases(reader, test, output, protocol, config_sha, source_sha, "candidate", PROMPT_C1,
                   training["adapter_sha256"])
    print("Completed candidate: 120 train, 100 paired dev predictions, 310 test predictions", flush=True)


def summarize(pairs):
    return {"n": len(pairs), "old": sum(a["metrics"]["strict_pass"] for a, _ in pairs),
            "new": sum(b["metrics"]["strict_pass"] for _, b in pairs),
            "improved": sum(not a["metrics"]["strict_pass"] and b["metrics"]["strict_pass"] for a, b in pairs),
            "regressed": sum(a["metrics"]["strict_pass"] and not b["metrics"]["strict_pass"] for a, b in pairs),
            **{metric: [sum(pair[index]["metrics"][metric] for pair in pairs) for index in (0, 1)]
               for metric in ("false_refusal", "unsupported_answer")}}


def decision(output, breadth_output, activate=False):
    from c8_visual_breadth import compare as compare_breadth

    _, dev, test, _, new_dev_ids = load()
    baseline = read_json(DATA / "baseline.json")
    verify_baseline(baseline)
    config_sha = file_hash(output / "config.json")
    config = read_json(output / "config.json")
    training = read_json(output / "training-summary.json")
    sha = file_hash(output / "adapter/adapter_model.safetensors")
    if (config["baseline_sha256"] != file_hash(DATA / "baseline.json") or config["sizes"] != [120, 50, 310]
            or config["new_qa_sha256"] != file_hash(QA / "protocol.json")
            or config["new_photo_sha256"] != file_hash(PHOTOS / "protocol.json")
            or training["adapter_sha256"] != sha or training["examples"] != 120 or training["optimizer_updates"] <= 0):
        raise ValueError("Incomplete or incompatible candidate run")
    rows = read_jsonl(output / "results.jsonl")
    by_key = {(r["variant"], r["question_id"]): r for r in rows}
    dev_ids = {r["question_id"] for r in dev}
    test_ids = {r["question_id"] for r in test}
    if len(rows) != len(by_key) or by_key.keys() != ({("prior", q) for q in dev_ids} |
                                                     {("candidate", q) for q in dev_ids | test_ids}):
        raise ValueError("Incomplete or duplicate predictions")
    old = {r["question_id"]: r for r in read_jsonl(OLD_COMBINED / "results.jsonl")
           if r["variant"] == "C2" and r["split"] in ("test", "photo")}
    if old.keys() != test_ids:
        raise ValueError("Accepted frozen records differ")
    grouped = defaultdict(list)
    paired = []
    for case in dev + test:
        qid, split = case["question_id"], case["split"]
        new = by_key["candidate", qid]
        previous = by_key["prior", qid] if split == "dev" else old[qid]
        for row, expected_sha in ((new, sha), (previous, baseline["adapter_sha256"])):
            if any(row[key] != case[key] for key in ("question", "reference_answer", "reference_pages", "expected_action")):
                raise ValueError(f"Changed frozen question or label: {qid}")
            if [p["page_id"] for p in row["retrieved_pages"]] != [p["page_id"] for p in case["retrieved_pages"]]:
                raise ValueError(f"Changed evidence order: {qid}")
            if row["adapter_sha256"] != expected_sha or row["status"] != "ok":
                raise ValueError(f"Invalid prediction provenance: {qid}")
        if new["config_sha256"] != config_sha or (split == "dev" and previous["config_sha256"] != config_sha):
            raise ValueError(f"Invalid prediction configuration: {qid}")
        grouped[split, "all"].append((previous, new))
        if split != "dev":
            grouped[split, case["suite"]].append((previous, new))
        elif qid in new_dev_ids:
            grouped[split, "new-only"].append((previous, new))
        if split != "dev" and previous["metrics"]["strict_pass"] != new["metrics"]["strict_pass"]:
            paired.append({"id": qid, "split": split, "suite": case["suite"], "question": case["question"],
                           "reference": case["reference_answer"], "old_pass": previous["metrics"]["strict_pass"],
                           "new_pass": new["metrics"]["strict_pass"], "old": previous["raw_output"], "new": new["raw_output"]})
    metrics = {f"{split}/{suite}": summarize(items) for (split, suite), items in grouped.items()}
    qa, photos, dev_new = (metrics[k] for k in ("test/all", "photo/all", "dev/new-only"))
    if (qa["old"], photos["old"]) != (RULE["frozen_qa_min"], RULE["frozen_photo_min"]):
        raise ValueError("Accepted scores do not match the gate")
    frozen = [v for key, v in metrics.items() if not key.startswith("dev/")]
    checks = {
        "qa_at_least_199": qa["new"] >= qa["old"], "photos_at_least_74": photos["new"] >= photos["old"],
        "at_least_one_strict_gain": qa["new"] > qa["old"] or photos["new"] > photos["old"],
        "no_frozen_stratum_loss": all(v["new"] >= v["old"] for v in frozen),
        "no_frozen_refusal_regression": all(v[k][1] <= v[k][0] for v in frozen
                                          for k in ("false_refusal", "unsupported_answer")),
        "new_dev_no_net_loss": dev_new["new"] >= dev_new["old"],
    }
    breadth_pass, breadth_metrics = compare_breadth(breadth_output, sha)
    checks["broader_photos_no_net_loss_or_refusal_regression"] = breadth_pass
    metrics["breadth/all"] = breadth_metrics["all"]
    accepted = all(checks.values())
    write_json(output / "decision.json", {"accepted": accepted, "checks": checks, "metrics": metrics})
    write_json(output / "paired-changes.json", paired)
    lines = ["8B 60/25 与 120/50 版本验收", "判定：" + ("通过，允许切换" if accepted else "未通过，保留 60/25"),
             "旧 310 道冻结题用原 C2 输出配对。新增 25 道开发题与 75 道照片各自统计。", ""]
    for key, value in metrics.items():
        lines.append(f"{key} | {value['old']}/{value['n']} → {value['new']}/{value['n']} | "
                     f"改善 {value['improved']}，退化 {value['regressed']} | "
                     f"误拒答 {value['false_refusal']}，无证据作答 {value['unsupported_answer']}")
    lines.extend(("", "切换门槛", *(f"{key} | {'通过' if passed else '未通过'}" for key, passed in checks.items()),
                  "", "逐题变化", json.dumps(paired, ensure_ascii=False, indent=2)))
    (output / "升级验收.txt").write_text("\n".join(lines) + "\n", "utf-8")
    if activate and accepted:
        if read_json(ACTIVE)["adapter_sha256"] not in (baseline["adapter_sha256"], sha):
            raise ValueError("The active C configuration changed during candidate evaluation")
        selected = {"kernel": KERNEL, "run": output.name, "train": 120, "dev": 50,
                    "adapter": (output / "adapter/adapter_model.safetensors").relative_to(WORKSPACE).as_posix(),
                    "adapter_sha256": sha, "config_sha256": config_sha}
        temporary = ACTIVE.with_suffix(".json.tmp")
        write_json(temporary, selected)
        temporary.replace(ACTIVE)
    print("Candidate", "accepted" if accepted else "rejected", json.dumps(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("prepare", "run", "decide"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path, default=REPORTS / RUN_NAME)
    parser.add_argument("--breadth-output", type=Path)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if args.task == "prepare":
        prepare()
    elif args.task == "run":
        if not args.model_dir or not args.baseline_dir:
            parser.error("run requires --model-dir and --baseline-dir")
        run(args.model_dir, args.baseline_dir, RUN_NAME)
    else:
        if not args.breadth_output:
            parser.error("decide requires --breadth-output")
        decision(args.output.resolve(), args.breadth_output.resolve(), args.activate)

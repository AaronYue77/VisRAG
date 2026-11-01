"""Check original feature retrieval artifacts and comparable ten-question results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import statistics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "a_baseline"))
from common import WORKSPACE, file_hash, read_jsonl, write_json, operation, issue, timestamp


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-name", default="smoke-dpi200-v1")
    parser.add_argument("--retrieval-name", default="smoke-top2-v1")
    parser.add_argument("--run-name", default="smoke10-visrag-ret-v1")
    args = parser.parse_args()
    index = WORKSPACE / "data/visrag_repro" / args.index_name
    retrieval = WORKSPACE / "data/visrag_repro" / args.retrieval_name
    result = WORKSPACE / "reports/a_baseline" / args.run_name
    baseline = WORKSPACE / "reports/a_baseline/smoke10-a-v2"
    manifest = json.loads((index / "index-manifest.json").read_text("utf-8"))
    matrix = np.load(index / "reps.npy", allow_pickle=False)
    queries = np.load(retrieval / "query_reps.npy", allow_pickle=False)
    scores = np.load(retrieval / "similarities.npy", allow_pickle=False)
    mapping = read_jsonl(index / "page-map.jsonl")
    ranking = read_jsonl(retrieval / "retrieval.jsonl")
    rows, old_rows = read_jsonl(result / "results.jsonl"), read_jsonl(baseline / "results.jsonl")
    config = json.loads((result / "config.json").read_text("utf-8"))
    old = json.loads((baseline / "config.json").read_text("utf-8"))
    assert matrix.shape == (27, 2304) and queries.shape == (10, 2304) and scores.shape == (10, 27)
    assert all(np.isfinite(array).all() for array in (matrix, queries, scores))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1, atol=2e-5)
    # CUDA/CPU float32 reductions can differ slightly; compare with a numeric tolerance.
    assert np.allclose(scores, queries @ matrix.T, atol=2e-5)
    assert file_hash(index / "reps.npy") == manifest["reps_sha256"]
    assert file_hash(index / "page-map.jsonl") == manifest["page_map_sha256"]
    assert len(rows) == len(old_rows) == len(ranking) == 10
    for i, (row, prior, retrieved) in enumerate(zip(rows, old_rows, ranking)):
        for key in ("question_id", "question", "reference_answer", "reference_pages", "answer_aliases"):
            assert row[key] == prior[key]
        assert row["retrieved_pages"] == retrieved["retrieved_pages"]
        assert row["config_sha256"] == file_hash(result / "config.json")
        assert row["status"] == "ok"
        for rank, page in enumerate(row["retrieved_pages"], start=1):
            assert page["rank"] == rank
            assert mapping[page["vector_row"]]["page_id"] == page["page_id"]
            assert abs(page["score"] - float(scores[i, page["vector_row"]])) < 1e-6
        assert set(row["citations"]) <= {p["page_id"] for p in row["retrieved_pages"]}
        assert all((WORKSPACE / path).exists() for path in row["image_paths"])
    for key in ("model", "model_revision", "model_config_sha256", "max_pixels_per_page", "max_new_tokens", "generation", "questions_sha256"):
        assert config[key] == old[key], key
    # The generation core file and prompt are byte-identical to the original baseline snapshot.
    assert file_hash(result / "source_snapshot/core.py") == file_hash(baseline / "source_snapshot/core.py")
    digest = hashlib.sha256()
    for path in sorted((result / "source_snapshot").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    assert digest.hexdigest() == config["source_sha256"]
    summary = json.loads((result / "summary.json").read_text("utf-8"))
    previous_summary = json.loads((baseline / "summary.json").read_text("utf-8"))
    report = {"verified_at": timestamp(), "all_checks_passed": True, "environment": json.loads((WORKSPACE / "reports/visrag-repro/environment.json").read_text("utf-8")),
              "model_revision": manifest["runtime"]["model_revision"], "page_vector_shape": list(matrix.shape), "query_vector_shape": list(queries.shape),
              "same_question_and_gold": True, "same_generator_and_prompt": True,
              "topk_scores_match_saved_matrix": True, "config_and_source_hashes_match": True,
              "generation_gpu_peak_gib": summary["peak_cuda_gib"], "retrieval_gpu_peak_gib": manifest["peak_cuda_gib"],
              "retrieval_mean_seconds": statistics.mean(row["retrieval_seconds"] for row in rows),
              "baseline_metrics": previous_summary["metrics"], "visual_retrieval_metrics": summary["metrics"],
              "note": "Ten previously inspected development questions; not a held-out significance/generalization claim. Generator is Qwen3.5-2B, not the paper's exact original MiniCPM generator."}
    write_json(WORKSPACE / "reports/visrag-repro/verification.json", report)
    lines = ["原版 VisRAG-Ret 视觉检索与小 VLM 运行验收", "核对时间：" + report["verified_at"], "",
             "1. 独立环境：.venv-visrag，Python 3.11.18；torch 2.1.2+cu121、transformers 4.40.2 等原版核心版本。", "环境与完整依赖锁保存在 reports/visrag-repro/environment.json、requirements.freeze.txt。仅省略与 PDF 推理无关的训练/视频/额外评测扩展。", "",
             "2. 原始编码：固定官方 VisRAG-Ret revision；直接调用未改动的上游 utils.encode，原版加权池化和 L2 归一化。", "27 页按 200 DPI 渲染，reps.npy 为 27x2304；10 个问题的 query_reps.npy 为 10x2304；页 ID 和图片/向量行映射已保存。", "未量化权重；仅将未参与 embedding 前向的 llm.lm_head 放到 CPU，以适配 8GB 显存。模型源码与权重仍按官方哈希校验。", "",
             "3. 完整运行：先结束旧版检索进程，再在原 Qwen 环境生成。问题、参考标签、VLM、提示、像素与解码参数与原 BM25 基线相同。", "区别仅为检索产生的页面；引用、原始输出、耗时、版本与输入图片均保存。", "",
             "基线指标：" + json.dumps(previous_summary["metrics"], ensure_ascii=False), "视觉检索指标：" + json.dumps(summary["metrics"], ensure_ascii=False),
             f"检索峰值已分配显存：{report['retrieval_gpu_peak_gib']:.3f} GiB；生成峰值：{report['generation_gpu_peak_gib']:.3f} GiB；二者不同时驻留。", "",
             "逐题对照（保留所有正确、错误和拒答）："]
    for row, prior in zip(rows, old_rows):
        lines += [row["question_id"] + "：" + row["question"], "参考：" + row["reference_answer"] + "；证据：" + str(row["reference_pages"]),
                  "BM25：" + str([p["page_id"] for p in prior["retrieved_pages"]]) + "；回答：" + prior["answer"],
                  "视觉检索：" + str([p["page_id"] for p in row["retrieved_pages"]]) + "；回答：" + row["answer"], ""]
    lines += ["结果目录：" + str(result), "向量索引：" + str(index), "问题向量与检索缓存：" + str(retrieval),
              "这是原始检索方法的运行复现与小 VLM 适配，不是原论文完整数据集排行榜复现。现有 50 道最终测试题未运行。"]
    path = WORKSPACE / "记录/VisRAG原版检索复现验收与对照.txt"
    path.write_text("\n".join(lines) + "\n", "utf-8", newline="\n")
    operation("验收原版视觉编码、Top-k 与同批问答", "核对向量形状/归一化、独立重算点积分数、页 ID/行号映射、同题/同标签/同 Reader、配置/源码快照哈希及引用范围。", json.dumps(report, ensure_ascii=False), [path, result / "逐题结果.txt", WORKSPACE / "reports/visrag-repro/verification.json"], "所有断言通过；只用 10 道已见开发题，不评估封存测试；保留原 BM25 与新视觉结果。")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        key = issue(f"原版视觉复现验收失败：{type(error).__name__}: {error}", "核对原始文件、映射、分数或配置对照断言。", "保留所有输出，定位差异后重新验证，不能把未通过断言的运行宣布完成。")
        operation("视觉复现验收失败", "verify_results.py 静态证据对照", str(error), status="失败，待修复", related_issues=key)
        raise

"""Human-readable TXT exports alongside machine-readable JSONL results."""
import json
from pathlib import Path

from common import WORKSPACE, read_jsonl

ALL_RESULTS_TXT = WORKSPACE / "记录/逐题结果汇总.txt"


def format_question(row: dict) -> str:
    lines = ["=" * 72, f"运行名称：{row['run_name']}", f"问题 ID：{row['question_id']}",
             f"问题：{row['question']}", f"中文译文：{row.get('question_zh', '未提供')}",
             f"参考答案：{row['reference_answer']}", f"参考证据页：{', '.join(row['reference_pages'])}",
             "页码规则：PDF 物理页，从 1 开始；不一定等于论文印刷页号。", "召回页面及排序："]
    for page in row["retrieved_pages"]:
        lines.append(f"  第 {page['rank']} 名 | {page['page_id']} | score={page['score']:.6f} | 来源={page['text_source']}")
    generated = row.get("model") is not None
    lines.extend([f"模型答案：{row.get('answer', '生成失败' if generated else '未运行生成（仅检索模式）')}",
                  f"输出引用：{', '.join(row.get('citations', [])) or '无'}",
                  f"原始引用标签：{row.get('citation_labels', [])}",
                  f"模型原始输出：{row.get('raw_output', '无')}",
                  f"检索耗时（秒）：{row['retrieval_seconds']:.6f}",
                  f"生成耗时（秒）：{row.get('generation_seconds', '未运行或失败')}",
                  f"输入准备耗时（秒）：{row.get('prepare_seconds', '未运行或失败')}",
                  f"答案来源：{row.get('answer_source', '模型生成' if generated else '未执行模型生成')}",
                  f"业务动作：{row.get('action', '未单独标注')}",
                  f"模型：{row.get('model') or '本题未调用生成模型'}",
                  f"模型版本（revision）：{row.get('model_revision') or '不适用'}",
                  f"配置版本（SHA256）：{row['config_sha256']}",
                  f"运行配置文件：{row.get('run_directory', 'reports/a_baseline/' + row['run_name'])}/config.json",
                  f"代码版本（SHA256）：{row['source_sha256']}",
                  f"执行状态：{row['status']}", f"错误及问题编号：{row.get('error', '无')}；{row.get('issue_id', '无')}",
                  f"评测结果：{json.dumps(row['metrics'], ensure_ascii=False)}",
                  f"实际输入图片：{json.dumps(row.get('image_paths', []), ensure_ascii=False)}", ""])
    return "\n".join(lines) + "\n"


def append_question(run_dir: Path, row: dict) -> None:
    """Append immediately after JSONL flush, so an interrupted run still has TXT."""
    content = format_question(row)
    for path in (run_dir / "逐题结果.txt", ALL_RESULTS_TXT):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(content)


def export_existing_runs() -> list[dict]:
    """Rebuild derived TXT from preserved JSONL without modifying original evidence."""
    aggregate = ["逐题结果汇总：由原始 results.jsonl 导出，JSONL 与原始配置保留不变。\n"]
    counts = []
    result_files = [p for folder in ("reports/a_baseline", "reports/ecommerce_rag") for p in (WORKSPACE / folder).glob("*/results.jsonl")]
    for result_path in sorted(result_files):
        rows = read_jsonl(result_path)
        content = "逐题结果：参考标签只参与计分，不进入问答模型提示。\n\n" + "".join(format_question(row) for row in rows)
        (result_path.parent / "逐题结果.txt").write_text(content, "utf-8", newline="\n")
        aggregate.append(content)
        counts.append({"run": result_path.parent.name, "records": len(rows), "txt": str(result_path.parent / "逐题结果.txt")})
    ALL_RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    ALL_RESULTS_TXT.write_text("\n".join(aggregate), "utf-8", newline="\n")
    return counts


if __name__ == "__main__":
    from common import operation
    counts = export_existing_runs()
    operation("补齐/重建逐题 TXT", "读取每个运行目录的原始 results.jsonl，逐字段导出 TXT；不改动原结果或配置。", json.dumps(counts, ensure_ascii=False), [ALL_RESULTS_TXT], "TXT 数量与原 JSONL 记录数一致；含仅检索与完整问答的模式说明。")
    print(json.dumps(counts, ensure_ascii=False, indent=2))

"""Replay the reported issue and adjacent scenarios through the shared app."""
import json
import time

from full_app import FullApplication
from common import WORKSPACE, read_jsonl, write_json


def run():
    out = WORKSPACE / "reports/ecommerce_rag" / time.strftime("product-reference-%Y%m%d-%H%M%S")
    app = FullApplication("B", report_root=out)
    session = app.session()
    checks = []

    def ask(question, action, extra=lambda response, row: True):
        response = session.respond(question)
        row = read_jsonl(session.out / "results.jsonl")[-1]
        passed = response["status"] == "ok" and response["action"] == action and extra(response, row)
        checks.append({"question": question, "expected": action, "actual": response["action"],
                       "passed": passed, "answer": response["answer"], "citations": response["citations"],
                       "routing": row["routing"], "source": row.get("answer_source")})
        write_json(out / "checks.json", checks)
        print(question, action, response["action"], passed, flush=True)
        return response

    def category(kind):
        return lambda v, r: bool(v["candidates"]) and {item["product_type"] for item in v["candidates"]} == {kind}

    def source(v, r):
        return bool(v["citations"] and r.get("claims")) and r["answer_source"] == "verified_source_fields_and_notes"

    def ordinary(v, r):
        return bool(v["answer"]) and not v["citations"] and not v["retrieved_pages"]

    try:
        ask("这里有哪些商品", "catalog")
        ask("还有呢", "catalog")
        ask("背包", "search_results", category("BACKPACK"))
        ask("我想查看背包", "search_results", category("BACKPACK"))
        ask("查看这个 B076QBF37R@amazon.in", "answer",
            lambda v, r: source(v, r) and "B076QBF37R" in v["answer"] and "B0760BF37R" not in v["answer"])
        ask("他有什么功能？", "answer", lambda v, r: source(v, r) and "15.6" in v["answer"])
        ask("我说这个背包有什么功能属性", "answer", source)
        ask("这款耳机有什么功能？", "clarify")
        ask("选耳机类呢", "search_results", category("HEADPHONES"))
        ask("看看第九十九款", "clarify")
        ask("看看第二款", "answer", source)
        ask("查这款的库存", "requires_tool")
        ask("查看这个 B00Q3OH45G@amazon.co.uk", "answer", source)
        ask("它有什么用途？", "answer", source)
        ask("换个话题，帮我写一句描写晚霞的句子", "chat", ordinary)
        ask("再给我一个例子", "chat", ordinary)
        ask("你有什么功能？", "chat", ordinary)
        ask("B005FIG90C@amazon.co.jp", "selected")
        ask("这个商品的材质是什么？", "answer", lambda v, r: bool(v["citations"]) and bool(r.get("raw_output")))
        ask("查看 B076QBF37R@amazon.in 和 B005FIG90C@amazon.co.jp", "answer", source)
        ask("它有什么功能？", "clarify")
        ask("查看 B000000000", "not_found")
        ask("商品列表", "catalog")
        ask("GROCERY", "search_results", category("GROCERY"))
        ask("重置", "reset")
        ask("这个商品的相关信息", "clarify")
    finally:
        report = {"checks": len(checks), "passed": sum(check["passed"] for check in checks),
                  "per_question_txt": str(session.out / "逐题结果.txt"), "dialogue_txt": str(session.transcript)}
        write_json(out / "summary.json", report)
        (out / "会话验收.txt").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n\n" +
                                          json.dumps(checks, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        app.close()
    if not checks or report["passed"] != report["checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()

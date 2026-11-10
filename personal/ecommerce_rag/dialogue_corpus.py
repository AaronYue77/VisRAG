"""An additive, source-faithful corpus for dialogue; never rewrite frozen v1."""
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "rag_common"))
from common import WORKSPACE, file_hash, operation, write_json
from stage_a_index import extract

CORPUS = WORKSPACE / "data/ecommerce_rag/abo-dialogue-v2"


def build(destination=CORPUS):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import pymupdf
    source = WORKSPACE / "data/ecommerce_rag/abo-cards-v1"
    if destination.exists():
        raise FileExistsError("Preserve corpus versions; destination already exists: " + str(destination))
    destination.mkdir(parents=True)
    (destination / "pdfs").mkdir()
    records = json.loads((source / "selected-products.json").read_text("utf-8"))
    docs = json.loads((source / "documents.json").read_text("utf-8"))
    by_sku = {r["item"]["item_id"]: r for r in records}
    pdfmetrics.registerFont(TTFont("DialogueCard", "C:/Windows/Fonts/simhei.ttf"))
    body = ParagraphStyle("body", fontName="DialogueCard", fontSize=12, leading=18, spaceAfter=12)
    heading = ParagraphStyle("heading", parent=body, fontSize=17, leading=23, spaceAfter=18)
    total_points = 0
    registered = []
    coverage = []
    for old in docs:
        sku = old["sku"]
        item = by_sku[sku]["item"]
        points = [v["value"] for v in item.get("bullet_point", []) if v.get("language_tag") == "en_US"]
        extra = destination / "pdfs" / (sku + ".supplement.pdf")
        story = [Paragraph("产品完整要点 / COMPLETE PRODUCT NOTES", heading),
                 Paragraph(html.escape(sku + " | " + old["title"]), body),
                 Paragraph("以下为ABO历史商品记录中的英文原文，不是实测结论。退货或保修文案仅供追溯历史描述，不能据此确认当前店铺政策。", body)]
        for i, point in enumerate(points, 1):
            story += [Paragraph("Source bullet " + str(i), body), Paragraph(html.escape(point), body)]
        optional = []
        for key in ("style", "fabric_type", "finish_type", "item_shape", "model_name", "pattern", "product_description"):
            values = [v.get("value", "") for v in item.get(key, []) if v.get("language_tag") in ("en_US", None)]
            if values:
                optional.append((key, " | ".join(str(v) for v in values)))
        if optional:
            story += [Spacer(1, 15), Paragraph("其他原始属性 / OTHER SOURCE FIELDS", heading)]
            story += [Paragraph(html.escape(k + ": " + v), body) for k, v in optional]
        def footer(canvas, doc):
            canvas.setFont("DialogueCard", 8)
            canvas.drawString(38, 30, sku + " | ABO historical source | CC BY 4.0 | Supplement " + str(doc.page))
        SimpleDocTemplate(str(extra), pagesize=(595, 842), leftMargin=38, rightMargin=38,
                          topMargin=42, bottomMargin=55).build(story, onFirstPage=footer, onLaterPages=footer)
        target = destination / "pdfs" / (sku + ".pdf")
        with pymupdf.open(WORKSPACE / old["pdf"]) as original, pymupdf.open(extra) as supplement:
            merged = pymupdf.open()
            merged.insert_pdf(original)
            merged.insert_pdf(supplement)
            merged.save(target)
            count = len(merged)
            merged.close()
        extra.unlink()  # This temporary file is created by this invocation, never a historical source.
        doc = {**old, "doc_id": "abo2-" + sku, "pdf": str(target.relative_to(WORKSPACE)).replace("\\", "/"),
               "sha256": file_hash(target), "version": "abo-dialogue-v2", "page_count": count,
               "source_card_sha256": old["sha256"], "supplement_first_page": 3}
        registered.append(doc)
        total_points += len(points)
        coverage.append({"sku": sku, "english_bullets": len(points), "additional_fields": [k for k, _ in optional],
                         "pages": count, "raw_record": old["source_record"]})
    for name in ("ATTRIBUTION.txt", "SOURCE-LICENSE-CC-BY-4.0.txt", "SOURCE-README.md", "selected-products.json"):
        shutil.copy2(source / name, destination / name)
    write_json(destination / "documents.json", registered)
    write_json(destination / "coverage.json", {"version": "abo-dialogue-v2", "products": len(registered),
               "pages": sum(d["page_count"] for d in registered), "source_bullets": total_points, "items": coverage,
               "scope": "Original two pages plus all en_US source bullets and available optional fields. No current policy, new product or external data claimed.",
               "source_manifest_sha256": file_hash(source / "selected-products.json")})
    result = extract(destination)
    operation("建立独立的丰富商品资料版本", "保留旧两页卡，追加全部英文产品要点和可用风格/面料等原始属性；保留来源与授权，标注历史政策边界；新目录解析/OCR。",
              f"{len(registered)}商品，{total_points}条完整英文要点，{result['pages']}页。没有改动冻结v1数据或参考标签。",
              [destination / "coverage.json", destination / "documents.json", destination / "pages.jsonl"],
              "后续核对所有源要点在PDF文本中完整出现，并渲染检查分页。")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))

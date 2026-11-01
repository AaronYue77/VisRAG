"""Official ABO acquisition and deterministic, attributed product-card corpus."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import io
import json
import hashlib
import html
from pathlib import Path
import sys
import re
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "rag_common"))
from common import WORKSPACE, fetch, file_hash, operation, issue, timestamp, write_json, write_jsonl

SOURCE = WORKSPACE / "data/ecommerce_rag/abo-source"
OFFICIAL = "https://amazon-berkeley-objects.s3.amazonaws.com/"


def download(key, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        data = fetch(OFFICIAL + key)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return {"url": OFFICIAL + key, "file": str(destination.relative_to(WORKSPACE)),
            "bytes": destination.stat().st_size, "sha256": file_hash(destination),
            "verified_at": timestamp(), "hash_note": "Locally computed; upstream publishes no checksum."}


def acquire():
    keys = ["README.md", "LICENSE-CC-BY-4.0.txt", "index.html", "listings/README.md",
            "images/README.md", "archives/abo-listings.tar", "images/metadata/images.csv.gz"]
    records = []
    for key in keys:
        target = SOURCE / key
        # Earlier source-inspection downloads used flattened archive filenames.
        if not target.exists() and key.startswith("archives/") and (SOURCE / Path(key).name).exists():
            target = SOURCE / Path(key).name
        record = download(key, target)
        records.append(record)
        print(key, record["bytes"], flush=True)
        write_json(SOURCE / "downloads.json", records)
    operation("下载并校验 ABO 官方商品数据", "官方 S3 HTTPS + 重试；保留原始归档、图片元数据、README 和 CC BY 4.0，计算本地 SHA256。",
              f"{len(records)} 个来源文件可复核。", [SOURCE / "downloads.json"], "未用第三方抓取商品；未下载全量图片/3D/视频。")


def listings():
    archive = SOURCE / "archives/abo-listings.tar"
    if not archive.exists():
        archive = SOURCE / "abo-listings.tar"
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json.gz"):
                with gzip.open(tar.extractfile(member), "rt", encoding="utf-8") as stream:
                    for line, value in enumerate(stream, 1):
                        yield json.loads(value), {"member": member.name, "line": line}


def text_field(item, key):
    values = item.get(key) or []
    if not isinstance(values, list):
        return str(values)
    for language in ("en_US", "en_GB", "zh_CN"):
        for value in values:
            if value.get("language_tag") == language:
                return str(value["value"])
    return str(values[0]["value"]) if values else ""


def profile():
    types, router = Counter(), []
    count = 0
    for item, source in listings():
        count += 1
        types[text_field(item, "product_type") or "unknown"] += 1
        if "router" in text_field(item, "item_name").lower():
            router.append({"item": item, "source": source})
    write_json(SOURCE / "profile.json", {"products": count, "types": types, "router_examples": router[:8]})
    print(json.dumps({"products": count, "types": types.most_common(35), "routers": len(router), "examples": router[:2]}, ensure_ascii=False, indent=2))


def number(value):
    return format(float(value), ".10g")


def eligible_item(item):
    return (text_field(item, "product_type") == "CHAIR" and item.get("domain_name") == "amazon.com"
            and item.get("main_image_id") and text_field(item, "color") and text_field(item, "material")
            and item.get("item_weight") and all(k in item.get("item_dimensions", {}) for k in ("width", "height", "length")))


def select():
    """Select before seeing model outputs; no duplicate image/model or close title variants."""
    if (WORKSPACE/"data/ecommerce_rag/abo-cards-v1/frozen-test-config.json").exists():
        raise ValueError("Use a new corpus version after test freeze")
    candidates = [(item, source) for item, source in listings() if eligible_item(item)]
    candidates.sort(key=lambda x: hashlib.sha256(("abo-a-2026-11-01:" + x[0]["item_id"]).encode()).hexdigest())
    selected, title_sets, models, images = [], [], set(), set()
    for item, source in candidates:
        tokens = set(re.findall(r"[a-z]+", text_field(item, "item_name").lower())) - set("amazon brand amazonbasics basics chair chairs a an the with and for of in".split())
        model = text_field(item, "model_number").lower()
        if item["main_image_id"] in images or (model and model in models):
            continue
        if any(len(tokens & other) / max(1, len(tokens | other)) >= .45 for other in title_sets):
            continue
        selected.append({"item": item, "source": source, "family_key": "selected-family-"+item["item_id"]})
        title_sets.append(tokens); images.add(item["main_image_id"])
        if model:
            models.add(model)
        if len(selected) == 24:
            break
    if len(selected) != 24:
        raise ValueError(f"Only {len(selected)} sufficiently distinct qualifying products")
    root = WORKSPACE / "data/ecommerce_rag/abo-cards-v1"
    if (root/"frozen-test-config.json").exists():
        raise ValueError("Use a new corpus version after test freeze")
    root.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(selected):
        row["split"] = "dev" if i % 2 == 0 else "test"
        row["scan_page2"] = i % 6 in (0, 1)
    write_json(root / "selected-products.json", selected)
    with gzip.open(SOURCE / "images/metadata/images.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        metadata = {r["image_id"]: r for r in csv.DictReader(stream) if r["image_id"] in images}
    def get_image(row):
        item = row["item"]
        key = "images/original/" + metadata[item["main_image_id"]]["path"]
        record = download(key, root / "images" / (item["item_id"] + Path(key).suffix))
        record.update(image_id=item["main_image_id"], sku=item["item_id"])
        print(item["item_id"], record["bytes"], flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(get_image, selected))
    write_json(root / "images-manifest.json", downloaded)
    write_json(root / "selection.json", {"qualifying_products": len(candidates), "selected": 24, "split_counts": {"dev": 12, "test": 12},
               "selection_seed": "abo-a-2026-11-01", "strategy": "Hash order, CHAIR, amazon.com, required fields; reject shared image/model and title token Jaccard >= .45. Selected families do not repeat.",
               "split_note": "Custom application split, not an official ABO benchmark split. No training split or model training in A.",
               "snapshot_note": "Registration/download date is not source publication time or guarantee of current catalog accuracy."})
    operation("选择真实商品与下载对应原图", "固定种子散列排序，按类别/站点/字段筛选，排除相同图片/型号和近似标题；按商品家族划分开发与测试，不按页面随机拆分。",
              f"{len(candidates)} 个字段完整候选中选择24个独立商品，各12个开发/测试；24张官方原图已下载。", [root / "selection.json", root / "selected-products.json", root / "images-manifest.json"], "分组先于模型运行；无价格、订单或政策事实补造。")


def cards():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    import pymupdf as fitz
    from PIL import Image
    root = WORKSPACE / "data/ecommerce_rag/abo-cards-v1"
    if (root/"frozen-test-config.json").exists():
        raise ValueError("Use a new corpus version after test freeze")
    selected = json.loads((root / "selected-products.json").read_text("utf-8"))
    downloaded = {r["sku"]: r for r in json.loads((root / "images-manifest.json").read_text("utf-8"))}
    pdfmetrics.registerFont(TTFont("CardFont", "C:/Windows/Fonts/simhei.ttf"))
    style = ParagraphStyle("body", fontName="CardFont", fontSize=12, leading=18, textColor="#18283b")
    documents, questions, facts = [], [], []
    def paragraph(c, text, x, y, width=500, fontsize=12):
        sty = ParagraphStyle("temp", parent=style, fontSize=fontsize, leading=fontsize*1.5)
        p = Paragraph(html.escape(text), sty)
        _, height = p.wrap(width, 700)
        p.drawOn(c, x, y-height)
        return y-height-12
    def header(c, sku, page):
        c.setFillColorRGB(.08,.18,.30); c.rect(0,780,595,62,fill=1,stroke=0)
        c.setFillColorRGB(1,1,1);c.setFont("CardFont",17);c.drawString(36,808,"ABO 商品资料卡 / PRODUCT CARD")
        c.setFont("CardFont",10);c.drawString(36,788,f"商品编号 {sku}    |    PDF {page}/2")
        c.setFillColorRGB(.12,.16,.21)
        paragraph(c,"真实公开数据的排版转换件；非厂家原始手册，非实时商品信息。",36,765,fontsize=10)
        c.setFillColorRGB(.40,.43,.48);c.setFont("CardFont",8)
        c.drawString(36,39,"Data/images (c) Amazon.com | ABO | CC BY 4.0 | Layout/OCR variant by AaronYue77")
        c.drawString(36,26,"Source: amazon-berkeley-objects.s3.amazonaws.com | Saved snapshot 2026-11-01")
    for index, row in enumerate(selected):
        item, sku = row["item"], row["item"]["item_id"]
        doc_id = "abo-" + sku
        pdf = root / "pdfs" / (sku + ".pdf");pdf.parent.mkdir(exist_ok=True)
        c = canvas.Canvas(str(pdf), pagesize=(595,842), invariant=1)
        c.setTitle("ABO derived product card " + sku)
        header(c,sku,1)
        y=paragraph(c,text_field(item,"item_name"),36,724,fontsize=16)
        image=Image.open(WORKSPACE / downloaded[sku]["file"]).convert("RGB")
        ratio=min(480/image.width,300/image.height)
        c.drawImage(ImageReader(image), (595-image.width*ratio)/2, y-300, width=image.width*ratio,height=image.height*ratio, preserveAspectRatio=True,anchor='c')
        y-=320
        color, material = text_field(item,"color"), text_field(item,"material")
        for label, value in [("品牌 / brand",text_field(item,"brand")),("颜色 / color",color),("材质 / material",material),("型号 / model_number",text_field(item,"model_number") or "Not supplied")]:
            y=paragraph(c,label+": "+value,36,y,fontsize=13)
        if y < 70:
            raise ValueError("Card layout overflow: "+sku)
        c.showPage();header(c,sku,2)
        y=paragraph(c,"规格表 / SPECIFICATIONS - source item_dimensions and item_weight",36,724,fontsize=14)
        dimension_facts={}
        for key, label in [("width","宽度 / width"),("height","高度 / height"),("length","长度 / length")]:
            value=item["item_dimensions"][key]
            dimension_facts[key]={"value":number(value["value"]),"unit":value["unit"]}
            y=paragraph(c,label+": "+number(value["value"])+" "+value["unit"],48,y,fontsize=15)
            c.setStrokeColorRGB(.82,.85,.88);c.line(36,y+4,559,y+4)
        weight=item["item_weight"][0]
        dimension_facts["weight"]={"value":number(weight["value"]),"unit":weight["unit"]}
        y=paragraph(c,"重量 / weight: "+number(weight["value"])+" "+weight["unit"],48,y,fontsize=15)
        y=paragraph(c,"产品特点 / bullet_point (original source text)",36,y-10,fontsize=13)
        points=[v["value"] for v in item.get("bullet_point",[]) if v.get("language_tag")=="en_US"][:2]
        for value in points:
            # Keep source text verbatim; fit at most two complete source bullet points.
            p=Paragraph(html.escape(value),style);_,height=p.wrap(500,700)
            if y-height < 100:
                break
            p.drawOn(c,36,y-height);y-=height+12
        paragraph(c,"资料范围：仅图示、上述静态属性及已显示原文。未列字段不能推断为支持或不支持。",36,95,fontsize=10)
        c.showPage();c.save()
        if row["scan_page2"]:
            # Controlled OCR fixture: same authentic facts, rasterized second page.
            source=fitz.open(pdf); target=fitz.open()
            target.insert_pdf(source,from_page=0,to_page=0)
            raster=source[1].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            page=target.new_page(width=595,height=842);page.insert_image(page.rect,stream=raster.tobytes("png"))
            temp=pdf.with_suffix(".raster.pdf");target.save(temp);target.close();source.close();temp.replace(pdf)
        document={"doc_id":doc_id,"sku":sku,"title":text_field(item,"item_name"),"pdf":str(pdf.relative_to(WORKSPACE)).replace("\\","/"),
                  "sha256":file_hash(pdf),"store_id":"amazon.com","doc_type":"derived_product_card","version":"abo-card-v1",
                  "snapshot_date":"2026-11-01","source_publication_date":None,"source_type":"official_ABO_derived_PDF",
                  "family_key":row["family_key"],"split":row["split"],"ocr_fixture":row["scan_page2"],
                  "source_record":row["source"],"source_image_id":item["main_image_id"]}
        documents.append(document)
        item_facts={"sku":sku,"color":color,"material":material,**dimension_facts}
        facts.append(item_facts)
        request={"skus":[sku],"store_id":"amazon.com","as_of":"2026-11-01"}
        specs=[("color","颜色 / color 是什么？",color,1),("material","材质 / material 是什么？",material,1),
               ("width","规格表中的宽度 / width 是多少？保留资料中的单位。",dimension_facts["width"]["value"]+" "+dimension_facts["width"]["unit"],2),
               ("weight","规格表中的重量 / weight 是多少？保留资料中的单位。",dimension_facts["weight"]["value"]+" "+dimension_facts["weight"]["unit"],2),
               ("unknown","请给出内置电池的容量（mAh）；没报告就拒答。","NOT_FOUND",None)]
        for field,text,answer,page in specs:
            q={"question_id":f"{sku}-{field}","question":f"{sku} 的{text}","request":request,"split":row["split"],"family_key":row["family_key"],
               "reference_answer":answer,"answer_aliases":[answer],"reference_pages":[f"{doc_id}:p{page:04d}"] if page else [],
               "expected_action":"answer" if page else "not_found","field":field,"gold_source_field":field,"annotation_type":"source_field_template_verified",
               "expected_fact":item_facts[field] if page else None}
            questions.append(q)
    for split in ("dev","test"):
        product=next(d for d in documents if d["split"]==split);sku=product["sku"]
        cases=[("missing","这款椅子的宽度是多少？",{},"clarify"),
               ("unknown-sku","B0ZZZZZZZZ 的宽度？",{},"not_found"),
               ("order","请查订单物流",{},"requires_tool"),
               ("price",f"{sku} 现在多少钱？",{"skus":[sku]},"requires_tool"),
               ("store",f"{sku} 的颜色？",{"skus":[sku],"store_id":"unconnected_shop"},"not_found"),
               ("date",f"{sku} 的颜色？",{"skus":[sku],"as_of":"2020-01-01"},"not_found"),
               ("policy","这个店铺的退货政策？",{},"not_found"),
               ("scope","明天的天气怎么样？",{},"unsupported")]
        for name,text,request,action in cases:
            questions.append({"question_id":split+"-route-"+name,"question":text,"request":request,"split":split,"family_key":product["family_key"],
                "reference_answer":action,"answer_aliases":[action],"reference_pages":[],"expected_action":action,"field":"route",
                "annotation_type":"custom_business_boundary_case"})
    write_json(root/"documents.json",documents);write_json(root/"facts.json",facts);write_jsonl(root/"questions.jsonl",questions)
    write_json(root/"protocol.json",{"version":"abo-a-v1","created_at":timestamp(),"questions":len(questions),"documents":len(documents),
       "freeze":"Product groups and labels frozen before first VLM run. Develop on dev; test only after configuration is chosen.",
       "sources":"Official ABO records/images, converted to two-page cards; questions are custom, NOT official ABO QA annotations.",
       "scan_note":"8 page-2 raster conversions are controlled OCR fixtures, NOT naturally scanned manufacturer PDFs.",
       "limitations":["CHAIR category, not router manuals", "Catalog attributes, not live policy/price/order facts", "Two-page cards are simpler than long industrial manuals", "Template questions; no claim of broad user coverage"]})
    operation("生成有来源的真实商品 PDF 与冻结评测标签", "原始 ABO 属性值/单位直接排版，不改写事实；24个商品各2页，8页作受控光栅化 OCR 样例；按商品家族预先划分。",
              f"24份48页商品资料卡，{len(questions)} 道自建题；不是官方问答基准或厂家原始PDF。", [root/"documents.json",root/"questions.jsonl",root/"protocol.json"], "原始 JSON 记录和 image_id 可追溯；随后执行渲染检查和模型前标签校验。")


def annotate():
    from common import read_jsonl
    root=WORKSPACE/"data/ecommerce_rag/abo-cards-v1"
    if (root/"frozen-test-config.json").exists():
        raise ValueError("Cannot revise labels after test freeze")
    questions=[q for q in read_jsonl(root/"questions.jsonl") if q["field"] not in ("visual_armrest","comparison")]
    documents=json.loads((root/"documents.json").read_text("utf-8"))
    facts={r["sku"]:r for r in json.loads((root/"facts.json").read_text("utf-8"))}
    # Manually inspected all 48 rendered pages and the first 12 source product photos.
    armrest={"B07TLFBRKD":"Yes","B07HYYJKMD":"Yes","B07124WN69":"Yes","B075X4JB1L":"Yes",
             "B07GNF6Q9L":"No","B00EUL2B16":"No","B07B4M5BYY":"Yes","B07DBB7H9G":"Yes",
             "B07QK2JK3Y":"No","B084T7TZDG":"Yes","B082QCHXNK":"Yes","B07KXF19FN":"Yes"}
    for q in questions:
        if q["field"]=="color":
            q["question"]=q["request"]["skus"][0]+" 的资料卡 color（颜色）字段原文是什么？请保留原文，不根据照片改写该字段。"
    for d in documents:
        sku=d["sku"]
        if sku not in armrest:continue
        questions.append({"question_id":sku+"-visual-armrest","question":sku+" 的商品图片中，椅子是否有扶手？只回答 Yes 或 No。", "request":{"skus":[sku]},
            "split":d["split"],"family_key":d["family_key"],"reference_answer":armrest[sku],"answer_aliases":[armrest[sku]],
            "expected_fact":armrest[sku],"expected_action":"answer","reference_pages":[d["doc_id"]+":p0001"],"field":"visual_armrest",
            "annotation_type":"manual_visual_inspection","annotation_evidence":"reports/ecommerce_rag/a-pdf-quality-v1/page-1-contact.png"})
    for split in ("dev","test"):
        subset=[d for d in documents if d["split"]==split]
        for i in (0,2):
            pair=subset[i:i+2]; skus=[d["sku"] for d in pair]
            expected=[{"sku":sku,**facts[sku]["width"]} for sku in skus]
            questions.append({"question_id":split+f"-compare-{i}","question":"比较 "+" 和 ".join(skus)+" 的规格表宽度 / width，按商品编号分别给出原始数值和单位。",
                "request":{"skus":skus},"split":split,"family_key":pair[0]["family_key"],"document_families":[d["family_key"] for d in pair],
                "reference_answer":"; ".join(f"{v['sku']}: {v['value']} {v['unit']}" for v in expected),"answer_aliases":[],"expected_fact":expected,
                "expected_action":"answer","reference_pages":[d["doc_id"]+":p0002" for d in pair],"field":"comparison","annotation_type":"source_field_pair"})
    write_jsonl(root/"questions.jsonl",questions)
    protocol=json.loads((root/"protocol.json").read_text("utf-8"));protocol.update(questions=len(questions),question_counts=dict(Counter(q["split"] for q in questions)),
      annotation_note="12 photo armrest labels manually inspected; 4 cross-product width questions; attribute and boundary questions are source-derived custom templates.",
      data_quality_note="B00EUL2B16 source color is 'Pierced Splat Back', a structural description. Preserve raw record; color questions explicitly ask source field, not visible true color. No silent correction.")
    write_json(root/"protocol.json",protocol)
    operation("人工图像核验与评测标签冻结前质检","检查48页排版和12张商品照片，标注扶手；补4道跨商品问题；原始color字段异常单独登记，不擅自改官方记录。",
              f"{len(questions)} 道题，开发/测试各76题；仍未运行VLM。",[root/"questions.jsonl",root/"protocol.json"],"问题/图片/商品组划分与来源可核验；官方ABO不是官方问答标注。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "profile", "select", "cards", "annotate"])
    args = parser.parse_args()
    try:
        {"download":acquire,"profile":profile,"select":select,"cards":cards,"annotate":annotate}[args.command]()
    except Exception as error:
        key = issue(f"ABO {args.command} 失败：{type(error).__name__}: {error}", "官方连接或数据读取失败；保留已下载文件。", "同命令可重试，已完成文件不重复下载。")
        operation("ABO 数据操作失败", args.command, str(error), status="失败", related_issues=key)
        raise

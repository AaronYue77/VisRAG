"""Use original listing fields and actual photo pages for different evidence needs."""
import json
import re

from full_cards import choose_pages, evidence_fields


VERSION = "source-evidence-v1"
FIELDS = {
    "brand": ("品牌", "brand"),
    "color": ("颜色", "color"),
    "material": ("材质", "材料", "material"),
    "model_number": ("型号", "model_number"),
    "width": ("宽度", "多宽", "width"),
    "height": ("高度", "多高", "height"),
    "length": ("长度", "多长", "length"),
    "weight": ("重量", "有多重", "多重", "weight"),
}
ALIASES = {alias: field for field, aliases in FIELDS.items() for alias in aliases}
FIELD_NAMES = "|".join(sorted(ALIASES, key=len, reverse=True))
FIELD_REQUEST = re.compile(
    rf"(?P<fields>(?:{FIELD_NAMES})(?:(?:和|、|以及)(?:{FIELD_NAMES}))*)"
    r"\s*(?:字段)?\s*(?:分别)?\s*(?:原文|原始值|数值|值)?\s*"
    r"(?:是什么|是多少|有多少|多少|是什么样的|呢|吗)?",
    re.I,
)
LISTING_REF = re.compile(
    r"(?<![A-Z0-9])(?:B[0-9][A-Z0-9]{8}|[0-9]{9}[0-9X])"
    r"(?:@[a-z0-9.-]+)?(?![A-Z0-9])",
    re.I,
)
PHOTO_REQUEST = re.compile(
    r"图片|照片|外观|图案|photo|image|appearance|侧面.{0,12}(?:口袋|侧袋|水壶袋)|"
    r"(?:有没有|有无|有)\s*(?:水壶袋|侧袋|扶手)",
    re.I,
)


def requested_fields(question):
    """Recognize plain source lookups without interpreting units or comparisons."""
    text = question.strip().rstrip("？?。！! ")
    text = re.sub(r"^(?:请问|请告诉我|麻烦告诉我|我想知道)\s*", "", text)
    refs = list(LISTING_REF.finditer(text))
    if refs:
        if text[:refs[0].start()].strip():
            return ()
        if any(not re.fullmatch(r"\s*(?:和|与|、|,|，)\s*", text[a.end():b.start()])
               for a, b in zip(refs, refs[1:])):
            return ()
        text = text[refs[-1].end():].lstrip()
    else:
        text = re.sub(r"^(?:那么|那)?(?:这(?:两款|两个|两件|款|件|双|把|个商品)|它|该商品|所选商品)?",
                      "", text).lstrip()
    text = re.sub(r"^(?:的\s*)?(?:商品资料里记录的\s*)?", "", text)
    match = FIELD_REQUEST.fullmatch(text)
    if not match:
        return ()
    return tuple(dict.fromkeys(ALIASES[name.casefold()] for name in
                               re.split(r"以及|和|、", match["fields"])))


def split_photo_request(question):
    """Split a plain field lookup followed by a photo question about the same item."""
    parts = re.split(r"[，,；;？?]\s*|以及", question, maxsplit=1)
    if len(parts) == 2 and PHOTO_REQUEST.search(parts[1]) and not LISTING_REF.search(parts[1]):
        fields = requested_fields(parts[0])
        if fields:
            return fields, parts[1]
    return (), question


def read_fields(products, fields):
    answers, claims, evidence = [], [], []
    missing = 0
    for product in products:
        key = product["listing_key"]
        values = evidence_fields(json.loads(product["raw_json"]))
        for field in fields:
            actual = list(dict.fromkeys(value.strip() for name, value in values
                                        if name == field and value.strip()))
            citation = f"{key}:metadata:{field}"
            label = FIELDS[field][0]
            if not actual:
                missing += 1
                answers.append(f"{key} 的{label}在这条商品记录中未报告。")
                continue
            source = {"member": product["source_member"], "line": product["source_line"],
                      "sha256": product["raw_sha256"]}
            claims.append({"listing_key": key, "field": field, "values": actual,
                           "citation": citation, "source": source})
            evidence.append({"page_id": citation, "doc_id": key, "page": None,
                             "sku": product["sku"], "listing_key": key,
                             "rank": len(evidence) + 1, "score": 1.0,
                             "text_source": "original_metadata_field", "source": source,
                             "field": field})
            rendered = " / ".join(actual)
            if len(actual) > 1:
                rendered += "（源记录有多个值）"
            answers.append(f"{key} 的{label}为 {rendered}。")
    return {
        "action": "answer" if claims else "not_found",
        "answer": "\n".join(answers),
        "answer_source": "original_metadata_fields",
        "claims": claims,
        "citations": [claim["citation"] for claim in claims],
        "retrieved_pages": evidence,
        "retrieval_trace": {"requested_fields": list(fields), "missing_fields": missing},
    }


def photo_pages(document, question):
    pages = choose_pages(document, question, budget=len(document["pages"]))
    return [page for page in pages if page["photo_page"]][:1]


def alternate_pages(documents, first, question, budget):
    """Replace at most one page with a genuinely new, query-matching page."""
    existing = {page["page_id"] for page in first}
    for doc in documents:
        for page in choose_pages(doc, question, budget=len(doc["pages"])):
            if page["page_id"] in existing or page["score"] <= 0:
                continue
            same_product = [item for item in first if item["listing_key"] == page["listing_key"]]
            if len(first) < budget:
                return [*first, page]
            if same_product:
                replaced = min(same_product, key=lambda item: item["score"])
                return [page if item["page_id"] == replaced["page_id"] else item for item in first]
    return []

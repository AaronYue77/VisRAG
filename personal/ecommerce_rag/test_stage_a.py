import json
from pathlib import Path
import tempfile
import unittest

from stage_a_index import route, chunks, BM25Retriever, DEFAULT_CORPUS
from stage_a import fact_match, score
from common import read_jsonl

DOCS=[{"sku":"B07124WN69","store_id":"amazon.com","snapshot_date":"2026-11-01"},
      {"sku":"B075X4JB1L","store_id":"amazon.com","snapshot_date":"2026-11-01"}]


class StageATests(unittest.TestCase):
    def test_known_real_asin_can_route_without_demo_prefix(self):
        self.assertEqual(route("B07124WN69 的宽度",{},DOCS)["skus"],["B07124WN69"])

    def test_context_conflict_requires_clarification(self):
        self.assertEqual(route("B07124WN69 的宽度",{"skus":["B075X4JB1L"]},DOCS)["reason"],"context_conflict")

    def test_unknown_asin_not_silently_matched(self):
        self.assertEqual(route("B0ZZZZZZZZ 的宽度",{},DOCS)["action"],"not_found")

    def test_live_business_must_not_be_answered_from_static_cards(self):
        for text in ("B07124WN69 库存", "查订单", "帮我退款"):
            self.assertEqual(route(text,{},DOCS)["action"],"requires_tool")

    def test_static_snapshot_is_not_historical_effectivity_claim(self):
        self.assertEqual(route("宽度",{"skus":["B07124WN69"],"as_of":"2020-01-01"},DOCS)["reason"],"no_historical_snapshot")

    def test_store_scope_enforced(self):
        self.assertEqual(route("宽度",{"skus":["B07124WN69"],"store_id":"other"},DOCS)["reason"],"store_scope")

    def test_gold_keys_in_request_cannot_change_route(self):
        request={"skus":["B07124WN69"]}
        self.assertEqual(route("宽度",request,DOCS),route("宽度",{**request,"reference_answer":"SECRET","reference_pages":["wrong"],"expected_action":"answer"},DOCS))

    def test_invalid_date_and_sku_schema_rejected(self):
        for request in ({"as_of":"yesterday"},{"skus":"B07124WN69"}):
            with self.assertRaises(ValueError):route("宽度",request,DOCS)

    def test_overlapping_chunks_never_cross_parent_pages(self):
        pages=[{"page_id":p,"text":"x","blocks":[{"text":str(i)} for i in range(8)]} for p in ("p1","p2")]
        values=chunks(pages,"blocks",1)
        self.assertEqual(values[0]["text"].splitlines()[-1],values[1]["text"].splitlines()[0])
        self.assertEqual(set(v["page_id"] for v in values),{"p1","p2"})

    def test_numeric_scoring_preserves_unit_and_rejects_extra_wrong_number(self):
        fact={"value":"18.5","unit":"inches"}
        self.assertEqual(fact_match("宽度18.5英寸",fact),1)
        self.assertEqual(fact_match("18.5 cm",fact),0)
        self.assertEqual(fact_match("18.5 inches, 20 inches",fact),0)

    def test_text_scoring_handles_case_space_and_hyphen(self):
        self.assertEqual(fact_match("材质：Solid-Wood","solid wood"),1)

    def test_citation_validity_alone_is_not_factual_support(self):
        q={"expected_action":"answer","expected_fact":"red","reference_pages":["a"]}
        row={"action":"answer","answer":"blue","citations":["a"],"retrieved_pages":[{"page_id":"a"}]}
        self.assertEqual(score(row,q)["citation_gold_recall"],1)
        self.assertEqual(score(row,q)["evidence_aligned_fact"],0)

    def test_corpus_separates_products_and_image_ids(self):
        docs=json.loads((DEFAULT_CORPUS/"documents.json").read_text("utf-8"))
        dev=[d for d in docs if d["split"]=="dev"];test=[d for d in docs if d["split"]=="test"]
        for key in ("sku","family_key","source_image_id","sha256"):
            self.assertFalse({d[key] for d in dev}&{d[key] for d in test})
        pages={p["page_id"] for p in read_jsonl(DEFAULT_CORPUS/"pages.jsonl")}
        for q in read_jsonl(DEFAULT_CORPUS/"questions.jsonl"):
            self.assertTrue(set(q["reference_pages"])<=pages)

    def test_multi_product_query_reserves_each_sku_and_never_duplicates_pages(self):
        r=BM25Retriever();docs=r.documents[:2];skus=[d["sku"] for d in docs]
        result=r.search("比较宽度",{"skus":skus},2)
        self.assertEqual({p["sku"] for p in result["pages"]},set(skus))
        self.assertEqual(len({p["page_id"] for p in result["pages"]}),2)

    def test_document_cache_has_ocr_and_native_sources(self):
        pages=read_jsonl(DEFAULT_CORPUS/"pages.jsonl")
        self.assertEqual(sum(p["text_source"]=="ocr" for p in pages),8)
        self.assertTrue(all(p["blocks"] and p["text"] for p in pages))


if __name__=="__main__":unittest.main()

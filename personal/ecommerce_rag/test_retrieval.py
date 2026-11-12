import unittest
from retrieval import active, eligible, plan, rrf
from stage_a import decode_business_output

OLD = {"sku": "DN-R100", "store_id": "demo_store", "valid_from": "2025-01-01", "valid_to": "2025-12-31", "doc_type": "product_manual"}
NEW = {**OLD, "valid_from": "2026-01-01", "valid_to": None}
OTHER = {**NEW, "sku": "DN-R200"}
POLICY = {**NEW, "sku": "*", "doc_type": "store_policy"}
DOCS = [OLD, NEW, OTHER, POLICY]

class BusinessRetrievalTests(unittest.TestCase):
    def test_explicit_missing_information_text_is_abstention(self):
        raw = '{"action":"answer","answer":"未报告。请查阅补充资料。","citations":["P1"]}'
        result = decode_business_output(raw, [{"page_id": "a:p0001"}])
        self.assertEqual(result["action"], "not_found")
        self.assertEqual(result["output_guard"], "explicit_missing_information_phrase")

    def test_supported_negative_fact_is_not_abstention(self):
        raw = '{"action":"answer","answer":"不支持 PoE。","citations":["P1"]}'
        self.assertEqual(decode_business_output(raw, [{"page_id": "a:p0001"}])["action"], "answer")

    def test_explicit_abstention_becomes_control_status_with_raw_preservable(self):
        raw = '{"action":"not_found","answer":"资料未报告","citations":["P1"]}'
        decoded = decode_business_output(raw, [{"page_id": "a:p0001"}])
        self.assertEqual(decoded["action"], "not_found")
        self.assertEqual(decoded["answer"], "NOT_FOUND")
        self.assertEqual(decoded["output_guard"], "explicit_abstention_canonicalized")

    def test_answer_cannot_cite_page_outside_context(self):
        decoded = decode_business_output('{"action":"answer","answer":"42","citations":["P99"]}', [{"page_id": "a:p0001"}])
        self.assertEqual(decoded["action"], "not_found")
        self.assertEqual(decoded["invalid_citations"], ["P99"])

    def test_effective_date_excludes_expired_manual(self):
        self.assertFalse(active(OLD, "demo_store", "2026-11-01"))
        self.assertTrue(active(NEW, "demo_store", "2026-11-01"))

    def test_historical_request_uses_old_version_only(self):
        p = plan("保修期多久", {"skus": ["DN-R100"], "as_of": "2025-06-01"}, DOCS)
        self.assertTrue(eligible(OLD, p))
        self.assertFalse(eligible(NEW, p))

    def test_wrong_model_or_shop_never_enters_candidates(self):
        p = plan("WAN 口速率", {"skus": ["DN-R100"]}, DOCS)
        self.assertTrue(eligible(NEW, p))
        self.assertFalse(eligible(OTHER, p))
        self.assertFalse(eligible({**NEW, "store_id": "another_shop"}, p))

    def test_missing_model_requires_clarification(self):
        self.assertEqual(plan("这款有几个 LAN 口", {}, DOCS)["action"], "clarify")

    def test_inconsistent_explicit_context_is_not_silently_replaced(self):
        self.assertEqual(plan("DN-R200 有几个接口", {"skus": ["DN-R100"]}, DOCS)["action"], "clarify")

    def test_unknown_model_does_not_fall_back_to_similar_name(self):
        self.assertEqual(plan("DN-R900 有几个接口", {}, DOCS)["action"], "not_found")

    def test_order_and_price_are_not_answered_from_manual(self):
        for text in ("我的订单在哪里", "DN-R100 现在卖多少钱", "现在有货吗"):
            self.assertEqual(plan(text, {}, DOCS)["action"], "requires_tool")

    def test_store_policy_needs_no_product_model(self):
        p = plan("DemoMall 的退货期限", {}, DOCS)
        self.assertEqual(p["action"], "retrieve")
        self.assertTrue(eligible(POLICY, p))
        self.assertFalse(eligible(NEW, p))

    def test_rrf_uses_rank_and_does_not_double_count_duplicates(self):
        order, scores = rrf([["a", "a", "b"], ["b", "a"]])
        self.assertAlmostEqual(scores["a"], 1/61 + 1/62)
        self.assertAlmostEqual(scores["b"], 1/63 + 1/61)
        self.assertEqual(order[0], "a")

    def test_multi_product_request_is_preserved(self):
        p = plan("比较 DN-R100 和 DN-R200", {}, DOCS)
        self.assertEqual(p["skus"], ["DN-R100", "DN-R200"])

if __name__ == "__main__":
    unittest.main(verbosity=2)

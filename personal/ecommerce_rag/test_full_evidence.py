from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from full_app import FullApplication
from common import WORKSPACE
from full_evidence import alternate_pages, requested_fields, split_photo_request
from full_policy import policy_answer


class EvidenceTests(unittest.TestCase):
    def test_source_lookup_is_not_an_unsupported_unit_conversion(self):
        self.assertEqual(requested_fields("B07PGMSL4D@amazon.com 有多重？"), ("weight",))
        self.assertEqual(requested_fields("这款重量换算成公斤是多少？"), ())
        self.assertEqual(requested_fields("包装重量是多少？"), ())

    def test_mixed_question_preserves_field_and_photo_clause(self):
        fields, photo = split_photo_request("这款高度是多少，照片上正面拉链是什么颜色？")
        self.assertEqual(fields, ("height",))
        self.assertIn("拉链", photo)

    def test_alternative_preserves_both_product_slots_and_excludes_duplicate(self):
        first = {"page_id": "P1", "listing_key": "A", "score": 1}
        second = {"page_id": "P2", "listing_key": "B", "score": 2}
        new = {"page_id": "P3", "listing_key": "A", "score": 2}
        with patch("full_evidence.choose_pages", return_value=[first, new]):
            found = alternate_pages([{"pages": [first, new]}], [first, second], "query", 2)
        self.assertEqual(found, [new, second])

    def test_policy_cannot_cross_site_or_cover_unknown_qualifications(self):
        self.assertEqual(policy_answer("amazon.ca 的退货政策是什么？")["action"], "not_found")
        self.assertEqual(policy_answer("amazon.com 的退货政策是什么？", "amazon.ca")["action"], "clarify")
        self.assertEqual(policy_answer("amazon.com 节日期间的退货政策是什么？")["action"], "not_found")
        self.assertIsNone(policy_answer("我的 amazon.com 退款到哪里了？"))

    def test_full_search_handles_adjacent_chinese_color_and_category(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / "cache/tmp") as folder:
            app = FullApplication(report_root=Path(folder), audit=False)
            try:
                result = app.search_catalog("黑色双肩包")
                self.assertEqual(result["trace"]["effective_query"], "black backpack")
                self.assertTrue(result["items"])
                self.assertEqual({r["product_type"] for r in result["items"]}, {"BACKPACK"})
            finally:
                app.close()

    def test_rejected_listing_filters_do_not_replace_last_selection(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / "cache/tmp") as folder:
            app = FullApplication(report_root=Path(folder), audit=False)
            try:
                session = app.session()
                session.respond("B00T8JBREW@amazon.com")
                result = session.respond("B07PGMSL4D@amazon.com 的重量是多少？", {"product_type": "CHAIR"})
                self.assertEqual(result["action"], "clarify")
                self.assertEqual(result["current_listing_keys"], ["B00T8JBREW@amazon.com"])
            finally:
                app.close()

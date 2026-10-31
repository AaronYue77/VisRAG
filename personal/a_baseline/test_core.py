"""Fast safeguards for evaluation validity; no model download or GPU needed."""
import unittest
from pathlib import Path

from core import answer_metrics, retrieval_metrics, verify_questions, build_messages, parse_answer
from common import read_jsonl, write_jsonl


class EvaluationSafeguards(unittest.TestCase):
    def test_pdf_unicode_separators_do_not_split_json_records(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.jsonl"
            rows = [{"text": "first\u2028second\u0085third"}, {"text": "next"}]
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)

    def test_multi_page_recall_and_first_hit_are_different(self):
        metrics = retrieval_metrics(["d:p0003", "d:p0002"], ["d:p0001", "d:p0002"])
        self.assertEqual(metrics, {"hit_at_k": 1.0, "recall_at_k": .5, "reciprocal_rank": .5})

    def test_repeated_page_does_not_inflate_recall(self):
        self.assertEqual(retrieval_metrics(["a", "a"], ["a", "b"])["recall_at_k"], .5)

    def test_document_overlap_between_dev_and_test_is_rejected(self):
        pages = [{"page_id": "d:p0001"}, {"page_id": "d:p0002"}]
        questions = [{"question_id": str(i), "reference_answer": "x", "reference_pages": [f"d:p000{i}"],
                      "document_ids": ["d"], "split": split} for i, split in [(1, "dev"), (2, "test")]]
        with self.assertRaisesRegex(ValueError, "Document split leakage"):
            verify_questions(questions, pages)

    def test_zero_based_or_missing_gold_page_is_rejected(self):
        question = {"question_id": "x", "reference_answer": "x", "reference_pages": ["d:p0000"], "document_ids": ["d"], "split": "dev"}
        with self.assertRaisesRegex(ValueError, "Unknown gold page"):
            verify_questions([question], [{"page_id": "d:p0001"}])

    def test_reader_prompt_excludes_gold_fields_in_page_record(self):
        page = {"page_id": "d:p0001", "page": 1, "title": "Doc", "reference_answer": "SECRET_GOLD_123", "text": "SECRET_OCR_456"}
        prompt = str(build_messages("What is shown?", [page], [Path("page.png")]))
        self.assertNotIn("SECRET_GOLD_123", prompt)
        self.assertNotIn("SECRET_OCR_456", prompt)  # reader receives pixels, not hidden text
        self.assertIn("page.png", prompt)

    def test_invalid_json_is_not_silently_repaired(self):
        for raw in ['{"answer":"4000","citations":"P1"}', '{"answer":"2.8","citations":[P1]}']:
            parsed = parse_answer(raw, [{"page_id": "d:p0001"}])
            self.assertFalse(parsed["parse_ok"])
            self.assertEqual(parsed["answer"], raw)

    def test_citations_map_only_to_supplied_pages(self):
        parsed = parse_answer('{"answer":"512","citations":["P2","P99"]}', [{"page_id": "d:p0003"}, {"page_id": "d:p0007"}])
        self.assertEqual(parsed["citations"], ["d:p0007"])
        self.assertEqual(parsed["invalid_citations"], ["P99"])

    def test_numeric_aliases_keep_wrong_number_distinct(self):
        self.assertEqual(answer_metrics("2,048", ["2048"])["normalized_em"], 1.0)
        self.assertEqual(answer_metrics("76.2%", ["76.2"])["normalized_em"], 1.0)
        self.assertEqual(answer_metrics("76.3", ["76.2"])["normalized_em"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

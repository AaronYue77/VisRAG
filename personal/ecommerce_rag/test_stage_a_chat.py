"""Conversation-state checks use a temporary fake backend; no fabricated benchmark rows."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage_a_chat import ChatSession, run_chat
from stage_a_index import route
from common import WORKSPACE
from stage_a_app import Application
from audit import format_question

DOCS=[{"sku":"B07124WN69","title":"Chair A","store_id":"amazon.com","snapshot_date":"2026-11-01"},
      {"sku":"B075X4JB1L","title":"Chair B","store_id":"amazon.com","snapshot_date":"2026-11-01"}]


class FakeApp:
    def __init__(self, root):
        self.root=root
        self.retriever=type("Retriever",(),{"documents":DOCS})()
        self.calls=[]
        self.model_load_count=0
        self.fail_next=False
        self.runs=0

    def create_run(self,prefix="chat"):
        self.runs+=1
        output=self.root/(prefix+str(self.runs));output.mkdir()
        return output

    def local_reply(self,question,request,action,answer,**kwargs):
        return {"action":action,"answer":answer,"citations":[],"retrieved_pages":[],"status":"error" if kwargs.get("error") else "ok",
                "record_txt":str(kwargs["output"]/"逐题结果.txt"),"model_load_count":self.model_load_count}

    def ask(self,question,request,**kwargs):
        self.calls.append((question,json.loads(json.dumps(request))))
        if self.fail_next:
            self.fail_next=False
            raise RuntimeError("controlled test failure")
        plan=route(question,request,DOCS)
        if plan["action"]!="retrieve":
            return self.local_reply(question,request,plan["action"],plan["answer"],**kwargs)
        if not self.model_load_count:self.model_load_count=1
        return {"action":"answer","answer":"25 inches","citations":["test-page"],"retrieved_pages":[],"status":"ok",
                "record_txt":str(kwargs["output"]/"逐题结果.txt"),"model_load_count":self.model_load_count}


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=WORKSPACE/"cache/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.app=FakeApp(Path(self.tmp.name))
        self.session=ChatSession(self.app)

    def test_greeting_never_calls_model_or_retrieval(self):
        self.assertEqual(self.session.respond("你好！")["action"],"greeting")
        self.assertEqual(self.app.calls,[])
        self.assertEqual(self.app.model_load_count,0)

    def test_greeting_prefix_does_not_swallow_a_product_question(self):
        self.session.respond("你好，B07124WN69 的宽度是多少？")
        self.assertEqual(len(self.app.calls),1)

    def test_pronoun_keeps_original_question_and_injects_only_product_context(self):
        self.session.respond("查询 B07124WN69 的宽度")
        self.session.respond("它的重量呢？")
        self.assertEqual(self.app.calls[-1][0],"它的重量呢？")
        self.assertEqual(self.app.calls[-1][1]["skus"],["B07124WN69"])
        self.assertNotIn("history",self.app.calls[-1][1])
        self.assertEqual(self.app.model_load_count,1)

    def test_explicit_product_switch_replaces_old_context(self):
        self.session.respond("B07124WN69 的宽度")
        self.session.respond("B075X4JB1L 的重量")
        self.assertEqual(self.app.calls[-1][1]["skus"],["B075X4JB1L"])

    def test_unknown_product_clears_old_context(self):
        self.session.respond("B07124WN69 的宽度")
        self.assertEqual(self.session.respond("B0ZZZZZZZZ 的宽度")["action"],"not_found")
        self.assertEqual(self.session.state["skus"],[])
        self.assertEqual(self.session.respond("它的重量呢")["action"],"clarify")

    def test_conflicting_api_context_clarifies_and_preserves_previous_product(self):
        self.session.respond("B07124WN69")
        result=self.session.respond("B075X4JB1L 的宽度",{"skus":["B07124WN69"]})
        self.assertEqual(result["action"],"clarify")
        self.assertEqual(self.session.state["skus"],["B07124WN69"])

    def test_reset_clears_context_but_does_not_unload_model(self):
        self.session.respond("B07124WN69 的宽度")
        self.assertEqual(self.session.respond("重置")["action"],"reset")
        self.assertEqual(self.session.state["skus"],[])
        self.assertEqual(self.app.model_load_count,1)

    def test_two_product_singular_followup_requires_clarification(self):
        self.session.respond("比较 B07124WN69 和 B075X4JB1L 的宽度")
        count=len(self.app.calls)
        self.assertEqual(self.session.respond("它的重量呢")["action"],"clarify")
        self.assertEqual(len(self.app.calls),count)
        self.assertEqual(self.session.respond("它们分别多重？")["action"],"answer")

    def test_sessions_do_not_share_selected_products(self):
        other=ChatSession(self.app)
        self.session.respond("B07124WN69")
        self.assertEqual(other.respond("它的重量呢")["action"],"clarify")

    def test_error_is_saved_and_next_turn_still_works(self):
        self.session.respond("B07124WN69")
        self.app.fail_next=True
        with patch("stage_a_chat.issue",return_value="TEST-ISSUE"):
            self.assertEqual(self.session.respond("它的重量呢")["action"],"error")
        self.assertEqual(self.session.respond("它的宽度呢")["action"],"answer")
        self.assertEqual(self.session.state["skus"],["B07124WN69"])
        transcript=self.session.transcript.read_text("utf-8")
        self.assertIn("本轮处理失败",transcript)

    def test_public_request_rejects_reference_labels(self):
        with self.assertRaises(ValueError):self.session.respond("宽度",{"reference_answer":"leak"})

    def test_first_model_failure_keeps_explicit_selection_for_retry(self):
        self.app.fail_next=True
        with patch("stage_a_chat.issue",return_value="TEST-ISSUE"):
            self.assertEqual(self.session.respond("B07124WN69 的宽度")["action"],"error")
        self.assertEqual(self.session.state["skus"],["B07124WN69"])
        self.assertEqual(self.session.respond("它的重量呢")["action"],"answer")

    def test_exit_records_and_ends_only_this_session(self):
        self.assertEqual(self.session.respond("退出")["action"],"exit")
        self.assertTrue(self.session.ended)
        with self.assertRaises(ValueError):self.session.respond("你好")

    def test_terminal_eof_exits_without_exception(self):
        with patch("builtins.input",side_effect=EOFError),patch("builtins.print"),patch("stage_a_chat.operation"):
            run_chat(self.app)

    def test_terminal_ctrl_c_exits_without_traceback(self):
        with patch("builtins.input",side_effect=KeyboardInterrupt),patch("builtins.print"),patch("stage_a_chat.operation"):
            run_chat(self.app)


class PersistedConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=WORKSPACE/"cache/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/"session";self.output.mkdir()
        (self.output/"config.json").write_text(json.dumps({"source_sha256":"unit-test-fixture"}),"utf-8")
        self.created=[]
        owner=self
        class TestReader:
            def __init__(self):owner.created.append(self);self.failure=None
            def read(self,question,request,pages,image_dir,max_pixels):
                if self.failure:
                    failure=self.failure;self.failure=None;raise failure
                return {"action":"answer","answer":"UNIT_TEST_ONLY","citations":[pages[0]["page_id"]],
                        "raw_output":"UNIT_TEST_ONLY","answer_source":"test_reader","generation_seconds":0.0}
        self.app=Application(reader_factory=TestReader)
        self.addCleanup(patch.stopall)
        patch.object(self.app,"create_run",return_value=self.output).start()
        patch("stage_a_app.issue",return_value="TEST-ISSUE").start()
        patch("stage_a_app.operation").start()
        def isolated_txt(folder,row):
            with (folder/"逐题结果.txt").open("a",encoding="utf-8") as stream:stream.write(format_question(row))
        patch("stage_a_app.append_question",side_effect=isolated_txt).start()
        self.session=ChatSession(self.app)

    def rows(self):
        return [json.loads(line) for line in (self.output/"results.jsonl").read_text("utf-8").splitlines()]

    def test_multiple_turns_append_all_required_fields_and_load_once(self):
        for question in ("你好","B07124WN69 的宽度","它的重量呢","退出"):self.session.respond(question)
        rows=self.rows()
        self.assertEqual(len(rows),4)
        self.assertEqual(len(self.created),1)
        self.assertEqual(rows[2]["request"]["skus"],["B07124WN69"])
        self.assertEqual(rows[2]["question"],"它的重量呢")
        self.assertIsNone(rows[0]["model"])
        for row in rows:
            for key in ("reference_answer","reference_pages","retrieved_pages","answer","citations","raw_output","retrieval_seconds","generation_seconds","model_revision","config_sha256"):
                self.assertIn(key,row)
        self.assertEqual((self.output/"逐题结果.txt").read_text("utf-8").count("问题 ID："),4)

    def test_actual_application_error_is_preserved_and_followup_continues(self):
        self.session.respond("B07124WN69 的宽度")
        self.app.reader.failure=RuntimeError("controlled unit failure")
        self.assertEqual(self.session.respond("它的重量呢")["action"],"error")
        self.assertEqual(self.session.respond("你好")["action"],"greeting")
        self.assertEqual(self.session.respond("它的重量呢")["action"],"answer")
        rows=self.rows()
        self.assertEqual(rows[1]["status"],"error")
        self.assertEqual(rows[1]["issue_id"],"TEST-ISSUE")
        self.assertIn("controlled unit failure",rows[1]["error"])
        self.assertEqual(len(self.created),1)

    def test_interrupted_model_turn_is_recorded_before_exit(self):
        self.session.respond("B07124WN69 的宽度")
        self.app.reader.failure=KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.session.respond("它的重量呢")
        self.assertTrue(self.session.ended)
        self.assertEqual(self.rows()[-1]["status"],"cancelled")


if __name__=="__main__":unittest.main()

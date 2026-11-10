from copy import deepcopy
import unittest
import test_dialogue_v2 as fixtures
from test_dialogue_v2 import route,A,B
from dialogue_model import LocalDialogueModel
from dialogue_open import general_messages,product_guard,validate_general_reply,significant_lane,quoted_glosses

class OpenCases(unittest.TestCase):
    setUp=fixtures.DialogueTests.setUp

    def general(self):
        self.model.classify_scope=lambda question,context,trace:'general'

    def test_general_question_needs_no_sku_or_retrieval(self):
        self.general();r=self.s.respond('为什么天空是蓝的？')
        self.assertEqual(r['action'],'chat');self.assertEqual(r['conversation_lane'],'general')
        self.assertEqual(r['answer_source'],'local_model_general_knowledge');self.assertFalse(r['retrieved_pages'])

    def test_general_followup_receives_previous_general_answer(self):
        self.general();received=[]
        def reply(q,h,t,clarify_only=False):received.append(general_messages(deepcopy(h)));return 'RAG先检索资料再生成答案。'
        self.model.open_converse=reply
        self.s.respond('解释RAG');self.s.respond('我没懂，举个例子')
        self.assertIn('RAG先检索',received[-1][-1]['content'])

    def test_selected_product_does_not_capture_clear_general_topic(self):
        self.s.respond(A);self.general();r=self.s.respond('二分查找有什么优势')
        self.assertEqual(r['conversation_lane'],'general');self.assertEqual(r['current_skus'],[A])
        self.assertFalse(r['retrieved_pages'])

    def test_product_answers_are_excluded_from_open_reply_context(self):
        self.s.respond(A);self.s.respond('它的重量呢');self.general();received=[]
        def reply(q,h,t,clarify_only=False):received.extend(general_messages(h));return '向量可以表示方向和大小。'
        self.model.open_converse=reply;self.s.respond('什么是向量')
        self.assertFalse(received)

    def test_general_answers_never_become_reader_input_facts(self):
        self.general();self.s.respond('解释RAG');self.s.respond(A)
        self.model.next=route('product','weight','current');self.s.respond('它的重量呢')
        reads=[c for c in self.model.calls if c[0]=='read']
        self.assertEqual(reads[-1][1],'它的重量呢');self.assertNotIn('RAG',reads[-1][1])

    def test_missing_product_fact_never_falls_back_to_free_conversation(self):
        def forbidden(*args,**kwargs):raise AssertionError('Missing evidence must not enter open reply')
        self.model.open_converse=forbidden
        self.assertEqual(self.s.respond(A+'电池多少毫安时')['action'],'not_found')
        self.assertEqual(self.s.respond('你就凭常识编一个这款电池参数')['action'],'not_found')

    def test_product_restatement_retrieves_again_without_previous_assistant_facts(self):
        self.s.respond(A);self.s.respond('它的重量呢')
        self.model.classify_scope=lambda question,context,trace:'product_followup'
        self.s.respond('重新解释刚才的参数')
        reads=[c for c in self.model.calls if c[0]=='read']
        self.assertEqual(len(reads),2);self.assertIn('它的重量呢',reads[-1][1]);self.assertNotIn('33 pounds',reads[-1][1])

    def test_unclear_request_without_product_does_not_demand_product_id(self):
        self.model.classify_scope=lambda q,c,t:'unclear'
        r=self.s.respond('帮我处理这个')
        self.assertEqual(r['action'],'clarify');self.assertNotIn('编号',r['answer'])

    def test_reset_and_sessions_isolate_open_history(self):
        self.general();self.s.respond('解释蒸发');other=self.app.session()
        self.assertEqual(general_messages(other.history),[])
        self.s.respond('重置');self.assertEqual(general_messages(self.s.history),[])

    def test_missing_previous_context_is_not_fabricated_after_reset(self):
        self.general();self.s.respond('解释蒸发');self.s.respond('重置')
        def forbidden(*args,**kwargs):raise AssertionError('Absent history must not be invented')
        self.model.open_converse=forbidden
        r=self.s.respond('把刚才的解释再说一次')
        self.assertEqual(r['action'],'clarify');self.assertIn('没有可引用',r['answer'])
        self.assertFalse(r['retrieved_pages'])

    def test_partial_generation_is_reported_honestly(self):
        self.general()
        def reply(q,h,t,clarify_only=False):t.append({'stage':'general_reply','raw_output':'说明','model_executed':False,'stop_reason':'token_or_time_budget'});return '说明'
        self.model.open_converse=reply;r=self.s.respond('解释一个复杂问题')
        self.assertEqual(r['status'],'partial');self.assertIn('预算',r['answer'])

class OpenContracts(unittest.TestCase):
    def test_unit_conversion_and_user_requested_json_are_allowed(self):
        self.assertEqual(validate_general_reply('1米等于100厘米。'),'1米等于100厘米。')
        self.assertEqual(validate_general_reply('{"n":1}'),'{"n":1}')

    def test_general_reply_rejects_specific_facts_and_fake_citations(self):
        for reply in [A+'宽25英寸','证据【第2页】','已替你提交退款','这把椅子的重量为33磅']:
            with self.assertRaises(ValueError):validate_general_reply(reply)

    def test_factual_guard_does_not_match_general_inventory_concept(self):
        self.assertIsNone(product_guard('解释库存这个概念',[A],[],[]))
        self.assertEqual(product_guard('它的电池是多少',[A],[],[]),'product')
        self.assertEqual(product_guard('它是什么材料',[A,B],[],[]),'product')
        self.assertEqual(product_guard('这个店铺现在怎么退货',[],[],[]),'product')
        self.assertEqual(product_guard('帮我看看现在有没有货',[],[],[]),'business')
        self.assertEqual(product_guard('换一款',[A],[],[]),'product')

    def test_scope_unknown_enum_is_not_silently_accepted(self):
        m=LocalDialogueModel();m.generate=lambda *args:'whatever'
        with self.assertRaises(ValueError):m.classify_scope('问题',{},[])

    def test_control_thanks_does_not_erase_significant_general_topic(self):
        h=[{'lane':'general','action':'chat'},{'lane':'control','action':'chat'}]
        self.assertEqual(significant_lane(h),'general')

    def test_ellipsis_is_bound_to_most_recent_lane_not_older_topic(self):
        h=[{'lane':'general','action':'chat'},{'lane':'product','action':'answer'}]
        self.assertEqual(product_guard('我刚才没懂，换个说法',[A],h,[]),'product_followup')
        self.assertEqual(product_guard('给我讲讲什么是向量',[A],h,[]),'general')

    def test_glosses_cannot_add_unquoted_cleaning_methods(self):
        text=quoted_glosses([{'quote':'wipe with a damp cloth'}])
        self.assertIn('湿布',text);self.assertNotIn('干布',text)

if __name__=='__main__':unittest.main()

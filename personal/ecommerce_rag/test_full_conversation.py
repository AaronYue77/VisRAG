"""Routing regressions using spies; real model verification lives separately."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from full_catalog import ROOT
from full_app import FullApplication
from full_conversation import clear_route,bound_history,direct_business_request,FullConversation
from common import WORKSPACE,read_jsonl


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp');self.addCleanup(self.folder.cleanup)
        self.app=FullApplication('B',report_root=Path(self.folder.name),audit=False);self.addCleanup(self.app.close)
        self.s=self.app.session();self.app.conversation=Mock()
        self.app.conversation.reply.side_effect=lambda text,history,trace:'普通交流：'+text
        self.app.conversation.route.return_value={'intent':'general','reason':'model_spy'}
    def row(self):return read_jsonl(self.s.out/'results.jsonl')[-1]
    def test_user_report_identity_and_capability_never_search(self):
        with patch.object(self.app.retriever,'search',side_effect=AssertionError('General must not search')):
            for text in ['你好','你是谁','你会什么','你有什么功能？']:
                r=self.s.respond(text);self.assertEqual(r['action'],'chat');self.assertEqual(r['retrieved_pages'],[])
                self.assertEqual(r['citations'],[]);self.assertEqual(self.row()['conversation_lane'],'general')
    def test_open_topics_not_restricted_to_a_question_whitelist(self):
        with patch.object(self.app.retriever,'search',side_effect=AssertionError('General must not search')):
            for text in ['为什么海水是蓝色的？','心情不太好，陪我聊两句','我想讨论一个新故事设定']:
                self.assertEqual(self.s.respond(text)['action'],'chat')
        self.assertEqual(self.app.conversation.route.call_count,3)
    def test_selected_product_does_not_hijack_general_followup(self):
        self.s.respond('B07C5PRQHN@amazon.ca')
        self.assertTrue(self.s.selected)
        for text in ['什么是RAG？','再举个例子','帮我写Python二分查找','你是谁']:
            self.assertEqual(self.s.respond(text)['action'],'chat')
            self.assertEqual(self.s.selected,['B07C5PRQHN@amazon.ca'])
    def test_general_context_bounded_and_reset(self):
        for i in range(10):self.s.respond('你是谁'+('？'*(i+1)))
        self.assertLessEqual(len(self.s.general_history),6)
        self.assertTrue(self.s.general_history)
        self.s.respond('重置');self.assertFalse(self.s.general_history);self.assertIsNone(self.s.last_lane)
    def test_business_definitions_do_not_trigger_tools(self):
        for text in ['什么是库存？','帮我写订单管理代码','解释退款的定义']:
            self.assertEqual(self.s.respond(text)['action'],'chat')
        self.assertEqual(self.s.respond('帮我办理退款')['action'],'requires_tool')
        self.assertEqual(self.s.respond('今天还有库存吗')['action'],'requires_tool')
    def test_product_question_missing_context_does_not_chat(self):
        self.assertEqual(self.s.respond('这款有多重？')['action'],'clarify')
        self.app.conversation.reply.assert_not_called()
    def test_search_and_selection_continue_to_work(self):
        self.assertEqual(self.s.respond('搜索 headphones')['action'],'search_results')
        self.assertEqual(self.s.respond('第一个')['action'],'selected')
        self.assertTrue(self.s.selected)
    def test_specific_product_fact_never_uses_chat(self):
        self.assertEqual(self.s.respond('B000000000 的品牌是什么')['action'],'not_found')
        self.app.conversation.reply.assert_not_called()
    def test_general_history_does_not_contain_product_answers(self):
        self.s.respond('你好');before=list(self.s.general_history)
        self.s.respond('B07C5PRQHN@amazon.ca')
        self.assertEqual(self.s.general_history,before)
    def test_rules_respect_general_context_and_specific_reference(self):
        self.assertEqual(clear_route('什么是品牌',True,'product')['intent'],'general')
        self.assertEqual(clear_route('这款品牌是什么',True,'general')['intent'],'fact')
        self.assertEqual(clear_route('品牌是什么',True,'general')['intent'],'fact')
        self.assertIsNone(clear_route('我刚想起一件事',False,None))
        self.assertEqual(clear_route('再举个例子',True,'general')['intent'],'general')
    def test_commerce_topic_words_alone_do_not_force_tools(self):
        for text in ['为什么电商需要库存管理？','如何设计物流跟踪系统？','退款流程有哪些常见问题？']:
            self.assertFalse(direct_business_request(text))
            self.assertEqual(self.s.respond(text)['action'],'chat')
        for text in ['查询今天库存','帮我办理退款','当前店铺退货政策是什么']:
            self.assertTrue(direct_business_request(text))
    def test_public_product_context_does_not_force_meta_question_into_rag(self):
        with patch.object(self.app.retriever,'search',side_effect=AssertionError('No search for identity')):
            self.assertEqual(self.s.respond('你是谁',{'skus':['B07C5PRQHN'],'domain':'amazon.ca'})['action'],'chat')
    def test_partial_open_reply_is_retained_and_can_continue(self):
        conversation=FullConversation(Mock())
        def partial(messages,stage,trace,budget):
            trace.append({'stage':stage,'status':'partial'});return '第一段内容'
        conversation.generate=Mock(side_effect=partial)
        self.assertIn('可以让我继续',conversation.reply('写一篇长文',[],[]))
    def test_non_product_pronouns_and_search_words_are_not_product_commands(self):
        questions=['这件事让我很纠结，朋友老迟到怎么办？','这把游戏又输了，怎么复盘？',
                   '这款游戏怎么通关？','搜索算法和排序算法有什么区别？','找一个周末放松的方法']
        for text in questions:
            route=clear_route(text,True,'product')
            self.assertTrue(route is None or route['intent']=='general',text)
            self.assertEqual(self.s.respond(text)['action'],'chat')
    def test_unlisted_topics_work_with_product_selected_or_public_context(self):
        self.s.respond('B07C5PRQHN@amazon.ca')
        with (patch.object(self.app.retriever,'search',side_effect=AssertionError('No product search')),
              patch.object(self.app.retriever,'get',side_effect=AssertionError('No product lookup'))):
            for q in ['一个国家为什么可能有多个时区？','我想练习和陌生人交流','如何安排小说中的冲突？']:
                value=self.s.respond(q,{'skus':['B07C5PRQHN'],'domain':'amazon.ca'})
                self.assertEqual(value['action'],'chat');self.assertEqual(value['citations'],[])
    def test_product_id_as_programming_data_does_not_force_rag(self):
        value=self.s.respond('写一个Python函数，从字符串提取示例编号B07C5PRQHN')
        self.assertEqual(value['action'],'chat');self.assertEqual(value['retrieved_pages'],[])
    def test_router_invalid_json_preserves_open_conversation_default(self):
        conversation=FullConversation(Mock());conversation.generate=Mock(return_value='unparseable')
        result=conversation.route('聊聊怎样保持好奇心',True,'product',[],[])
        self.assertEqual(result['intent'],'general')
    def test_attribute_and_business_concepts_are_still_open_topics(self):
        self.s.respond('B07C5PRQHN@amazon.ca')
        for q in ['品牌营销有哪些方法？','颜色心理学是什么意思？','当前库存管理有哪些基本原理？','查询物流系统设计的资料应该从哪入手？']:
            route=clear_route(q,True,'product')
            self.assertTrue(route is None or route['intent']=='general',q)
            self.assertFalse(direct_business_request(q),q)
            self.assertEqual(self.s.respond(q)['action'],'chat')


if __name__=='__main__':unittest.main()

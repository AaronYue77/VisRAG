import json
import unittest
import test_dialogue_v2 as fixtures
from test_dialogue_v2 import A,B,route
from dialogue_open import product_guard
from dialogue_catalog import catalogue_request
from dialogue_model import LocalDialogueModel

class CatalogueTests(unittest.TestCase):
    setUp=fixtures.DialogueTests.setUp

    def test_listing_is_exact_metadata_with_no_reader_and_clears_pending(self):
        self.s.state['pending']={'question':'多宽','topic':'width','missing':'product'}
        r=self.s.respond('请你给我你知道的商品编码和对应的名称')
        self.assertEqual(r['action'],'catalog');self.assertEqual(len(r['catalog_items']),24)
        self.assertEqual(r['catalog_items'],self.app.catalogue.items());self.assertIsNone(self.s.state['pending'])
        self.assertFalse(any(c[0]=='read' for c in self.model.calls));self.assertFalse(r['citations'])
        raw=json.loads((self.s.output/'results.jsonl').read_text('utf-8').strip())
        self.assertEqual(raw['answer_source'],'current_catalog_metadata');self.assertTrue(raw['pending_transitions'])

    def test_novel_semantic_catalogue_paraphrase(self):
        self.model.classify_scope=lambda *args:'catalog'
        r=self.s.respond('让我先挑挑你这边已收录的东西')
        self.assertEqual(r['action'],'catalog');self.assertEqual(len(r['catalog_items']),24)

    def test_catalogue_after_unknown_category_is_current_request(self):
        self.model.classify_scope=lambda *args:'coverage'
        r=self.s.respond('拖鞋');self.assertEqual(r['action'],'coverage');self.assertIn('不能',r['answer'])
        r=self.s.respond('列出商品名称和编码');self.assertEqual(r['action'],'catalog')
        self.assertNotIn('拖鞋',r['answer'])

    def test_coverage_is_corpus_derived_without_search_promise(self):
        self.model.classify_scope=lambda *args:'coverage'
        r=self.s.respond('还有其他类别吗')
        self.assertIn('24',r['answer']);self.assertIn('椅子',r['answer']);self.assertIn('没有其他品类',r['answer'])
        self.assertEqual(self.app.catalogue.manifest['product_types'],['CHAIR'])

    def test_code_word_not_invention_verb(self):
        history=[{'lane':'product','action':'clarify'}]
        for q in ('我不知道编号，你告诉我你有哪些商品的信息？','解释商品编码的含义','请编写商品查询函数'):
            self.assertNotEqual(product_guard(q,[],history,[]),'product_followup')
        self.assertEqual(product_guard('编一个电池参数',[A],history,[]),'product_followup')
        self.assertIn(product_guard('给这款编一个电池参数',[A],history,[]),{'product','product_followup'})

    def test_catalogue_guard_does_not_capture_code_concept_or_facts(self):
        self.assertIsNone(catalogue_request('解释什么是商品编号'))
        self.assertEqual(product_guard('解释什么是商品编号',[],[],[]),'general')
        self.assertEqual(catalogue_request('列出商品和它们的全部宽度'),'mixed_catalog')

    def test_catalogue_order_wins_over_old_two_product_comparison(self):
        self.s.respond(A+' '+B)
        r=self.s.respond('商品列表');wanted=r['catalog_items'][1]['sku']
        r=self.s.respond('就选第二个')
        self.assertEqual(r['action'],'selected');self.assertEqual(r['current_skus'],[wanted])

    def test_new_general_question_retires_pending_but_greeting_preserves_it(self):
        self.s.state['pending']={'question':'多宽','topic':'width','missing':'product'}
        self.model.classify_scope=lambda *args:'general'
        self.s.respond('在吗');self.assertIsNotNone(self.s.state['pending'])
        self.model.pending_relation=lambda *args:'replace'
        self.s.respond('解释哈希表');self.assertIsNone(self.s.state['pending'])
        self.s.respond(A);self.assertFalse(any(c[0]=='read' for c in self.model.calls))

    def test_new_fact_query_does_not_resume_old_field(self):
        self.model.next=route('product','width','none')
        self.s.respond('宽度是多少')
        self.model.next=route('product','weight','explicit')
        self.s.respond(A+'重量多少')
        read=[c for c in self.model.calls if c[0]=='read'][-1]
        self.assertEqual(read[1],A+'重量多少');self.assertIsNone(self.s.state['pending'])

    def test_old_pending_expires_after_bounded_turns(self):
        self.s.state['pending']={'question':'多宽','topic':'width','missing':'product','created_turn':1}
        self.s.turn=9;self.model.classify_scope=lambda *args:'general'
        self.s.respond('在吗');self.assertIsNone(self.s.state['pending'])

    def test_general_reply_sees_actual_manifest(self):
        model=LocalDialogueModel();model.catalog_manifest=self.app.catalogue.manifest;captured=[]
        def fake(messages,*args):captured.extend(messages);return '当前只有已收录的椅子资料；可聊其他一般知识。'
        model.generate=fake;model.open_converse('能做什么',[],[])
        self.assertIn('product_count',captured[0]['content']);self.assertIn('不能邀请',captured[0]['content'])

    def test_scope_and_pending_invalid_outputs_remain_conservative(self):
        model=LocalDialogueModel();model.generate=lambda *args:'nonsense'
        with self.assertRaises(ValueError):model.pending_relation('问题',{},[])

    def test_catalogue_does_not_ignore_requested_store_or_date(self):
        for req in ({'store_id':'other-shop'},{'as_of':'2025-01-01'}):
            r=self.s.respond('列出商品名称和编号',req)
            self.assertEqual(r['action'],'not_found');self.assertFalse(r['catalog_items'])

    def test_invention_with_known_product_is_evidence_checked_even_if_unsupported(self):
        self.s.respond(A);self.model.next=route('unsupported','unknown','current')
        r=self.s.respond('凭空编造这款商品的电池参数')
        self.assertEqual(r['action'],'not_found');self.assertFalse(r['citations'])

    def test_catalogue_words_cannot_swallow_live_inventory_or_price(self):
        for q in ('列出商品编号以及实时库存','给我商品名称和最新价格清单'):
            self.assertIsNone(catalogue_request(q))

    def test_current_field_request_cannot_enter_general_or_coverage(self):
        self.s.respond(A)
        for domain,q,topic in [('general','从宽度看放得下吗？','width'),('coverage','资料有没有说适用年龄？','age')]:
            self.model.classify_scope=lambda *args:domain
            self.model.next=route('product',topic,'current')
            r=self.s.respond(q)
            self.assertEqual(r['action'],'answer');self.assertTrue(r['citations'])

    def test_general_reply_cannot_report_a_product_measurement_as_fact(self):
        from dialogue_open import validate_general_reply
        with self.assertRaises(ValueError):validate_general_reply('椅子宽度约40厘米，可轻松放入门洞。')
        self.assertIn('100',validate_general_reply('1米等于100厘米。'))
        self.assertIn('假设',validate_general_reply('假设一把椅子的宽度是40厘米，这是计算示例。'))

    def test_fabrication_cannot_mutate_needs_even_if_model_wants_requirements(self):
        self.s.respond(B);before=dict(self.s.state['needs'])
        self.model.next=route('requirements','unknown','current',focus='猜电池')
        r=self.s.respond('没有资料也请猜出这款的电池容量')
        self.assertEqual(r['action'],'not_found');self.assertEqual(self.s.state['needs'],before)
        self.assertIn('不能编造',r['answer']);self.assertFalse(r['citations'])

    def test_explicit_temporary_stop_retires_before_scope_model_even_if_it_says_aside(self):
        self.s.state['pending']={'question':'多宽','topic':'width','missing':'product'}
        def scope(q,context,trace):
            self.assertIsNone(context['pending']);return 'general'
        self.model.classify_scope=scope;self.model.pending_relation=lambda *args:'aside'
        self.s.respond('暂时不查这个了，讲讲列表和字典的区别')
        self.assertIsNone(self.s.state['pending'])

if __name__=='__main__':unittest.main()

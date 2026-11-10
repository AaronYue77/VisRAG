"""Catalog, constrained public search, and latest-task continuations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from full_app import FullApplication
from full_catalogue import catalogue_request,normalize_search
from common import WORKSPACE,read_jsonl


class CatalogFollowupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp');self.addCleanup(self.tmp.cleanup)
        self.app=FullApplication('B',report_root=Path(self.tmp.name),audit=False);self.addCleanup(self.app.close)
        self.s=self.app.session();self.app.conversation=Mock()
        self.app.conversation.route.return_value={'intent':'general','reason':'test_spy'}
        self.app.conversation.reply.return_value='普通话题的接续回答'
    def row(self):return read_jsonl(self.s.out/'results.jsonl')[-1]
    def select(self):self.s.respond('B005FIG90C@amazon.co.jp')
    def test_catalog_questions_use_real_categories(self):
        for q in ['你包含哪些商品？','你们知识库都有哪些品类？','这里可以查哪些商品？','除了鞋子还收录哪些类目？']:
            v=self.s.respond(q);self.assertEqual(v['action'],'catalog');self.assertIn('147,702',v['answer'])
            self.assertIn('SHOES',v['answer']);self.assertIn('12,965',v['answer'])
        self.app.conversation.reply.assert_not_called()
    def test_catalog_continuation_reads_next_categories(self):
        self.select();self.s.respond('你包含哪些商品？');first=self.row()['catalog_categories']
        v=self.s.respond('还有呢');second=self.row()['catalog_categories']
        self.assertEqual(v['action'],'catalog')
        self.assertFalse({r['product_type'] for r in first}&{r['product_type'] for r in second})
    def test_shoe_search_normalizes_task_words_and_has_hard_types(self):
        plan=normalize_search('我想查询鞋子类的商品',self.app.retriever.manifest)
        self.assertEqual(plan['query'],'shoes');self.assertNotIn('商品',plan['query'])
        v=self.s.respond('我想查询鞋子类的商品')
        self.assertEqual(v['action'],'search_results')
        self.assertTrue(all(r['product_type'] in {'SHOES','BOOT','SANDAL','TECHNICAL_SPORT_SHOE'} for r in v['candidates']))
    def test_category_family_filters_are_parameterized(self):
        v=self.app.retriever.search('shoes',filters={'product_types':['SHOES','BOOT'],'domain':"x' OR 1=1 --"})
        self.assertEqual(v['items'],[])
        with self.assertRaises(ValueError):self.app.retriever.search('shoes',filters={'product_types':["SHOES' OR 1=1 --"]})
    def test_overview_more_is_source_grounded_and_exhausts(self):
        self.select();self.s.respond('给我这个商品的相关信息');first=self.row()
        self.assertEqual(first['action'],'answer');self.assertIn('B005FIG90C',first['answer']);self.assertNotIn('B005F1G90C',first['answer'])
        self.s.respond('还有呢');second=self.row()
        self.assertEqual(second['action'],'answer');self.assertTrue(second['claims'])
        self.assertFalse({r['item_id'] for r in first['claims']}&{r['item_id'] for r in second['claims']})
        self.assertTrue(any(r['source_field']=='bullet_point' for r in second['claims']))
        self.s.respond('我说的还有什么相关信息');third=self.row()
        self.assertEqual(third['action'],'answer');self.assertTrue(third['retrieval_trace']['overview_exhausted'])
        self.assertIn('展示完',third['answer']);self.app.conversation.reply.assert_not_called()
        country=next(c for c in second['claims'] if c['source_field']=='country')
        self.assertTrue(country['citation'].endswith(':metadata'))
    def test_general_chat_does_not_capture_subsequent_product_more(self):
        self.s.respond('你会什么');self.select();self.s.respond('给我这个商品的相关信息')
        v=self.s.respond('还有呢');self.assertEqual(v['action'],'answer');self.assertEqual(self.s.last_task['kind'],'overview')
        self.assertNotIn('写代码',v['answer'])
    def test_explicit_topic_switch_restores_general_continuation(self):
        self.select();self.s.respond('给我这个商品的相关信息')
        self.assertEqual(self.s.respond('换个话题，帮我写一首诗')['action'],'chat')
        self.assertEqual(self.s.respond('还有呢')['action'],'chat')
        self.assertEqual(self.s.last_task['lane'],'general')
    def test_catalog_and_generic_concepts_are_separate(self):
        self.assertFalse(catalogue_request('什么是商品分类？'))
        self.assertFalse(catalogue_request('这个商品包含哪些配件？'))
        self.assertEqual(self.s.respond('给我这个商品的相关信息')['action'],'clarify')


if __name__=='__main__':unittest.main()

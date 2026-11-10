"""Full-catalog identities, global retrieval, leakage and app boundary regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from full_catalog import ROOT,connect
from full_protocol import EVAL
from full_retrieval import FullRetriever,exact_skus
from full_app import FullApplication
from common import WORKSPACE,read_jsonl,file_hash


class FullCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db=connect();cls.retriever=FullRetriever()
        cls.addClassCleanup(cls.db.close);cls.addClassCleanup(cls.retriever.close)
    def test_all_official_rows_are_in_actual_search_index(self):
        self.assertEqual(self.db.execute('SELECT count(*) FROM products').fetchone()[0],147702)
        self.assertEqual(self.db.execute('SELECT count(*) FROM products_fts').fetchone()[0],147702)
        self.assertEqual(self.db.execute('SELECT count(DISTINCT product_type) FROM products').fetchone()[0],576)
        self.assertEqual(self.db.execute('SELECT count(*) FROM images').fetchone()[0],398212)
    def test_marketplace_identity_is_not_deduplicated_as_sku(self):
        sku=self.db.execute('SELECT sku FROM products GROUP BY sku HAVING count(*)>1 LIMIT1'.replace('LIMIT1','LIMIT 1')).fetchone()[0]
        rows=self.retriever.get(sku)
        self.assertGreater(len(rows),1)
        self.assertEqual(len(rows),len({r['listing_key'] for r in rows}))
    def test_sku_detection_does_not_treat_normal_words_as_ids(self):
        self.assertEqual(exact_skus('headphones background strawberry'),[])
        self.assertEqual(exact_skus('选择 b072v8ds1w'),['B072V8DS1W'])
    def test_global_search_finds_non_chair_classes(self):
        for word,kind in [('headphones','HEADPHONES'),('backpack','BACKPACK'),('coffee','COFFEE')]:
            result=self.retriever.search(word,'bm25',5,{'product_type':kind})
            self.assertTrue(result['items'])
            self.assertTrue(all(r['product_type']==kind for r in result['items']))
            self.assertEqual(result['trace']['global_index_records'],147702)
    def test_filters_are_parameterized(self):
        found=self.retriever.search('backpack','bm25',5,{'domain':"amazon.com' OR 1=1 --"})
        self.assertEqual(found['items'],[])
        self.assertEqual(self.db.execute('SELECT count(*) FROM products').fetchone()[0],147702)
    def test_empty_stop_query_and_invalid_options(self):
        self.assertEqual(self.retriever.search('the and or','bm25')['items'],[])
        with self.assertRaises(ValueError):self.retriever.search('chair','bm25',0)
        with self.assertRaises(ValueError):self.retriever.search('chair','bm25',filters={'gold_answer':'x'})
    def test_eval_has_no_related_family_split_overlap(self):
        chosen=json.loads((EVAL/'selected.json').read_text('utf-8'))
        self.assertEqual(len(chosen),80)
        self.assertEqual(len({r['product_type'] for r in chosen}),20)
        self.assertFalse({r['family_key'] for r in chosen if r['split']=='dev'}&{r['family_key'] for r in chosen if r['split']=='test'})
    def test_eval_freeze_and_global_query_scope(self):
        protocol=json.loads((EVAL/'protocol.json').read_text('utf-8'))
        self.assertEqual(file_hash(EVAL/'questions.jsonl'),protocol['questions_sha256'])
        for q in read_jsonl(EVAL/'retrieval-questions.jsonl'):
            self.assertFalse(q['filters'])
            self.assertNotIn(q['reference_listing_key'].split('@')[0],q['question'])
    def test_missing_fields_are_not_filtered_out_of_catalog(self):
        missing=self.db.execute("SELECT count(*) FROM products WHERE json_extract(raw_json,'$.material') IS NULL").fetchone()[0]
        self.assertGreater(missing,90000)
    def test_controls_and_field_missing_do_not_load_model(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp') as folder:
            app=FullApplication('A',report_root=Path(folder),audit=False);self.addCleanup(app.close);s=app.session()
            self.assertEqual(s.respond('商品列表')['action'],'catalog')
            self.assertEqual(s.respond('帮我办理退款')['action'],'requires_tool')
            self.assertEqual(s.respond('这款颜色是什么')['action'],'clarify')
            self.assertEqual(s.respond('B000000000 的颜色是什么')['action'],'not_found')
            self.assertEqual(s.respond('B072V8DS1W 的 manufacturing_timestamp_utc 字段值是什么')['action'],'not_found')
            self.assertIsNone(app.reader.reader)
            self.assertEqual(s.respond('重置')['current_listing_keys'],[])
    def test_unknown_request_fields_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp') as folder:
            app=FullApplication('A',report_root=Path(folder),audit=False);self.addCleanup(app.close)
            with self.assertRaises(ValueError):app.respond('hi',{'reference_answer':'gold'})
            with self.assertRaises(ValueError):app.respond('hi',session_id='../other')
    def test_search_clears_previous_product_context(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp') as folder:
            app=FullApplication('A',report_root=Path(folder),audit=False);self.addCleanup(app.close);s=app.session()
            self.assertEqual(s.respond('选择 B072V8DS1W')['action'],'selected')
            self.assertTrue(s.selected)
            self.assertEqual(s.respond('搜索 coffee')['action'],'search_results')
            self.assertFalse(s.selected)
    def test_composite_identity_can_be_selected_without_model(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'cache/tmp') as folder:
            app=FullApplication('A',report_root=Path(folder),audit=False);self.addCleanup(app.close);s=app.session()
            value=s.respond('选择 B01HTRD79M@primenow.amazon.co.uk')
            self.assertEqual(value['action'],'selected')
            self.assertEqual(value['current_listing_keys'],['B01HTRD79M@primenow.amazon.co.uk'])
            self.assertIsNone(app.reader.reader)
    def test_card_page_contract_contains_fixed_reader_version(self):
        from full_cards import ensure_card,choose_pages
        selected=json.loads((EVAL/'selected.json').read_text('utf-8'))[0]
        row=dict(self.db.execute('SELECT * FROM products WHERE id=?',(selected['id'],)).fetchone())
        pages=choose_pages(ensure_card(row),'brand 品牌')
        self.assertTrue(pages)
        self.assertTrue(all({'page_id','doc_id','sku','pdf','page','version'}<=p.keys() for p in pages))


if __name__=='__main__':unittest.main()

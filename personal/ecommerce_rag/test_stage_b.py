import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from b_protocol import ROOT,CORPUS,rewrite,query_key,MODES
from b_retrieval import BRetriever
from retrieval import rrf
from common import read_jsonl,file_hash


class StageBTests(unittest.TestCase):
    def test_protocol_cannot_modify_original_A(self):
        protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))
        self.assertEqual(file_hash(ROOT.parent/'abo-cards-v1/questions.jsonl'),protocol['original_A_questions_sha256'])
        self.assertEqual(file_hash(ROOT/'questions.jsonl'),protocol['questions_sha256'])

    def test_public_queries_have_no_gold_or_split(self):
        tasks=read_jsonl(ROOT/'queries.public.jsonl')
        self.assertTrue(all(set(t)=={'text','query_key'} and query_key(t['text'])==t['query_key'] for t in tasks))

    def test_splits_do_not_overlap_products(self):
        docs=json.loads((CORPUS/'documents.json').read_text('utf-8'))
        self.assertFalse({d['family_key'] for d in docs if d['split']=='dev'}&{d['family_key'] for d in docs if d['split']=='test'})

    def test_rewrite_is_bounded_and_only_current_text(self):
        self.assertIn('width',rewrite('多宽'))
        self.assertEqual(rewrite('电池容量'), '电池容量')
        self.assertNotIn('25',rewrite('多宽'))
        self.assertLessEqual(len(rewrite('宽度'*1000)),1200)

    def test_fusion_uses_rank_and_ignores_duplicate_votes(self):
        order,scores=rrf([['x','x','y'],['y','x']])
        self.assertEqual(len(order),2)
        self.assertAlmostEqual(scores['x'],1/61+1/62)

    def test_multi_product_reservation_respects_fixed_image_budget(self):
        pages={'a':{'sku':'one'},'b':{'sku':'one'},'c':{'sku':'two'}}
        self.assertEqual(BRetriever.reserve(['a','b','c'],pages,['one','two'],2),['a','c'])

    def test_all_modes_preserve_boundaries_without_calling_encoder(self):
        r=BRetriever()
        for mode in MODES:
            result=r.search('查订单物流',{},mode)
            self.assertEqual(result['plan']['action'],'requires_tool')
            self.assertEqual(result['pages'],[])
        self.assertIsNone(r.encoder)
        self.assertIsNone(r.reranker)

    def test_actual_candidates_never_mix_wrong_product(self):
        folder=ROOT.parent/'b-index-v1/retrieval-plans'
        for mode in MODES:
            path=folder/(mode+'.jsonl')
            if not path.exists():continue
            for r in read_jsonl(path):
                if r['plan']['action']=='retrieve':
                    self.assertTrue(all(p['sku'] in r['plan']['skus'] for p in r['pages']))
                    self.assertEqual(len({p['page_id'] for p in r['pages']}),len(r['pages']))
                    self.assertLessEqual(len(r['pages']),2)

    def test_unchanged_retry_counts_both_retrievals_without_second_read(self):
        from b_app import BSession
        session=BSession.__new__(BSession)
        calls=[]
        def first_read(self,row,*args):
            calls.append(True)
            row.update(action='not_found',answer='missing',citations=[],retrieved_pages=[{'page_id':'same'}],retrieval_trace={},retrieval_seconds=1.0)
        retriever=SimpleNamespace(search=lambda *args:([{'page_id':'same'}],{},2.0))
        session.app=SimpleNamespace(b_retry=True,retriever=retriever)
        session.state={'needs':{}}
        row={}
        with patch('b_app.DialogueSession._answer_product',first_read):
            session._answer_product(row,'宽度是多少',['one'],'width')
        self.assertEqual(len(calls),1)
        self.assertEqual(row['retrieval_seconds'],3.0)
        self.assertEqual(row['b_retry']['retrieval_attempts'],2)
        self.assertEqual(row['b_retry']['stop_reason'],'unchanged_candidates_no_second_read')


if __name__=='__main__':unittest.main()

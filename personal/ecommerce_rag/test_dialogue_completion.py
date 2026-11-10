"""Regress operation/need independence and evidence-scoped source conflicts."""
import json
from copy import deepcopy
import unittest
import test_dialogue_v2 as fixtures
from test_dialogue_v2 import route, A, B, C
from dialogue_model import validate_facets,is_short_thanks
from dialogue_evidence import current_page_conflicts,field_excerpt,conflict_notice


class CompletionCases(unittest.TestCase):
    setUp=fixtures.DialogueTests.setUp
    def facets(self,task,need='none'):
        # These unit cases isolate the product sub-router, after scope selection.
        self.model.classify_scope=lambda question,context,trace:'product'
        self.model.analyze_turn=lambda question,context,trace:{'task':task,'need':need}

    def test_semantic_execution_cannot_fall_into_product_clarification(self):
        self.model.next=route('chat','unknown','none');self.facets('execute')
        r=self.s.respond('请代我提交一份退货申请')
        self.assertEqual(r['action'],'requires_tool');self.assertIn('不能',r['answer'])
        self.assertFalse(any(c[0]=='read' for c in self.model.calls))

    def test_policy_and_execution_are_separate(self):
        self.model.next=route('chat','unknown','none');self.facets('policy')
        r=self.s.respond('想看看这家店的退换期限')
        self.assertEqual(r['action'],'not_found');self.assertFalse(r['citations'])

    def test_compound_selection_cannot_discard_fact_request(self):
        self.model.next=route('select','unknown','explicit');self.facets('fact')
        r=self.s.respond(A+' 跟 '+C+' 各自的宽度分别报一下')
        self.assertEqual(r['action'],'answer');self.assertEqual(r['current_skus'],[A,C])
        self.assertEqual({p['sku'] for p in r['retrieved_pages']},{A,C})

    def test_known_field_preserved_even_if_second_classifier_says_select(self):
        self.model.next=route('select','unknown','explicit');self.facets('select')
        self.assertEqual(self.s.respond(A+'与'+C+'的width')['action'],'answer')

    def test_need_updates_even_when_original_router_says_chat(self):
        self.s.respond(B);self.model.next=route('chat','unknown','none');self.facets('none','focus')
        r=self.s.respond('我目前最想了解的是清洁方法')
        self.assertEqual(r['action'],'requirements');self.assertEqual(self.s.state['active_topic'],'cleaning')
        self.assertIn('清洁',self.s.state['needs']['focus'])

    def test_same_turn_need_and_query_updates_then_reads(self):
        self.s.respond(C);self.model.next=route('product','assembly','current');self.facets('fact','focus')
        r=self.s.respond('我最在意组装，请顺便查说明')
        self.assertEqual(r['action'],'answer');self.assertEqual(self.s.state['active_topic'],'assembly')
        self.assertIn('组装',self.s.state['needs']['focus'])

    def test_invalid_facets_are_recorded_without_claiming_success(self):
        self.model.next=route('chat','unknown','none')
        self.model.classify_scope=lambda question,context,trace:'product'
        def bad(question,context,trace):
            trace.append({'stage':'turn_facets','raw_output':'broken','model_executed':False})
            raise ValueError('controlled invalid facets')
        self.model.analyze_turn=bad
        r=self.s.respond('请帮忙处理这件事')
        self.assertEqual(r['status'],'guarded');self.assertEqual(r['action'],'clarify')
        saved=json.loads((self.s.output/'results.jsonl').read_text('utf-8').split('\n')[0])
        self.assertEqual(saved['model_calls'][-1]['raw_output'],'broken')

    def test_indirect_need_reference_does_not_erase_explicit_focus(self):
        self.s.respond(B);self.s.state['needs']['focus']='适用年龄';self.s.state['active_topic']='age'
        self.model.next=route('chat','unknown','none');self.facets('none','focus')
        self.model.focus_topic=lambda value,trace:'unknown'
        r=self.s.respond('沿用刚才的重点讲')
        self.assertEqual(r['action'],'answer');self.assertEqual(self.s.state['needs']['focus'],'适用年龄')
        self.assertEqual(self.s.state['active_topic'],'age')

    def test_fit_followup_uses_existing_concrete_space_without_replacing_it(self):
        self.s.respond(A);self.s.state['needs']['space']='可用宽度65厘米';self.s.state['active_topic']='width'
        self.model.next=route('chat','unknown','none');self.facets('none','focus')
        self.model.focus_topic=lambda value,trace:'fit'
        self.assertEqual(self.s.respond('能保证适合吗')['action'],'answer')
        self.assertEqual(self.s.state['needs']['focus'],'');self.assertEqual(self.s.state['active_topic'],'width')

    def test_fit_retrieval_uses_active_positive_focus_before_negated_old_word(self):
        _,trace,_=self.app.retriever.search('符合重点吗',[B],'fit',{'focus':'不看清洁了，只看年龄','_active_topic':'age'})
        self.assertEqual(trace['normalized_topic'],'age')

    def test_need_only_second_opinion_does_not_replace_explicit_fact_query(self):
        self.s.respond(B);self.s.state['needs']['focus']='我只关心清洁';self.s.state['active_topic']='cleaning'
        self.model.next=route('product','cleaning','current');self.facets('none','focus')
        self.assertEqual(self.s.respond('那应该怎么打理？')['action'],'answer')
        self.assertEqual(self.s.state['needs']['focus'],'我只关心清洁')

    def test_concrete_space_update_survives_empty_second_opinion(self):
        self.s.respond(A);self.model.next=route('requirements','fit','current',space='改成可用宽度70厘米');self.facets('none')
        self.assertEqual(self.s.respond('改成可用宽度70厘米')['action'],'requirements')
        self.assertIn('70',self.s.state['needs']['space'])

    def test_specific_purpose_is_not_lost_to_generic_focus_classification(self):
        self.model.next=route('requirements','unknown','none',purpose='我想给孩子用');self.facets('none','focus')
        self.s.respond('我想给孩子用')
        self.assertIn('孩子',self.s.state['needs']['purpose']);self.assertEqual(self.s.state['needs']['focus'],'')

    def test_valid_purpose_does_not_require_a_visual_property_enum(self):
        self.model.next=route('requirements','unknown','none',purpose='改成书房办公');self.facets('none','focus')
        def fail(*args):raise AssertionError('purpose should not require an optional property classification')
        self.model.focus_topic=fail
        r=self.s.respond('改成书房办公')
        self.assertEqual(r['action'],'requirements');self.assertIn('办公',self.s.state['needs']['purpose'])

    def pending_confirmation(self):
        self.s.respond(A);self.s.state['needs']['space']='预留宽度66厘米';self.s.state['active_topic']='width'
        self.model.next=route('clarify','unknown','none');self.facets('none')
        r=self.s.respond('请处理一下')
        self.assertEqual(r['action'],'clarify');self.assertIn('确认',r['answer'])

    def test_finite_courtesy_does_not_require_product_or_semantic_route(self):
        self.model.fail='route'
        r=self.s.respond('谢啦！');self.assertEqual(r['action'],'chat')
        self.assertFalse(any(c[0]=='route' for c in self.model.calls))

    def test_explicit_confirmation_reads_fresh_evidence(self):
        self.pending_confirmation();r=self.s.respond('确认')
        self.assertEqual(r['action'],'answer');self.assertTrue(r['citations']);self.assertIsNone(self.s.state['pending'])

    def test_confirmation_respects_site_and_product_scope(self):
        self.pending_confirmation()
        self.assertEqual(self.s.respond('确认',{'store_id':'other'})['action'],'not_found')
        self.assertEqual(self.s.respond('确认',{'skus':[C]})['action'],'clarify')
        self.assertFalse(any(c[0]=='read' for c in self.model.calls))

    def test_reset_clears_intent_confirmation(self):
        self.pending_confirmation();self.s.respond('重置')
        self.assertNotEqual(self.s.respond('确认')['action'],'answer')
        self.assertFalse(any(c[0]=='read' for c in self.model.calls))

    def test_confirmation_can_be_cancelled(self):
        self.pending_confirmation();self.s.respond('不是')
        self.assertIsNone(self.s.state['pending']);self.assertFalse(any(c[0]=='read' for c in self.model.calls))


class EvidenceCases(unittest.TestCase):
    def test_courtesy_grammar_rejects_embedded_product_request(self):
        self.assertTrue(is_short_thanks('多谢你啦！'));self.assertTrue(is_short_thanks('Thanks!'))
        self.assertFalse(is_short_thanks('谢谢，请查商品宽度'));self.assertFalse(is_short_thanks('谢罪是什么意思'))
    def test_conflict_is_derived_only_from_current_page(self):
        page={'sku':A,'page_id':'p2','page':2,'text':'宽度 / width: 25 inches\n35”D x 28.3”W x 33.9”H'}
        c=current_page_conflicts([page],'width')
        self.assertEqual(len(c),1);self.assertEqual(c[0]['page_id'],'p2')
        self.assertIn('28.3',conflict_notice(c));self.assertIn('不能确认',conflict_notice(c))
        self.assertEqual(current_page_conflicts([page],'weight'),[])
        self.assertEqual(current_page_conflicts([{**page,'text':'width: 25 inches'}],'width'),[])

    def test_equal_dimensions_and_depth_are_not_false_conflicts(self):
        p={'sku':A,'page_id':'p2','page':2,'text':'width: 25 inches\n25”W x 37”D'}
        self.assertEqual(current_page_conflicts([p],'width'),[])
        self.assertEqual(current_page_conflicts([{**p,'text':'length: 37 inches\n35”D'}],'length'),[])

    def test_numeric_excerpt_keeps_exact_field_units_and_decimal(self):
        quote='width: 27.25 inches\nheight: 36 inches\nweight: 33 pounds'
        self.assertEqual(field_excerpt(quote,'width'),'width: 27.25 inches')
        self.assertEqual(field_excerpt('No assembly. Wipe with dry cloth.','assembly'),'No assembly. Wipe with dry cloth.')

    def test_facets_contract_rejects_fabricated_operations(self):
        self.assertEqual(validate_facets('{"task":"execute","need":"none"}')['task'],'execute')
        for raw in ('{"task":"refund_done","need":"none"}','{"task":"fact","need":"none","answer":"25"}','[]','No'):
            with self.assertRaises((ValueError,TypeError)):validate_facets(raw)


if __name__=='__main__':unittest.main()

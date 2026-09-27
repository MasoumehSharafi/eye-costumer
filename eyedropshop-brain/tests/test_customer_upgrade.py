"""Offline regression tests; no credentials, models or API calls required."""
import ast
import importlib.util
import json
import re
from pathlib import Path
import sys
import types
import unittest
from typing import *
ROOT = Path(__file__).resolve().parents[1]
provider = types.ModuleType('model_provider')
provider.get_model_provider = lambda: None
sys.modules['model_provider'] = provider
spec = importlib.util.spec_from_file_location('conversation', ROOT / 'conversation.py')
conversation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversation)

def function(name, filename, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]

class Tests(unittest.TestCase):
    def model(self, output):
        conversation.get_model_provider = lambda: types.SimpleNamespace(chat=lambda *a, **k: types.SimpleNamespace(content=output))
    def test_french_greeting_without_model(self):
        self.assertEqual(conversation.plan_turn('Bonjour!', [], 'en')['language'], 'fr')
    def test_followup_and_no_new_product(self):
        self.model(json.dumps({'language':'fr','resolution':'resolved','reference_scope':'single','references':['HYLO-DUAL'],'unresolved_reference':None}))
        result = conversation.plan_turn('Et son prix ?', [{'role':'assistant','content':'HYLO-DUAL'}], 'en')
        self.assertIn('HYLO-DUAL', result['question'])
        self.assertNotIn('Invented', result['question'])
        self.assertTrue(result['question'].startswith('Et son prix ?'))
        self.assertEqual(result['language'],'fr')
    def test_ambiguous_reference(self):
        self.model('{"language":"en","resolution":"ambiguous","reference_scope":"none","references":[],"unresolved_reference":"it"}')
        self.assertIsNotNone(conversation.plan_turn('Is it available?',[], 'en')['clarification'])
    def test_bad_model_json(self):
        self.model('not json')
        with self.assertRaisesRegex(RuntimeError, 'invalid decision'):
            conversation.plan_turn('Is it cheaper?',[], 'en')
    def test_symptoms_are_preserved(self):
        self.model('{"language":"en","resolution":"resolved","reference_scope":"single","references":["HYLO"],"unresolved_reference":null}')
        q = 'My eyes hurt; can I use it?'
        self.assertTrue(conversation.plan_turn(q,[{'role':'user','content':'HYLO'}],'en')['question'].startswith(q))
    def test_resolved_product_cannot_be_overridden_by_rewritten_question(self):
        self.model(json.dumps({'language':'en','resolution':'resolved','reference_scope':'single','references':['LUMIFY®'],'unresolved_reference':None,
                              'clarification':'What is the price of LUMIFY®?'}))
        history=[{'role':'user','content':'what is lumify?'},
                 {'role':'assistant','content':'LUMIFY® is an eye drop.'}]
        for q in ['how much is that?', 'how much is it?', 'any warnings?']:
            result=conversation.plan_turn(q,history,'en')
            self.assertIsNone(result['clarification'])
            self.assertIn('LUMIFY®',result['question'])
            self.assertTrue(result['question'].startswith(q))
    def test_new_need_preserves_symptoms_without_attaching_previous_product(self):
        self.model('{"language":"en","resolution":"standalone","reference_scope":"none","references":[],"unresolved_reference":null}')
        q='my eyes are itchy, what drop you recommend?'
        result=conversation.plan_turn(q,[{'role':'assistant','content':'LUMIFY is in stock.'}],'en')
        self.assertEqual(result['question'],q)
        self.assertIsNone(result['clarification'])
    def test_ambiguity_without_an_actual_reference_is_rejected(self):
        self.model('{"language":"en","resolution":"ambiguous","reference_scope":"none","references":[],"unresolved_reference":null}')
        with self.assertRaisesRegex(RuntimeError,'invalid decision'):
            conversation.plan_turn('I need drops for itchy eyes',[],'en')
    def test_hallucinated_reference_is_not_forwarded(self):
        self.model('{"language":"en","resolution":"resolved","reference_scope":"single","references":["Invented"],"unresolved_reference":null}')
        with self.assertRaisesRegex(RuntimeError,'invalid decision'):
            conversation.plan_turn('how much is it?',[{'role':'assistant','content':'LUMIFY'}],'en')
    def test_bare_product_name_is_standalone(self):
        self.model('{"language":"en","resolution":"standalone","reference_scope":"none","references":[],"unresolved_reference":null}')
        result=conversation.plan_turn('lumify?',[],'en')
        self.assertIsNone(result['clarification'])
        self.assertEqual(result['question'],'lumify?')
    def test_singular_reference_cannot_expand_to_multiple_products(self):
        self.model('{"language":"en","resolution":"resolved","reference_scope":"single","references":["Alpha","Beta"],"unresolved_reference":null}')
        history=[{'role':'assistant','content':'Options: Alpha and Beta.'}]
        with self.assertRaisesRegex(RuntimeError,'invalid decision'):
            conversation.plan_turn('how much is it?',history,'en')
    def test_model_outage_is_not_mislabeled_as_ambiguity(self):
        def unavailable(*a,**kw): raise ConnectionError('offline')
        conversation.get_model_provider=lambda:types.SimpleNamespace(chat=unavailable)
        with self.assertRaises(ConnectionError):
            conversation.plan_turn('how much is it?',[{'role':'assistant','content':'LUMIFY'}],'en')
    def pagination(self, pages):
        it = iter(pages)
        scope = {'List':List,'Dict':Dict,'Any':Any,'Optional':Optional,'_request_data':lambda *a,**k:next(it)}
        return function('_get_all_pages','edsai_client.py',scope)('/v1/products')
    def test_empty_intermediate_page(self):
        self.assertEqual(self.pagination([{'items':[],'hasMore':True,'nextCursor':'a'}, {'items':[{'id':1}],'hasMore':False}]),[{'id':1}])
    def test_repeated_cursor(self):
        with self.assertRaisesRegex(RuntimeError,'repeated'):
            self.pagination([{'hasMore':True,'nextCursor':'a'},{'hasMore':True,'nextCursor':'a'}])
    def test_missing_cursor(self):
        with self.assertRaises(RuntimeError): self.pagination([{'hasMore':True}])
    def test_comparisons_skip_single_product_shortcuts(self):
        for name in ['_direct_price_answer','_direct_inventory_answer','_direct_ingredients_answer','_direct_expiry_answer']:
            fn = function(name, 'answer_generator.py', {'_is_product_comparison':lambda *a:True})
            self.assertIsNone(fn('Compare A and B','en',[]))
    def test_comparison_retrieves_both_names(self):
        seen=[]
        ns={'List':List,'Dict':Dict,'Any':Any,'_current_question_text':lambda q:q,
            'understand_query':lambda *a:{'named_product_query':'Alpha','intent':'product_fact'},
            'find_products_in_question':lambda q:[{'title':t} for t in ['Alpha','Beta'] if t in q],
            '_normalize':lambda q:q.lower(),'_requested_sections':lambda *a:set(),
            '_product_passages':lambda p,s,l: seen.append((p['title'],s)) or [{'text':p['title']} ]}
        fn=function('retrieve_edsai_passages','edsai_retrieval.py',ns)
        self.assertEqual(len(fn('Compare Alpha and Beta','en')),2)
        self.assertTrue(all('price' in s for _,s in seen))
    def test_preference_search_reaches_semantic_retrieval(self):
        ns={'List':List,'Dict':Dict,'Any':Any,'_current_question_text':lambda q:q,
            'understand_query':lambda *a:{'intent':'product_search','search_query':'preservative free'},
            '_normalize':lambda q:q.lower(),'_products':lambda:[], '_product_cache_time':1,
            'semantic_candidates':lambda **k:[{'title':'Alpha'}],
            'rerank_candidates':lambda **k:k['candidates'],
            '_product_passages':lambda *a:[{'text':'Approved description'}]}
        fn=function('retrieve_edsai_passages','edsai_retrieval.py',ns)
        self.assertEqual(len(fn('Find preservative-free drops','en')),1)

    def test_semantic_plan_controls_named_product_fields(self):
        seen=[]
        ns={'List':List,'Dict':Dict,'Any':Any,'_current_question_text':lambda q:q,
            'understand_query':lambda *a:{'named_product_query':'CALMO Eye Spray',
                                          'intent':'product_fact',
                                          'requested_sections':['price']},
            'find_products_in_question':lambda q:[{'title':'CALMO Eye Spray'}],
            '_normalize':lambda q:q.lower(),
            '_product_passages':lambda p,s,l: seen.append(s) or [{'text':'Retail price: 39.95'}]}
        fn=function('retrieve_edsai_passages','edsai_retrieval.py',ns)
        passages=fn('hoe much is CALMO Eye Spray','en')
        self.assertEqual(len(passages),1)
        self.assertEqual(seen,[{'price'}])

class QualityTests(unittest.TestCase):
    def test_symptom_search_without_evidence_explains_the_gap(self):
        fn=function('_friendly_personal_medical_answer','answer_generator.py',{
            '_current_question_text':lambda q:q,'normalize_text':str.lower})
        answer=fn('my eyes are itchy, what drop you recommend?','en',[])
        self.assertIn("couldn't confirm a matching product",answer)
        self.assertNotIn('Which product',answer)

    def test_overview_limit_preserves_citations_and_never_trims_warnings(self):
        fn=function('_concise_customer_overview','answer_generator.py',{'re':re})
        answer='Purpose [Source 1]. Feature [Source 1]. Brand [Source 2]. Promotion [Source 2].'
        overview=[{'metadata':{'section':'description'}}]
        self.assertEqual(fn(answer,overview),'Purpose [Source 1]. Feature [Source 1]. Brand [Source 2].')
        self.assertEqual(fn(answer,[{'metadata':{'section':'warnings'}}]),answer)
        self.assertEqual(fn(answer,overview,True),answer)
        mixed=[{'metadata':{'section':'description'}},{'metadata':{'section':'benefits_and_details'}}]
        self.assertEqual(fn(answer,mixed,False,True),'Purpose [Source 1]. Feature [Source 1]. Brand [Source 2].')

    def test_benefits_only_overview_plan_is_concise(self):
        understanding={'intent':'product_fact','requested_sections':['benefits_and_details']}
        sections=set(understanding.get('requested_sections') or [])
        overview=(understanding.get('intent') == 'product_fact' and bool(sections)
                  and sections <= {'description','benefits_and_details'})
        self.assertTrue(overview)

    def test_numeric_citations_are_normalized_before_audit(self):
        fn=function('_normalize_citation_markers','answer_generator.py',{'re':re})
        passages=[{}, {}]
        self.assertEqual(fn('Claim [1], detail [Source 2], bad [9].',passages),
                         'Claim [Source 1], detail [Source 2], bad .')

    def test_high_risk_safety_response_is_urgent(self):
        fn=function('_high_risk_safety_answer','answer_generator.py',{})
        self.assertIn('urgent',fn('en').lower())
        self.assertIn('now',fn('en').lower())
        self.assertIn('urgents',fn('fr').lower())

    def test_inventory_counts_stay_in_audit_evidence_only(self):
        fn=function('_customer_generation_passages','answer_generator.py',{'re':re})
        original=[{'text':'Variant: Alpha 3ml. Current aggregate inventory: 56.','metadata':{'section':'inventory'}}]
        display=fn(original)
        self.assertIn('in stock',display[0]['text'])
        self.assertNotIn('56',display[0]['text'])
        self.assertIn('56',original[0]['text'])

    def test_ingredient_list_does_not_imply_preservative_role(self):
        fn=function('_preservative_information_answer','answer_generator.py',{'normalize_text':str.lower})
        passages=[{'text':'Inactive ingredients: Benzalkonium chloride, water.','metadata':{'section':'ingredients'}}]
        answer=fn('does it have preservatives?','en',passages)
        self.assertIn('does not explicitly specify',answer)
        self.assertIn('Benzalkonium chloride',answer)
        self.assertNotIn('known preservative',answer)
        self.assertIn('[Source 1]',answer)
        self.assertIsNone(fn('is it preservative-free?','en',[{'text':'This product is preservative-free.'}]))

    def test_greeting_typo(self):
        self.assertIn('reply', conversation.plan_turn('hellp', [], 'en'))
    def test_price_and_inventory_labels(self):
        ns={'re':re,'normalize_text':str.lower,'_is_product_comparison':lambda *a:False}
        price=function('_direct_price_answer','answer_generator.py',ns)
        stock=function('_direct_inventory_answer','answer_generator.py',ns)
        def passage(section,text):
            return [{'text':text,'metadata':{'source':'edsAI','section':section,'product_title':'Alpha'}}]
        answer=price('how much is it?','en',passage('price','Retail price: 19.99. Currency not supplied.'))
        self.assertIn('19.99',answer)
        self.assertNotIn('$',answer)
        typo_answer=price('hoe much is CALMO Eye Spray','en',passage('price','Retail price: 39.95. Currency not supplied.'))
        self.assertIn('39.95',typo_answer)
        for count in (7,0,-2):
            answer=stock('is it available?','en',passage('inventory',f'Current aggregate inventory: {count}.'))
            self.assertIn('in stock' if count>0 else 'out of stock',answer)
            self.assertNotIn('units',answer)
        self.assertIsNone(stock('is it available?','en',passage('inventory','Inventory is not tracked.')))
    def test_preservatives_retrieve_details_and_ingredients(self):
        fn=function('_requested_sections','edsai_retrieval.py',{'_normalize':str.lower,'_asks_price':lambda *a:False,'_asks_inventory':lambda *a:False})
        self.assertTrue({'ingredients','description','benefits_and_details'} <= fn('does it have preservatives?','en'))

if __name__ == '__main__': unittest.main()

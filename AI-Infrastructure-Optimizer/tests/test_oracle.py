import inspect,json,tempfile,unittest
from pathlib import Path
from src.oracle_policy import quota,random_plan,validate,choose_candidate
from src.oracle_data import fresh,score

class TestOracle(unittest.TestCase):
    def test_quotas_rates_and_cooldown(self):
        for rate in (0,10,20,30,40,50):
            plan=random_plan(64,rate,11);validate(plan,rate)
            self.assertEqual(sum(map(len,plan)),28*rate*63//100)
        self.assertEqual(random_plan(32,30,77),random_plan(32,30,77))
    def test_privileged_selection_rule_is_explicit(self):
        cs=[{'index':0,'grade':{'content_correct':False,'format_ok':True},'baseline_token_match':True,'length_difference':0},{'index':1,'grade':{'content_correct':True,'format_ok':False},'baseline_token_match':False,'length_difference':4}]
        self.assertEqual(choose_candidate(cs),1)
    def test_new_tasks_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            short,long=fresh(d);self.assertEqual((len(short),len(long)),(8,4))
            self.assertTrue(all(len(t['target_words'])==24 for t in long))
            Path(d,'prior.json').write_text(json.dumps(short))
            with self.assertRaises(ValueError):fresh(d)
    def test_copy_not_just_prefix(self):
        t={'score_kind':'copy_words','target_words':['la']*24}
        self.assertTrue(score(t,'la '*24)['content_correct']);self.assertFalse(score(t,'la '*23)['content_correct'])
    def test_no_labels_or_distances_in_ideal_branch(self):
        s=Path('src/oracle_runtime.py').read_text();self.assertNotIn('reference',s);self.assertNotIn('content_correct',s)
        self.assertIn("if method=='overhead'",s);self.assertNotIn('choose_candidate',s)

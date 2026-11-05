import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src import pvcg_policy as policy,pvcg_dataset as dataset,pvcg_experiment as exp
from src.vrg_policy import parse
from src.rcav_runtime import LocalClient
from src.solver300_dataset import task_key
from src.solver300_experiment import check_references
from test_rcav import fake_send
from test_controlled_validation import DummySampler,fake_reset
Q='What does this Python program output?\n```python\nprint(4)\n```\nA) 4\nB) 3\nC) 2\nD) Error'
def p():return dict(parse(Q,'A'),answer='A',source='model',question=Q,task_id='one')
def profile(mean=.3,n=30):return {'version':policy.VERSION,'min_support':10,'uncertainty_weight':.5,'cpu_penalty':.1,'groups':{'coding|resolved':{'samples':n,'mean_gain':mean,'standard_error':.01,'median_cpu':.1}}}

class TestPVCG(unittest.TestCase):
    def test_support_value_and_cost_control_decision(self):
        with patch.object(policy,'choose',return_value={'verify':True}):
            self.assertTrue(policy.decide(Q,p(),profile(),{})['verify'])
            self.assertFalse(policy.decide(Q,p(),profile(-.1),{})['verify'])
            self.assertFalse(policy.decide(Q,p(),profile(n=2),{})['verify'])
            high=profile();high['cpu_penalty']=10
            self.assertFalse(policy.decide(Q,p(),high,{})['verify'])

    def test_no_gold_or_future_response_allowed(self):
        for key in policy.FORBIDDEN:
            initial=p();initial[key]='secret'
            with self.assertRaises(ValueError):policy.decide(Q,initial,profile(),{})

    def test_tool_and_legacy_skip_are_preserved(self):
        initial=p();initial['source']='tool'
        self.assertFalse(policy.decide(Q,initial,profile(),{})['verify'])
        with patch.object(policy,'choose',return_value={'verify':False}):self.assertFalse(policy.decide(Q,p(),profile(),{})['verify'])

    def test_gate_finishes_before_verifier_and_no_post_revision_rejection(self):
        initial=p();d={'verify':True,'reason':'test'}
        class Client:
            def primary(self,q):raise AssertionError('No primary regeneration')
            def verify(self,q,a):return {'answer':'B','call_id':'actual','status':'generated'},{}
        with patch.object(exp,'decision',return_value=d):r=exp.execute([initial],'pvcg',Client(),{}, {},['one'],{'one':d})
        self.assertEqual(r['rows'][0]['final_prediction'],'B')
        self.assertEqual(r['rows'][0]['decision_phase'],'before_verification')
        self.assertEqual(len(r['calls']),1)

    def test_skipped_gate_makes_zero_llm_calls_and_changed_plan_is_rejected(self):
        class Client:
            def verify(self,*args):raise AssertionError('Skipped requests must not call a model')
        r=exp.execute([p()],'none',Client(),{}, {},['one'],{'one':{'verify':False,'reason':'no_verification'}})
        self.assertEqual(r['calls'],[])
        with self.assertRaises(RuntimeError):exp.execute([p()],'none',Client(),{}, {},['one'],{'one':{'verify':True}})

    def test_new_dataset_splits_references_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            va,te=dataset.build(folder);self.assertEqual((len(va),len(te)),(100,300))
            self.assertFalse({task_key(r['question']) for r in va}&{task_key(r['question']) for r in te})
            self.assertEqual(len(check_references(va+te)),80)
            Path(folder,'used.json').write_text(json.dumps(te))
            with self.assertRaises(ValueError):dataset.build(folder)

    def test_neutral_prior_and_empirical_transition_effect(self):
        rr=[{'question':Q,'primary':p(),'gain':1 if i<20 else -1,'verifier_cpu_seconds':.2} for i in range(30)]
        with patch.object(policy,'choose',return_value={'verify':True}):g=policy.fit_groups(rr,{})
        self.assertEqual(g['coding|resolved']['samples'],30)
        self.assertAlmostEqual(g['coding|resolved']['mean_gain'],10/34)

    def test_full_mock_pipeline_only_in_temporary_directory(self):
        snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(exp,'check_ollama'),patch('src.controlled_validation.process_snapshot',return_value=snapshot), \
                 patch.object(exp.controls,'MemorySampler',DummySampler),patch.object(exp.controls,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(exp.controls,'http_json',return_value={'models':[{'name':m} for m in (exp.LARGE,exp.SMALL)]}), \
                 patch.object(LocalClient,'send',new=fake_send),contextlib.redirect_stdout(io.StringIO()):prefix=exp.run(folder)
            meta=json.loads(Path(f'{prefix}_metadata.json').read_text())
            self.assertEqual(meta['outcome_rows'],12000)
            self.assertEqual(meta['actual_primary_calls'],480)
            self.assertEqual(meta['new_calibration_calls'],360)

if __name__=='__main__':unittest.main()

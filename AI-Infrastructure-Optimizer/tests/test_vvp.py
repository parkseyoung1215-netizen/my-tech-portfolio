import copy,contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_rcav import fake_send
from test_controlled_validation import DummySampler,fake_reset
from src import vvp_dataset as data
from src import vvp_model as model
from src import vvp_runtime as runtime
from src import vvp_experiment as experiment
from src.rcav_runtime import LocalClient
from src.solver300_experiment import check_references
Q='What does this print?\n```python\nprint(sum(range(1,5)))\n```\nA) 1\nB) 2\nC) 3\nD) 10'
P={'question':Q,'task_id':'t','answer':'A','prediction':'A','format_compliant':True,'source':'model'}

def fake_predict(q,p,m):return {'p_repair':.4,'p_damage':.1,'p_unchanged':.5,'expected_value':.3,'estimated_cpu_seconds':.03,'value_per_estimated_cpu':10.}
def fake_fit(rows,validation):return {'version':model.VERSION,'fake_unit_test_model':True,'training_rows':len(rows),'validation_rows':len(validation)}

class TestVVP(unittest.TestCase):
    def test_new_disjoint_splits_reference_validated(self):
        with tempfile.TemporaryDirectory() as folder:v,t=data.build(folder)
        self.assertEqual((len(v),len(t)),(100,300));self.assertEqual(len({data.task_key(r['question']) for r in v+t}),400)
        self.assertEqual(len(check_references(v+t)),80)
        self.assertEqual(sum(r['cohort']=='A' for r in t),150)

    def test_fit_predictions_probabilities_and_gold_free(self):
        rows=[{'features':model.observed(Q,P),'label':i%3,'cpu_seconds':.03+i*.0001} for i in range(12)]
        fit=model.train(rows,rows[:6]);p=model.predict(Q,P,fit)
        self.assertAlmostEqual(p['p_repair']+p['p_damage']+p['p_unchanged'],1)
        self.assertAlmostEqual(p['expected_value'],p['p_repair']-p['p_damage'])
        more={**P,'expected':'D','verification_answer':'D'};self.assertEqual(p,model.predict(Q,more,fit))
        self.assertGreater(p['estimated_cpu_seconds'],0)
        tool={**P,'source':'tool'};self.assertEqual(model.predict(Q,tool,fit)['p_repair'],0)

    def test_budgets_enforced_without_gold_and_zero_calls(self):
        ps=[{**P,'task_id':str(i)} for i in range(20)];arrival=[p['task_id'] for p in ps]
        with patch.object(runtime,'predict',side_effect=fake_predict),patch.object(runtime,'estimate_cpu',return_value=.03):
            for pct in runtime.BUDGETS:
                plan=runtime.allocate(ps,'vvp',pct,{}, {},arrival)
                self.assertLessEqual(len(plan['selected']),int(20*pct/100))
                self.assertLessEqual(plan['selected_estimated_cpu'],plan['predicted_cpu_quota']+1e-12)
            self.assertEqual(len(runtime.allocate(ps,'full',100,{}, {},arrival)['selected']),20)
            self.assertEqual(runtime.allocate(ps,'none',100,{}, {},arrival)['selected'],[])

    def test_frozen_primary_not_generated_again(self):
        original=copy.deepcopy(P);client=LocalClient(experiment.LARGE)
        with patch.object(client,'primary',side_effect=AssertionError('regeneration')),patch.object(LocalClient,'send',new=fake_send), \
             patch.object(runtime,'predict',side_effect=fake_predict),patch.object(runtime,'estimate_cpu',return_value=.03):
            result=runtime.execute([P],'vvp',100,client,{}, {},['t'])
        self.assertEqual(P,original);self.assertEqual(len(result['calls']),1)

    def test_past_training_not_evaluations(self):
        keys=experiment.old_evaluation_keys(data.ROOT/'data')
        for path in experiment.TRAIN.values():self.assertFalse({data.task_key(r['question']) for r in experiment.read_csv(path)}&keys)

    def test_full_grid_offline_one_model_temp_only(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch('src.controlled_validation.process_snapshot',return_value=snapshot), \
                 patch.object(experiment.controls,'MemorySampler',DummySampler),patch.object(experiment.controls,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(experiment.controls,'http_json',return_value={'models':[{'name':experiment.LARGE}]}), \
                 patch.object(LocalClient,'send',new=fake_send),patch.object(model,'train',side_effect=fake_fit), \
                 patch.object(runtime,'predict',side_effect=fake_predict),patch.object(runtime,'estimate_cpu',return_value=.03),contextlib.redirect_stdout(io.StringIO()):
                prefix=experiment.run(folder)
            meta=json.loads(Path(f'{prefix}_metadata.json').read_text())
            self.assertEqual(meta['outcome_rows'],54000)
            self.assertTrue(Path(f'{prefix}_plans.json').exists())

if __name__=='__main__':unittest.main()

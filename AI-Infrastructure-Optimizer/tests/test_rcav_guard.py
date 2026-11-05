import contextlib,inspect,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter
from test_rcav import fake_send
from src import rcav_guard_policy as guard
from src import rcav_guard_dataset as data
from src import rcav_guard_experiment as experiment
from src import rcav_guard_runtime as runtime
from src import rcav_policy as base
from src.rcav_runtime import LocalClient
from src.solver300_experiment import check_references

Q='What does this Python program output?\n```python\nprint(sum(range(21, 28, 2)))\n```\nA) aa\nB) bb\nC) cc\nD) dd'
def profile():
    rows=[{'observed':base.observable(Q,{'answer':'A','prediction':'A','format_compliant':True}),
        'primary_correct':True,'final_correct':True,'verification_model_seconds':.04,'verification_prompt_chars':300} for i in range(20)]
    return base.fit(rows)
def training(n=12,fixed=4,harmed=0,cost=.04):
    return [{'question':Q,'response_status':'bare','primary_correct':i>=fixed,'final_correct':i<fixed or i>=fixed+harmed,
        'verification_model_seconds':cost} for i in range(n)]

class TestRCAVGuard(unittest.TestCase):
    def test_disjoint_balanced_reference_verified_tasks(self):
        with tempfile.TemporaryDirectory() as folder:c,e=data.build(folder)
        self.assertEqual(len({data.task_key(t['question']) for t in c+e}),250)
        self.assertEqual(Counter(t['category'] for t in e),{cat:30 for cat in ('arithmetic','coding','logic','knowledge','comprehension')})
        self.assertEqual(Counter(t['regime'] for t in e),{'near':100,'shift':40,'fact_holdout':10})
        self.assertEqual(len(check_references(c+e)),50)

    def test_duplicate_old_question_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'old.json').write_text(json.dumps({'question':'Which planet has the greatest mass in the Solar System?'}))
            with self.assertRaises(ValueError):data.build(folder)

    def test_historical_eleven_same_primary_and_measured_repairs(self):
        rows=experiment.historical_loss()
        self.assertEqual(Counter(r['category'] for r in rows),{'coding':6,'logic':5})
        self.assertTrue(all(r['same_primary_all3'] and r['rcav_skipped_all3'] and r['selective_repaired_all3'] for r in rows))

    def test_structure_features_do_not_execute_or_read_gold(self):
        self.assertEqual(guard.family(Q),'coding:aggregate')
        self.assertEqual(guard.family('Reason about links A->B, B->C.'),'logic:graph')
        self.assertEqual(list(inspect.signature(guard.decide).parameters),['question','primary','prompt_chars','base_profile','guard_profile'])

    def test_guard_adds_only_supported_profitable_affordable_repairs(self):
        p={'answer':'A','prediction':'A','source':'model','format_compliant':True}
        b=profile();g=guard.fit(training())
        self.assertFalse(base.decide(Q,p,300,b,.05)['verify'])
        self.assertTrue(guard.decide(Q,p,300,b,g)['additional_verification'])
        self.assertFalse(guard.decide(Q,p,300,b,guard.fit(training(n=7)))['verify'])
        self.assertFalse(guard.decide(Q,p,300,b,guard.fit(training(fixed=4,harmed=4)))['verify'])
        self.assertFalse(guard.decide(Q,p,300,b,guard.fit(training(cost=1)))['verify'])
        self.assertFalse(guard.decide(Q,p,300,b,guard.fit(training(fixed=0)))['verify'])
        with self.assertRaises(ValueError):guard.fit([])

    def test_no_guard_for_proven_tool(self):
        p={'answer':'2','prediction':'2','source':'tool','format_compliant':True}
        self.assertFalse(guard.decide('1+1',p,300,profile(),guard.fit(training()))['verify'])

    def test_runtime_decision_before_verify_and_no_hidden_retry(self):
        events=[];original=guard.decide
        def gate(*args):events.append('decision');return original(*args)
        def sender(self,prompt,stage):events.append(stage);return fake_send(self,prompt,stage)
        with patch.object(LocalClient,'send',new=sender),patch.object(guard,'decide',side_effect=gate):
            r=runtime.execute_block([Q],'guarded',LocalClient(experiment.LARGE),profile(),guard.fit(training()),{})
        self.assertEqual(events,['primary','decision','verify']);self.assertEqual(len(r['calls']),2)

    def test_complete_offline_pipeline_only_temp_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            keep=Path(folder,'keep.csv');keep.write_text('keep\n')
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(LocalClient,'send',new=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder)
            self.assertEqual(keep.read_text(),'keep\n')
            self.assertEqual(len(experiment.read_csv(f'{prefix}_evaluation.csv')),2700)
            self.assertEqual(len(experiment.read_csv(f'{prefix}_loss_rechecks.csv')),33)
            self.assertTrue(Path(f'{prefix}_guard_profile.json').exists())

if __name__=='__main__':unittest.main()

"""Offline tests. Artificial measurements never leave temporary test directories."""
import contextlib,hashlib,inspect,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from collections import Counter
from src import rcav_policy as policy
from src import rcav_runtime as runtime
from src import rcav_dataset as data
from src import rcav_experiment as experiment
from src.solver300_experiment import check_references
Q='Which element has this symbol?\nA) aa\nB) bb\nC) cc\nD) dd'

def primary():return {'answer':'A','prediction':'A','format_compliant':True,'source':'model'}
def calibration(wrong=15,fixed=12,harmed=0):
    return [{'observed':policy.observable(Q,primary()),'primary_correct':i>=wrong,
        'final_correct':(i<fixed if i<wrong else i>=wrong+harmed),
        'verification_prompt_chars':300+i*5,'verification_model_seconds':.02+i*.001} for i in range(20)]

def fake_send(self,prompt,stage):
    payload={'model':self.model,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':64}}
    return {'call_id':uuid4().hex,'stage':stage,'model':self.model,'request_payload':payload,
        'request_sha256':hashlib.sha256(json.dumps(payload).encode()).hexdigest(),'answer':'A','status':'generated','error':'',
        'input_tokens':10,'output_tokens':1,'eval_seconds':.01,'prefill_seconds':.02,'total_model_seconds':.04,
        'load_seconds':.001,'done_reason':'stop','format_compliant':False,'parse_mode':'unresolved','elapsed_seconds':.05}

class TestRCAV(unittest.TestCase):
    def test_balanced_disjoint_new_tasks_and_references(self):
        with tempfile.TemporaryDirectory() as folder:cal,ev=data.build(folder)
        self.assertEqual(Counter(t['category'] for t in cal),{c:20 for c in ('arithmetic','knowledge','logic','coding','comprehension')})
        self.assertEqual(Counter(t['category'] for t in ev),{c:30 for c in ('arithmetic','knowledge','logic','coding','comprehension')})
        self.assertEqual(len({data.task_key(t['question']) for t in cal+ev}),250)
        self.assertEqual(len(check_references(cal+ev)),50)

    def test_duplicate_existing_stem_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'old.json').write_text(json.dumps({'question':'Which chemical element has the symbol He?'}))
            with self.assertRaises(ValueError):data.build(folder)

    def test_no_gold_or_verifier_result_interface(self):
        self.assertEqual(list(inspect.signature(policy.decide).parameters),['question','primary','verification_prompt_chars','profile','weight'])
        self.assertEqual(list(inspect.signature(runtime.execute_block).parameters),['questions','method','client','profile','selective_profile'])
        p=primary();profile=policy.fit(calibration())
        a=policy.decide(Q,p,350,profile)
        p.update(expected='B',verification_result='D')
        self.assertEqual(a,policy.decide(Q,p,350,profile))

    def test_risk_benefit_and_cost_all_affect_decision(self):
        profile=policy.fit(calibration());p=primary()
        low=policy.decide(Q,p,350,profile,0)
        self.assertTrue(low['verify']);self.assertGreater(low['expected_error_reduction'],0)
        high=policy.decide(Q,p,350,profile,100)
        self.assertFalse(high['verify'])
        expensive=policy.decide(Q,p,100000,profile,.05)
        self.assertFalse(expensive['verify']);self.assertTrue(expensive['cost_extrapolation'])
        harmful=policy.fit(calibration(wrong=0,fixed=0,harmed=10))
        self.assertFalse(policy.decide(Q,p,350,harmful,0)['verify'])

    def test_safe_tool_proof_and_unresolved_handling(self):
        profile=policy.fit(calibration());p=primary();p.update(source='tool',prediction='2',answer='2')
        d=policy.decide('1+1',p,350,profile,0)
        self.assertFalse(d['verify']);self.assertEqual(d['risk_estimate'],0)
        p.update(source='model',prediction=None,answer='A or B')
        self.assertEqual(policy.decide(Q,p,350,profile)['risk_estimate'],1)

    def test_unknown_stratum_falls_back_without_fake_samples(self):
        profile=policy.fit(calibration())
        d=policy.decide('An unsupported request',primary(),400,profile)
        self.assertEqual(d['calibration_level'],'global');self.assertEqual(d['calibration_samples'],20)
        with self.assertRaises(ValueError):policy.decide(Q,primary(),300,{},.05)
        rows=calibration();rows[0]['verification_model_seconds']=None
        with self.assertRaises(ValueError):policy.fit(rows)

    def test_decision_precedes_verification(self):
        events=[];client=runtime.LocalClient(experiment.LARGE);original=policy.decide
        def gate(*args):events.append('decision');return original(*args)
        def sender(self,prompt,stage):events.append(stage);return fake_send(self,prompt,stage)
        with patch.object(runtime.LocalClient,'send',new=sender),patch.object(policy,'decide',side_effect=gate):
            r=runtime.execute_block([Q],'rcav_zero',client,policy.fit(calibration()),{})
        self.assertEqual(events,['primary','decision','verify'])
        self.assertEqual(len(r['calls']),2)

    def test_invalid_verifier_preserves_primary_without_retry(self):
        def send(self,prompt,stage):
            a=fake_send(self,prompt,stage)
            if stage=='verify':a['answer']='B or C'
            return a
        with patch.object(runtime.LocalClient,'send',new=send):
            r=runtime.execute_block([Q],'baseline',runtime.LocalClient(experiment.LARGE),{}, {})
        self.assertEqual(r['rows'][0]['final_prediction'],'A')
        self.assertTrue(r['rows'][0]['verification_preserved_primary']);self.assertEqual(len(r['calls']),2)

    def test_pareto_dominance_requires_quality_and_coverage(self):
        rr=[{'model':'m','method':m,'accuracy_mean':a,'answer_rate_mean':c,
            'model_processing_seconds_mean':t,'processing_seconds_mean':t,'ollama_cpu_seconds_mean':t}
            for m,a,c,t in [('a',.8,1,10),('b',.8,1,5),('c',.9,.8,2)]]
        p=experiment.pareto(rr)
        self.assertFalse(p[0]['on_observed_frontier']);self.assertTrue(p[3]['on_observed_frontier'])

    def test_full_offline_two_model_two_repeat_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            keep=Path(folder,'keep.csv');keep.write_text('keep\n')
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime.LocalClient,'send',new=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO(json.dumps({'models':[{'name':experiment.LARGE},{'name':experiment.SMALL}]}))
                prefix=experiment.run(folder,repeats=2)
            self.assertEqual(keep.read_text(),'keep\n')
            self.assertEqual(len(experiment.read_csv(f'{prefix}_evaluation.csv')),3000)
            self.assertTrue(Path(f'{prefix}_profiles.json').exists())
            self.assertTrue(Path(f'{prefix}_pareto.csv').exists())
            self.assertEqual(len(experiment.read_csv(f'{prefix}_aggregate.csv')),10)

if __name__=='__main__':unittest.main()

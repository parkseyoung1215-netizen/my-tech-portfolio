"""Offline control tests; these outputs are never experimental measurements."""
from collections import Counter
import contextlib
import inspect
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from src import skipwork_dataset as data
from src import skipwork_runtime as runtime
from src import skipwork_experiment as experiment
from src import skipwork_audit
from src.skipwork_diagnostics import recover_option

Q='Which city is a capital?\nA) aa\nB) bb\nC) cc\nD) dd'

def fake_send(prompt,stage,budget=64):
    answer='A'
    if stage=='batch_verify':
        items=json.loads(prompt.split('\n',1)[1])
        answer=json.dumps({'answers':{i['id']:'A' for i in items}})
    payload={'model':runtime.MODEL,'prompt':prompt,'stream':False,'options':{'num_predict':budget}}
    return {'call_id':uuid4().hex,'stage':stage,'model':runtime.MODEL,'request_payload':payload,
        'request_sha256':hashlib.sha256(json.dumps(payload).encode()).hexdigest(),'answer':answer,'status':'generated','error':'','input_tokens':10,
        'output_tokens':1,'eval_seconds':.01,'load_seconds':0,'done_reason':'stop','elapsed_seconds':.01}

class TestSkipwork(unittest.TestCase):
    def test_diagnostic_option_recovery_uses_no_gold(self):
        self.assertEqual(recover_option(Q,'B) bb'),'B')
        self.assertEqual(recover_option(Q,'A)'),'A')
        self.assertIsNone(recover_option(Q,'A) bb'))
        self.assertIsNone(recover_option(Q,'A or B'))
        self.assertEqual(list(inspect.signature(recover_option).parameters),['question','response'])
    def test_fresh_balanced_reference_checked_dataset(self):
        with tempfile.TemporaryDirectory() as folder:cal,tasks=data.build(folder)
        self.assertEqual(len(cal),100)
        self.assertEqual(Counter(t['category'] for t in tasks),{c:30 for c in experiment.CATEGORIES})
        keys={data.task_key(t['question']) for t in tasks}
        self.assertEqual(len(keys),150)
        self.assertFalse(keys & {data.task_key(t['question']) for t in cal})
        checks=experiment.check_references(tasks)
        self.assertEqual(len(checks),30)
        self.assertTrue(all(r['matches'] for r in checks))

    def test_duplicate_old_question_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'old.json').write_text(json.dumps({'question':'Which city is the capital of France?'}))
            with self.assertRaises(ValueError):data.build(folder)

    def test_no_gold_in_runtime_interfaces(self):
        self.assertEqual(list(inspect.signature(runtime.execute_block).parameters),['questions','method','profile'])
        self.assertEqual(list(inspect.signature(runtime.preflight).parameters),['question','method','profile'])
        self.assertEqual(list(inspect.signature(runtime.primary).parameters),['question','cap'])

    def test_conservative_profile_gate(self):
        profile={'version':runtime.VERSION,'categories':{'knowledge':{'samples':20,'corrected':0,'harmed':1}}}
        self.assertTrue(runtime.preflight(Q,'selective',profile)['skip_verifier'])
        profile['categories']['knowledge']['samples']=19
        self.assertFalse(runtime.preflight(Q,'selective',profile)['skip_verifier'])
        profile['categories']['knowledge'].update(samples=20,corrected=2)
        self.assertFalse(runtime.preflight(Q,'selective',profile)['skip_verifier'])
        hard=Q.replace('Which city is a capital?','Reason about a rule.')
        profile['categories']['logic']={'samples':20,'corrected':0,'harmed':1}
        self.assertFalse(runtime.preflight(hard,'selective',profile)['skip_verifier'])
        with self.assertRaises(ValueError):runtime.preflight(Q,'selective',{})
        with self.assertRaises(ValueError):runtime.preflight(Q,'invalid',{})

    def test_tool_early_exit_never_calls_model(self):
        with patch.object(runtime,'send') as send:r=runtime.execute_block(['18 * 21'],'early_exit',{})
        send.assert_not_called()
        self.assertEqual(r['rows'][0]['final_prediction'],'378')
        self.assertFalse(r['rows'][0]['verification_executed'])

    def test_tool_failure_is_not_silently_unverified_model_answer(self):
        with patch.object(runtime,'calculate',side_effect=ValueError('test')),patch.object(runtime,'send') as send:
            r=runtime.execute_block(['18 * 21'],'early_exit',{})
        send.assert_not_called()
        self.assertIsNone(r['rows'][0]['final_prediction'])
        self.assertIn('test',r['rows'][0]['tool_error'])

    def test_budget_caps_only_primary_choice(self):
        with patch.object(runtime,'send',side_effect=fake_send) as send:
            runtime.execute_block([Q],'budget_cap',{})
        self.assertEqual(send.call_args_list[0].args[2],8)
        self.assertEqual(send.call_args_list[1].args[1],'verify')

    def test_decisions_precede_all_solving(self):
        events=[];original=runtime.preflight
        def gate(*args):events.append('gate');return original(*args)
        def sender(*args):events.append('solve');return fake_send(*args)
        with patch.object(runtime,'preflight',side_effect=gate),patch.object(runtime,'send',side_effect=sender):
            runtime.execute_block([Q,Q],'baseline',{})
        self.assertEqual(events[:2],['gate','gate'])

    def test_batch_shared_call_and_exact_ids(self):
        with patch.object(runtime,'send',side_effect=fake_send):r=runtime.execute_block([Q,Q],'batch_verify',{})
        self.assertEqual(len(r['calls']),3)
        self.assertEqual(r['rows'][0]['call_ids'][-1],r['rows'][1]['call_ids'][-1])
        self.assertEqual(r['rows'][0]['completion_latency_seconds'],r['rows'][1]['completion_latency_seconds'])
        def invalid(*args):
            a=fake_send(*args)
            if args[1]=='batch_verify':a['answer']='{"answers":{"Q0":"B","extra":"A"}}'
            return a
        with patch.object(runtime,'send',side_effect=invalid):r=runtime.execute_block([Q,Q],'batch_verify',{})
        self.assertTrue(all(r['final_prediction']=='A' and r['verification_prediction'] is None for r in r['rows']))

    def test_malformed_verifier_preserves_primary(self):
        def malformed(*args):
            a=fake_send(*args)
            if args[1]=='verify':a['answer']='A or B'
            return a
        with patch.object(runtime,'send',side_effect=malformed):r=runtime.execute_block([Q],'baseline',{})
        row=experiment.score(r['rows'][0],{'expected':'A','category':'knowledge','task_format':'choice'})
        self.assertTrue(row['verification_format_error'])
        self.assertTrue(row['final_correct'])

    def test_profile_tracks_harm_not_primary_confidence(self):
        p=runtime.make_profile([{'inferred_category':'knowledge','primary_correct':True,'final_correct':False},
                                {'inferred_category':'knowledge','primary_correct':False,'final_correct':True}])
        self.assertEqual(p['categories']['knowledge']['harmed'],1)
        self.assertEqual(p['categories']['knowledge']['corrected'],1)

    def test_batch_plan_limits_and_coverage(self):
        qs=[Q]*9+['19 + 20']
        plan=runtime.batch_plan(qs)
        self.assertEqual(sorted(i for b in plan for i in b),list(range(10)))
        self.assertTrue(all(len(b)<=4 for b in plan))

    def test_offline_pipeline_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            cal,tasks=data.build(folder)
            tiny=[next(t for t in tasks if t['category']==c) for c in experiment.CATEGORIES]
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            original=Path(folder,'keep.csv');original.write_bytes(b'keep')
            with patch.object(experiment,'build',return_value=(cal[:5],tiny)), \
                 patch.object(experiment,'check_references',return_value=[{'matches':True}]), \
                 patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'send',side_effect=fake_send), \
                 patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder)
            self.assertEqual(original.read_bytes(),b'keep')
            self.assertTrue(Path(f'{prefix}_report.md').is_file())
            self.assertTrue(Path(f'{prefix}_comparison.csv').is_file())

    def test_complete_audit_on_offline_artifacts_only(self):
        with tempfile.TemporaryDirectory() as folder:
            cal,tasks=data.build(folder)
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'build',return_value=(cal,tasks)), \
                 patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'send',side_effect=fake_send), \
                 patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder)
                result=skipwork_audit.audit(prefix)
            self.assertEqual(result['evaluation_rows'],750)
            self.assertTrue(result['all_final_scores_recomputed'])
            self.assertTrue(result['no_duplicate_call_accounting'])
            self.assertTrue(Path(f'{prefix}_failure_cases.csv').is_file())

if __name__=='__main__':unittest.main()

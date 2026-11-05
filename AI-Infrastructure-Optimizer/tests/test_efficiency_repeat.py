"""Offline functional tests; all fake measurements stay in temporary directories."""
import contextlib,hashlib,inspect,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from src import efficiency_protocol as protocol
from src import efficiency_runtime as runtime
from src import efficiency_repeat as experiment

Q='Which city is the capital?\nA) Warsaw\nB) Oslo\nC) Paris\nD) Rome'

def fake_send(prompt,stage,budget=64):
    payload={'model':runtime.MODEL,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':budget}}
    answer='A) Warsaw' if stage=='primary' else 'A'
    if stage=='batch_verify':
        items=json.loads(prompt.split('\n',1)[1]);answer='```json\n'+json.dumps({'answers':{t['id']:'A' for t in items}})+'\n```'
    return {'call_id':uuid4().hex,'stage':stage,'model':runtime.MODEL,'request_payload':payload,
        'request_sha256':hashlib.sha256(json.dumps(payload).encode()).hexdigest(),'answer':answer,'status':'generated','error':'',
        'input_tokens':10,'output_tokens':2,'eval_seconds':.01,'prefill_seconds':.02,'total_model_seconds':.04,
        'load_seconds':.001,'done_reason':'stop','format_compliant':False,'parse_mode':'unresolved','elapsed_seconds':.05}

class TestEfficiencyProtocol(unittest.TestCase):
    def test_content_and_format_are_independent(self):
        for text in ('A) Warsaw','A)','Warsaw','FINAL_ANSWER: A','```text\nA\n```','a'):
            p=protocol.extract(Q,text,'choice');self.assertEqual(p['prediction'],'A',text);self.assertFalse(p['format_compliant'])
        self.assertTrue(protocol.extract(Q,'A','choice')['format_compliant'])

    def test_conflict_and_ambiguous_text_are_not_gold_rescued(self):
        for text in ('A) Oslo','A or B','The answer is A','FINAL_ANSWER: A\nFINAL_ANSWER: B','A\nB','A) Warsaw because it is correct'):
            self.assertIsNone(protocol.extract(Q,text,'choice')['prediction'],text)
        self.assertEqual(list(inspect.signature(protocol.extract).parameters),['question','response','kind'])

    def test_integer_normalization_does_not_accept_expressions(self):
        self.assertEqual(protocol.extract('1+1','+002','integer')['prediction'],'2')
        self.assertEqual(protocol.extract('1+1','FINAL_ANSWER: -2','integer')['prediction'],'-2')
        for text in ('2 or3','2.0','1+1','2,000'):self.assertIsNone(protocol.extract('1+1',text,'integer')['prediction'])

    def test_batch_fence_readable_but_noncompliant(self):
        text='{"answers":{"Q0":"A","Q1":"B"}}'
        plain=protocol.extract_batch(text,[Q,Q],['choice','choice'])
        self.assertTrue(plain['format_compliant'])
        fenced=protocol.extract_batch('```json\n'+text+'\n```',[Q,Q],['choice','choice'])
        self.assertEqual([r['prediction'] for r in fenced['items']],['A','B'])
        self.assertFalse(fenced['format_compliant'])

    def test_batch_rejects_duplicate_extra_missing_ids_and_boolean(self):
        for raw in ('{"answers":{"Q0":"A","Q0":"B"}}','{"answers":{"Q0":"A"},"answers":{"Q0":"B"}}',
                    '{"answers":{"Q0":"A","Q1":"B"}}','{"answers":{}}','{"answers":{"Q0":true}}'):
            r=protocol.extract_batch(raw,[Q],['choice'])
            self.assertIsNone(r['items'][0]['prediction'],raw)
        with self.assertRaises(ValueError):protocol.extract_batch('{}',[],[])

class TestEfficiencyExecution(unittest.TestCase):
    def test_preflight_interface_and_same_policy(self):
        self.assertEqual(list(inspect.signature(runtime.execute_block).parameters),['questions','method','profile'])
        profile={'version':'skipwork-v1','categories':{'knowledge':{'samples':20,'corrected':0,'harmed':1}}}
        with patch.object(runtime,'send',side_effect=fake_send):block=runtime.execute_block([Q],'selective',profile)
        self.assertEqual(len(block['calls']),1)
        self.assertEqual(block['rows'][0]['final_prediction'],'A')
        self.assertFalse(block['rows'][0]['final_format_compliant'])

    def test_batch_corrected_parser_adopts_actual_verification(self):
        def send(*args):
            a=fake_send(*args)
            if args[1]=='batch_verify':a['answer']='```json\n{"answers":{"Q0":"B","Q1":"C"}}\n```'
            return a
        with patch.object(runtime,'send',side_effect=send):block=runtime.execute_block([Q,Q],'batch_verify',{})
        self.assertEqual(len(block['calls']),3)
        self.assertEqual([r['final_prediction'] for r in block['rows']],['B','C'])
        self.assertFalse(any(r['final_format_compliant'] for r in block['rows']))
        self.assertEqual(block['rows'][0]['call_ids'][-1],block['rows'][1]['call_ids'][-1])

    def test_invalid_verifier_preserves_primary_without_retry(self):
        def send(*args):
            a=fake_send(*args)
            if args[1]=='verify':a['answer']='B or C'
            return a
        with patch.object(runtime,'send',side_effect=send):block=runtime.execute_block([Q],'baseline',{})
        self.assertEqual(block['rows'][0]['final_prediction'],'A')
        self.assertTrue(block['rows'][0]['verification_preserved_primary'])
        self.assertEqual(len(block['calls']),2)

    def test_missing_measurements_remain_unmeasured(self):
        self.assertIsNone(experiment.total([{'v':None}],'v'))
        self.assertEqual(experiment.total([{'v':0}],'v'),0)
        self.assertIsNone(experiment.total([],'v'))

    def test_aggregation_mean_range_and_sample_std(self):
        rows=[{'method':m,'repetition':rep,'accuracy':v,'cost':None,'cost_status':'unmeasured'}
              for m in runtime.METHODS for rep,v in [(1,.5),(2,.7),(3,.6)]]
        a=experiment.aggregate(rows)[0]
        self.assertEqual(a['accuracy_mean'],.6);self.assertEqual(a['accuracy_min'],.5);self.assertEqual(a['accuracy_max'],.7)
        self.assertAlmostEqual(a['accuracy_sample_std'],.1);self.assertIsNone(a['cost_mean'])

    def test_full_two_repeat_offline_pipeline_preserves_data(self):
        with tempfile.TemporaryDirectory() as folder:
            old=Path(folder,'keep.csv');old.write_text('keep\n')
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'send',side_effect=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder,repeats=2)
            self.assertEqual(old.read_text(),'keep\n')
            summaries=experiment.read_csv(f'{prefix}_repeat_summary.csv')
            self.assertEqual(len(summaries),10)
            self.assertEqual(len(experiment.read_csv(f'{prefix}_evaluation.csv')),1500)
            self.assertTrue(Path(f'{prefix}_report.md').is_file())
            self.assertTrue(Path(f'{prefix}_warmup.json').is_file())

if __name__=='__main__':unittest.main()

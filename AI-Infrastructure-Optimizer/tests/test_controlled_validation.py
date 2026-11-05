import contextlib,copy,io,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from collections import Counter
from test_rcav import fake_send
from src import controlled_dataset as data
from src import controlled_runtime as runtime
from src import controlled_validation as experiment
from src.rcav_runtime import LocalClient
from src.solver300_experiment import check_references

class DummySampler:
    def __init__(self,snapshot):pass
    def start(self):pass
    def stop(self,snapshot):return {'sampled_max_rss_mib':1.,'rss_samples':[[1,1]],'rss_sample_count':1,'rss_sampling_interval_seconds':.1,'rss_sampling_errors':[]}

def fake_reset(client,models):return {'target_model':client.model,'reset_observed':True,'events':[],'error':''},fake_send(client,'Return only A.','controlled_warmup')

class TestControlled(unittest.TestCase):
    def test_fresh300_balanced_unique_reference_checks(self):
        with tempfile.TemporaryDirectory() as folder:tasks=data.build(folder)
        self.assertEqual(Counter(t['category'] for t in tasks),{c:60 for c in ('arithmetic','knowledge','logic','coding','comprehension')})
        self.assertEqual(len({data.task_key(t['question']) for t in tasks}),300)
        self.assertEqual(len(check_references(tasks)),60)

    def test_rotations_balance_first_four_repeats(self):
        datasets={'x':[{'task_id':str(i)} for i in range(20)]}
        plan=experiment.make_schedule(datasets,['model'],5)
        for block in ('x:B000','x:B001'):
            for method in runtime.METHODS:
                positions=[p['position'] for p in plan if p['block_key']==block and p['method']==method and p['repetition']<=4]
                self.assertEqual(sorted(positions),[1,2,3,4])

    def test_runtime_never_generates_primary_or_mutates_cache(self):
        p={'question':'Which?\nA) a\nB) b\nC) c\nD) d','task_id':'t','answer':'B','prediction':'B','source':'model','format_compliant':True,'cache_entry_sha256':'h'}
        frozen=copy.deepcopy(p);client=LocalClient(experiment.LARGE)
        with patch.object(client,'primary',side_effect=AssertionError('Primary regeneration forbidden')),patch.object(LocalClient,'send',new=fake_send):
            b=runtime.execute_block([p],'baseline',client,{})
        self.assertEqual(p,frozen);self.assertEqual(len(b['calls']),1)
        self.assertIn('Candidate response: "B"',b['calls'][0]['request_payload']['prompt'])

    def test_cache_control_observes_unload_and_logs_warmup(self):
        client=LocalClient(experiment.LARGE)
        def http(path,payload=None):
            return {'models':[]} if path=='/api/ps' else {'done':True,'done_reason':'unload'}
        with patch.object(runtime,'http_json',side_effect=http),patch.object(LocalClient,'send',new=fake_send):
            control,warm=runtime.reset_and_warm(client,[experiment.LARGE])
        self.assertTrue(control['reset_observed']);self.assertEqual(warm['stage'],'controlled_warmup')

    def test_two_models_five_repeats_pipeline_temp_only(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'MemorySampler',DummySampler),patch.object(runtime,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(runtime,'http_json',return_value={'models':[{'name':experiment.LARGE},{'name':experiment.SMALL}]}), \
                 patch.object(LocalClient,'send',new=fake_send),contextlib.redirect_stdout(io.StringIO()):
                prefix=experiment.run(folder)
            meta=json.loads(Path(f'{prefix}_metadata.json').read_text())
            self.assertEqual(meta['outcome_rows'],18000);self.assertEqual(meta['shared_primary_model_calls'],720)
            cache=json.loads(Path(f'{prefix}_primary_cache.json').read_text())
            self.assertEqual(len(cache),900);self.assertTrue(all('expected' not in r for r in cache))
            calls=experiment.read_csv(f'{prefix}_calls.csv');self.assertTrue(all(a['stage']=='verify' for a in calls))

if __name__=='__main__':unittest.main()

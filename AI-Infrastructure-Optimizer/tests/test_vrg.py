import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src import vrg_policy as p
from src import vrg_experiment as exp
from src import vrg_dataset as dataset
from src.solver300_dataset import task_key
from src.solver300_experiment import check_references
from src.rcav_runtime import LocalClient
from test_rcav import fake_send
from test_controlled_validation import DummySampler,fake_reset

Q='What does this Python program output?\n```python\nprint(2+2)\n```\nA) 4\nB) 3\nC) 2\nD) Error'
def primary(answer='A',source='model'):
    return dict(p.parse(Q,answer),answer=answer,source=source,question=Q,task_id='id')
def revision(answer):return dict(p.parse(Q,answer),answer=answer)

class FakeClient:
    model='fake'
    def __init__(self,answer):self.answer=answer;self.count=0
    def verify(self,q,a):
        self.count+=1;return {'call_id':str(self.count),'answer':self.answer,'status':'generated'},{}
    def primary(self,q):raise AssertionError('Frozen primary must never regenerate')

class TestVRG(unittest.TestCase):
    def test_content_parsing_is_gold_independent_and_format_separate(self):
        self.assertEqual(p.parse(Q,'A.')['prediction'],'A')
        self.assertFalse(p.parse(Q,'A.')['format_compliant'])
        self.assertIsNone(p.parse(Q,'A or B')['prediction'])

    def test_heuristic_protects_resolved_and_tool_but_recovers_unknown(self):
        self.assertFalse(p.heuristic(Q,primary(),revision('B'))[0])
        self.assertFalse(p.heuristic(Q,primary(source='tool'),revision('B'))[0])
        self.assertTrue(p.heuristic(Q,primary('unclear'),revision('B'))[0])
        self.assertFalse(p.heuristic(Q,primary(),revision('unclear'))[0])

    def test_reference_label_is_rejected_by_both_guard_paths(self):
        a=primary();a['expected']='A'
        with self.assertRaises(ValueError):p.heuristic(Q,a,revision('B'))
        with self.assertRaises(ValueError):p.attributes(Q,a,revision('B'))

    def test_actual_revision_is_charged_even_when_guard_rejects(self):
        client=FakeClient('B');r=exp.execute([primary()],'heuristic',client,{},None,['id'])
        self.assertEqual(client.count,1);self.assertEqual(len(r['calls']),1)
        self.assertEqual(r['rows'][0]['final_prediction'],'A')
        self.assertFalse(r['rows'][0]['accepted_revision'])

    def test_always_unresolved_is_returned_none_never_calls_and_oracle_is_not_executable(self):
        c=FakeClient('unclear');r=exp.execute([primary()],'always',c,{},None,['id'])
        self.assertIsNone(r['rows'][0]['final_prediction'])
        self.assertEqual(exp.execute([primary()],'none',c,{},None,['id'])['calls'],[])
        with self.assertRaises(ValueError):exp.execute([primary()],'oracle',c,{},None,['id'])

    def test_learned_uses_fitted_weights_and_validity_guards(self):
        rows=[]
        for i in range(12):rows.append({'question':Q,'primary':primary('B'),'revision':revision('A'),'label':0})
        profile=p.train(rows,rows[:4]);self.assertTrue(p.learned(Q,primary('B'),revision('A'),profile)[0])
        self.assertFalse(p.learned(Q,primary('B'),revision('unclear'),profile)[0])
        self.assertFalse(p.learned(Q,primary('B',source='tool'),revision('A'),profile)[0])

    def test_dataset_disjoint_balanced_and_code_gold_independently_executed(self):
        va,te=dataset.build(dataset.ROOT/'data');self.assertEqual((len(va),len(te)),(100,300))
        self.assertFalse({task_key(t['question']) for t in va}&{task_key(t['question']) for t in te})
        self.assertEqual({c:sum(t['category']==c for t in te) for c in set(t['category'] for t in te)},dict.fromkeys(('arithmetic','knowledge','logic','coding','comprehension'),60))
        check_references(va+te)

    def test_complete_mock_grid_stays_in_temporary_directory(self):
        snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(exp,'check_ollama'),patch('src.controlled_validation.process_snapshot',return_value=snapshot), \
                 patch.object(exp.controls,'MemorySampler',DummySampler),patch.object(exp.controls,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(exp.controls,'http_json',return_value={'models':[{'name':m} for m in (exp.LARGE,exp.SMALL)]}), \
                 patch.object(LocalClient,'send',new=fake_send),contextlib.redirect_stdout(io.StringIO()):
                prefix=exp.run(folder)
            meta=json.loads(Path(f'{prefix}_metadata.json').read_text())
            self.assertEqual(meta['raw_rows'],15000)
            self.assertEqual(meta['primary_calls'],480)
            self.assertEqual(meta['calibration_calls'],360)
            self.assertTrue(Path(f'{prefix}_profiles.json').is_file())

if __name__=='__main__':unittest.main()

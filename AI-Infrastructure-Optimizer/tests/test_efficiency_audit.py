import contextlib,csv,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import hashlib
from src import efficiency_runtime as runtime
from src import efficiency_repeat as experiment
from src import efficiency_audit as auditing

def fake_send(prompt,stage,budget=64):
    answer='A'
    if stage=='batch_verify':
        items=json.loads(prompt.split('\n',1)[1])
        answer='```json\n'+json.dumps({'answers':{i['id']:'A' for i in items}})+'\n```'
    payload={'model':runtime.MODEL,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':budget}}
    return {'call_id':uuid4().hex,'stage':stage,'model':runtime.MODEL,'request_payload':payload,
        'request_sha256':hashlib.sha256(json.dumps(payload).encode()).hexdigest(),'answer':answer,'status':'generated','error':'',
        'input_tokens':10,'output_tokens':2,'eval_seconds':.01,'prefill_seconds':.02,'total_model_seconds':.04,
        'load_seconds':.001,'done_reason':'stop','format_compliant':False,'parse_mode':'unresolved','elapsed_seconds':.05}

class TestEfficiencyAudit(unittest.TestCase):
    def test_full_offline_reconstruction_and_tampered_gold_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'send',side_effect=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder,repeats=2)
                result=auditing.audit(prefix)
            self.assertEqual(result['evaluated_rows'],1500)
            self.assertTrue(result['reconstructed_summaries_aggregates_pairs'])
            path=Path(f'{prefix}_evaluation.csv');rows=experiment.read_csv(path)
            rows[0]['expected']='deliberate corruption in temporary unit test'
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(AssertionError):auditing.audit(prefix)

if __name__=='__main__':unittest.main()

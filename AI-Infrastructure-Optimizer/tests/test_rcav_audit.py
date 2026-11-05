import contextlib,csv,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_rcav import fake_send
from src import rcav_runtime as runtime
from src import rcav_experiment as experiment
from src import rcav_audit

class TestRCAVAudit(unittest.TestCase):
    def test_complete_offline_reconstruction_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime.LocalClient,'send',new=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO(json.dumps({'models':[{'name':experiment.LARGE}]}))
                prefix=experiment.run(folder,repeats=2)
                result=rcav_audit.audit(prefix)
            self.assertEqual(result['evaluation_rows'],1500)
            self.assertTrue(result['all_decisions_reconstructed_from_preverification_information'])
            self.assertEqual(result['old_or_split_task_overlap'],0)
            path=Path(f'{prefix}_evaluation.csv');rows=experiment.read_csv(path);rows[0]['expected']='corrupted test only'
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(AssertionError):rcav_audit.audit(prefix)

if __name__=='__main__':unittest.main()

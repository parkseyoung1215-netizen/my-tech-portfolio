import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src import vrg_experiment as experiment
from src.vrg_analysis import analyze
from src.rcav_runtime import LocalClient
from test_rcav import fake_send
from test_controlled_validation import DummySampler,fake_reset

class TestVRGAudit(unittest.TestCase):
    def test_all_arms_and_oracle_audit_on_mock_outputs_only_in_tempdir(self):
        snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(experiment,'check_ollama'),patch('src.controlled_validation.process_snapshot',return_value=snapshot), \
                 patch.object(experiment.controls,'MemorySampler',DummySampler),patch.object(experiment.controls,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(experiment.controls,'http_json',return_value={'models':[{'name':m} for m in (experiment.LARGE,experiment.SMALL)]}), \
                 patch.object(LocalClient,'send',new=fake_send),contextlib.redirect_stdout(io.StringIO()):
                prefix=experiment.run(folder);out=analyze(prefix)
            audit=json.loads((out/'VRG_AUDIT.json').read_text())
            self.assertEqual(audit['actual_rows'],15000)
            self.assertEqual(audit['oracle_rows_analysis_only'],3000)
            self.assertTrue(audit['all_acceptance_decisions_reconstructed'])
            self.assertTrue((out/'VRG_RESULTS.csv').exists())

if __name__=='__main__':unittest.main()

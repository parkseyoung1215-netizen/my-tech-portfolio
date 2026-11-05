import unittest,contextlib,io,tempfile
from pathlib import Path
from unittest.mock import patch
from src.vvp_analysis import auc
from src import vvp_analysis as analysis
from src import vvp_experiment as experiment
from src import vvp_model as model
from src import vvp_runtime as runtime
from src.rcav_runtime import LocalClient
from test_vvp import fake_predict,fake_fit
from test_rcav import fake_send
from test_controlled_validation import DummySampler,fake_reset

class TestVVPAnalysis(unittest.TestCase):
    def test_auc_ties_and_no_positive_labels(self):
        self.assertEqual(auc([.9,.1],[True,False]),1)
        self.assertEqual(auc([.1,.9],[True,False]),0)
        self.assertEqual(auc([.5,.5],[True,False]),.5)
        self.assertIsNone(auc([.2,.3],[False,False]))

    def test_full_offline_grid_audit_in_temporary_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch('src.controlled_validation.process_snapshot',return_value=snapshot), \
                 patch.object(experiment.controls,'MemorySampler',DummySampler),patch.object(experiment.controls,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(experiment.controls,'http_json',return_value={'models':[{'name':experiment.LARGE}]}), \
                 patch.object(LocalClient,'send',new=fake_send),patch.object(model,'train',side_effect=fake_fit),patch.object(model,'predict',side_effect=fake_predict), \
                 patch.object(runtime,'predict',side_effect=fake_predict),patch.object(runtime,'estimate_cpu',return_value=.03),contextlib.redirect_stdout(io.StringIO()):
                prefix=experiment.run(folder)
                result=analysis.analyze(prefix)
            self.assertEqual(result['outcome_rows'],54000)
            self.assertTrue(result['all_allocations_frozen_and_reconstructed'])
            self.assertTrue(Path(f'{prefix}_verdict.csv').is_file())

if __name__=='__main__':unittest.main()

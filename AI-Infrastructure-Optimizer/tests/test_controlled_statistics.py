import contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_controlled_validation import DummySampler,fake_reset
from test_rcav import fake_send
from src import controlled_statistics as stats
from src import controlled_analysis as analysis
from src import controlled_validation as experiment
from src import controlled_runtime as runtime
from src.rcav_runtime import LocalClient

class TestControlledStatistics(unittest.TestCase):
    def test_exact_bounds_include_unseen_loss_risk(self):
        self.assertAlmostEqual(stats.cp_lower(0,300),0)
        self.assertGreater(stats.cp_upper(0,300),.01)
        result=stats.accuracy_comparison([[0]*5 for i in range(300)],draws=100)
        self.assertLess(result['conservative_exact_lower95'],-.01)
        self.assertEqual(result['paired_bootstrap_low'],0)

    def test_two_axis_paired_cost_and_missing_values(self):
        result=stats.paired_cost([[-1]*5,[-2]*5],draws=100)
        self.assertEqual(result['delta'],-3)
        self.assertLess(result['ci95_high'],0)
        self.assertIsNone(stats.paired_cost([[None]*5])['delta'])

    def test_full_offline_analysis_one_model_five_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'MemorySampler',DummySampler),patch.object(runtime,'reset_and_warm',side_effect=fake_reset), \
                 patch.object(runtime,'http_json',return_value={'models':[{'name':experiment.LARGE}]}), \
                 patch.object(LocalClient,'send',new=fake_send),contextlib.redirect_stdout(io.StringIO()):
                prefix=experiment.run(folder)
                result=analysis.analyze(prefix)
            self.assertEqual(result['outcome_rows'],9000)
            self.assertEqual(result['primary_regeneration_during_verification'],0)
            self.assertTrue(Path(f'{prefix}_comparisons.csv').is_file())
            self.assertTrue(Path(f'{prefix}_verdict.csv').is_file())

if __name__=='__main__':unittest.main()

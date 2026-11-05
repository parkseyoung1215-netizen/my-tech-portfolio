import contextlib,csv,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_rcav import fake_send
from src.rcav_runtime import LocalClient
from src import rcav_guard_experiment as experiment
from src import rcav_guard_analysis as analysis

class TestGuardAnalysis(unittest.TestCase):
    def test_cluster_interval_and_exact_pair_test(self):
        self.assertEqual(analysis.paired_interval({'a':[0,0],'b':[0,0]}),(0,0))
        self.assertEqual(analysis.paired_interval({'a':[1,1],'b':[1,1]}),(1,1))
        self.assertAlmostEqual(analysis.exact_mcnemar(1,4),.375)
        self.assertEqual(analysis.exact_mcnemar(0,0),1)
        self.assertAlmostEqual(analysis.exact_mcnemar(0,4),.125)

    def test_offline_reconstruction_and_tampered_gold_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(LocalClient,'send',new=fake_send),patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                prefix=experiment.run(folder)
                result=analysis.analyze(prefix)
            self.assertEqual(result['evaluation_rows'],2700)
            self.assertEqual(result['guard_training_overlap_either_evaluation'],0)
            self.assertTrue(Path(f'{prefix}_uncertainty.csv').exists())
            path=Path(f'{prefix}_evaluation.csv');rows=experiment.read_csv(path);rows[0]['expected']='corruption in temporary test only'
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(AssertionError):analysis.analyze(prefix)

if __name__=='__main__':unittest.main()

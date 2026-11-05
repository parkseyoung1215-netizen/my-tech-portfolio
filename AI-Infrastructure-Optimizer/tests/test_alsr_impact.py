import inspect,json,tempfile,unittest
from pathlib import Path
from src import alsr_impact_data as data,alsr_impact_policy as policy

class TestLayerImpact(unittest.TestCase):
    def test_fresh_split_sizes_disjoint_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            splits=data.build(folder);self.assertEqual({k:len(v) for k,v in splits.items()},{'training':32,'calibration':16,'evaluation':64})
            ids=[t['task_id'] for ts in splits.values() for t in ts];self.assertEqual(len(ids),len(set(ids)))
            Path(folder,'old.json').write_text(json.dumps(splits))
            with self.assertRaises(ValueError):data.build(folder)

    def test_gold_does_not_enter_accept_interface(self):
        self.assertEqual(list(inspect.signature(policy.accept).parameters),['features','profile'])
        p={'candidate':7,'threshold':.2,'ridge':{'7':{'mean':[0],'scale':[1],'weights':[.1,0]}}}
        self.assertEqual(policy.accept([3],p),(True,.1))

    def test_validation_rejects_flip_despite_more_skips(self):
        threshold,trials=policy.select_threshold([{'prediction':.02,'kl':.001,'top1_changed':0},{'prediction':.07,'kl':.001,'top1_changed':1}])
        self.assertEqual(threshold,.025);self.assertFalse(next(t for t in trials if t['threshold']==.1)['eligible'])

    def test_auc_ties_and_missing_class(self):
        self.assertEqual(policy.auc([0,1],[.2,.7]),1)
        self.assertEqual(policy.auc([0,1],[1,1]),.5)
        self.assertIsNone(policy.auc([1,1],[.2,.7]))

class TestDistributions(unittest.TestCase):
    def test_full_vocabulary_kl_and_signed_margin(self):
        try:import numpy
        except ImportError:self.skipTest('NumPy tests run separately in existing MLX environment')
        effect=policy.distribution_effect([4,1,0],[4,1,0]);self.assertAlmostEqual(effect['kl'],0);self.assertEqual(effect['top1_changed'],0)
        changed=policy.distribution_effect([4,1,0],[0,4,1]);self.assertGreater(changed['kl'],0);self.assertEqual(changed['top1_changed'],1);self.assertLess(changed['full_top1_skip_margin'],0)

if __name__=='__main__':unittest.main()

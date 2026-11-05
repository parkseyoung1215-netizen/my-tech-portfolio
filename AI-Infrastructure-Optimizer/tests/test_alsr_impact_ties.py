import unittest
from src.alsr_impact_ties import canonical_top,correct

class TestGreedyTies(unittest.TestCase):
    def test_direct_full_distribution_tie_uses_greedy(self):
        try:
            import numpy
        except ImportError:self.skipTest('NumPy test run separately in isolated environment')
        from src.alsr_impact_ties import distribution_effect
        result=distribution_effect([2.,2.,0.],[3.,2.,0.]);self.assertEqual(result['full_top1'],0);self.assertEqual(result['top1_changed'],0)

    def test_min_id_among_max_ties_not_partition_order(self):
        self.assertEqual(canonical_top({'top_ids':[8,2,9],'top_probabilities':[.4,.4,.2]}),2)
    def test_saturated_saved_topk_requires_remeasurement(self):
        with self.assertRaises(ValueError):canonical_top({'top_ids':[8,2],'top_probabilities':[.5,.5]})
    def test_flip_correction_keeps_legacy_and_kl(self):
        context={'task_id':'t','position':0,'baseline_top1':2,'baseline_distribution':{'top_ids':[8,2,9],'top_probabilities':[.4,.4,.2]}}
        row={'task_id':'t','position':0,'full_top1':8,'skip_top1':2,'top1_changed':1,'full_top1_skip_margin':-1,'kl':.2,'skip_distribution':{'top_ids':[2,8,9],'top_probabilities':[.6,.3,.1]}}
        r=correct([row],[context])[0];self.assertEqual(r['top1_changed'],0);self.assertEqual(r['legacy_top1_changed'],1);self.assertEqual(r['kl'],.2);self.assertGreater(r['full_top1_skip_margin'],0)

import tempfile,unittest
from pathlib import Path

class TestImpactStatistics(unittest.TestCase):
    def test_json_boolean_is_not_integer_one(self):
        from src.alsr_impact_scoring import grade
        t={'score_kind':'json','expected_object':{'enabled':True,'label':'u'}}
        self.assertFalse(grade(t,'{"enabled":1,"label":"u"}')['correct'])
        self.assertTrue(grade(t,'{"enabled":true,"label":"u"}')['correct'])

    def test_rank_ties_and_cluster_auc_direction(self):
        try:
            from src.alsr_impact_analysis import rank,paired_signal
        except ImportError:self.skipTest('Run numeric audit tests in the isolated NumPy environment')
        self.assertEqual(rank([3,1,1]).tolist(),[2,.5,.5])
        rows=[dict(task_id=str(i),kl=.1 if i%2 else 0,top1_changed=i%2) for i in range(8)]
        result=paired_signal(rows,[i%2 for i in range(8)],[.5]*8,draws=100)
        self.assertEqual(result['auroc_delta'],.5);self.assertLess(result['risk_mae_delta'],0)

    def test_zero_variance_is_not_reported_as_correlation(self):
        try:from src.alsr_impact_analysis import correlation
        except ImportError:self.skipTest('Run numeric audit tests in the isolated NumPy environment')
        self.assertIsNone(correlation([1,1],[1,2]))

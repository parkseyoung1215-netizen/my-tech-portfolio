import unittest
from src.alsr_analysis import paired_ratio

class TestALSRStatistics(unittest.TestCase):
    def test_paired_identity_ratio_preserves_question_repetition_clusters(self):
        x={'q1':{1:1.,2:2.,3:3.},'q2':{1:4.,2:5.,3:6.}}
        r=paired_ratio(x,x,draws=200)
        self.assertEqual((r['ratio'],r['ci95_low'],r['ci95_high']),(1.,1.,1.))

    def test_constant_double_workload_ratio_is_two(self):
        base={'q1':{1:1.,2:2.},'q2':{1:8.,2:5.}}
        target={q:{r:2*v for r,v in rr.items()} for q,rr in base.items()};result=paired_ratio(target,base,draws=200)
        self.assertEqual((result['ratio'],result['ci95_low'],result['ci95_high']),(2.,2.,2.))

if __name__=='__main__':unittest.main()

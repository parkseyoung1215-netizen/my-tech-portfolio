import unittest
from src.controlled_telemetry import duration_problem

class TestControlledTelemetry(unittest.TestCase):
    def test_impossible_duration_is_not_accepted_or_rescaled(self):
        a={'prefill_seconds':898.,'eval_seconds':.03,'total_model_seconds':.4,'elapsed_seconds':.41}
        self.assertEqual(duration_problem(a),'reported_components_exceed_reported_total')
        self.assertEqual(a['prefill_seconds'],898.)
        a.update(prefill_seconds=.2,eval_seconds=.1)
        self.assertEqual(duration_problem(a),'')
        a['elapsed_seconds']=.1
        self.assertEqual(duration_problem(a),'reported_total_exceeds_client_elapsed')

if __name__=='__main__':unittest.main()

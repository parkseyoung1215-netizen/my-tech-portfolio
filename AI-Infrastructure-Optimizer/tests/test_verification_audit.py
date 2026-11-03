import unittest

from src import verification_audit as audit
from src import verify_routing200 as verify


class TestVerificationAudit(unittest.TestCase):
    def row(self, strategy, question, status, correct, attempts):
        return {"strategy": strategy, "question": question, "bucket": "addition_small",
                "model": verify.router.LARGE if status == "answered" else "",
                "status": status, "correct": correct, "attempts": attempts,
                "elapsed_seconds": 0.1, "cpu_seconds": 0.05, "cpu_percent": 50,
                "boundary_max_rss_mib": 100, "audit_mismatches": 0}

    def test_selected_format_failure_is_not_policy_refusal(self):
        rows = [self.row("fixed", "2 + 3", "rejected", False,
                         [{"model": verify.router.LARGE, "status": "extraction_failed"}])]
        summaries = verify.aggregate(rows)
        audit.enrich(summaries, [], rows)
        overall = next(s for s in summaries if s["bucket"] == "all")
        self.assertEqual(overall["primary_selection_ratios"][verify.router.LARGE], 1)
        self.assertEqual(overall["policy_refusals"], 0)
        self.assertEqual(overall["failed_selected_requests"], 1)

    def test_equal_correct_totals_can_hide_question_regression(self):
        rows = [self.row(strategy, question, "answered", correct, [])
                for strategy, question, correct in [
                    ("fixed", "a", True), ("fixed", "b", False),
                    ("improved", "a", False), ("improved", "b", True),
                ]]
        pairs = [p for p in verify.paired_comparisons(rows) if p["baseline"] == "fixed"]
        audit.enrich([], pairs, rows)
        self.assertEqual(pairs[0]["correct_to_wrong"], 1)
        self.assertEqual(pairs[0]["wrong_to_correct"], 1)
        self.assertEqual(pairs[0]["baseline_common_cpu_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()

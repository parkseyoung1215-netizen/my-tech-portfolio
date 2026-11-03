import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import mixed_reliability_audit as audit


class TestReliabilityAudit(unittest.TestCase):
    def test_content_parser_accepts_only_unambiguous_whole_answers(self):
        cases = [("C", "choice", "C"), (" c\n", "choice", "C"),
                 ("FINAL_ANSWER: B", "choice", "B"),
                 ("+0012", "integer", "12"), ("FINAL_ANSWER: -12", "integer", "-12"),
                 ("I think A or B", "choice", None), ("A because it fits", "choice", None),
                 ("FINAL_ANSWER: A\nFINAL_ANSWER: B", "choice", None),
                 ("3,904", "integer", None), ("12", "choice", None), ("A", "integer", None)]
        for response, kind, expected in cases:
            with self.subTest(response=response, kind=kind):
                self.assertEqual(audit.content_extract(response, kind)[0], expected)

    def test_parser_has_no_expected_answer_input(self):
        self.assertEqual(list(inspect.signature(audit.content_extract).parameters), ["response", "task_format"])

    def test_content_wrong_is_not_rescued_by_gold(self):
        row = {"strategy": "fixed", "category": "knowledge", "question": "Q", "expected": "A",
               "answer": "B", "predicted": "", "status": "rejected", "correct": False,
               "task_format": "choice", "attempts": [{"model": "m", "status": "extraction_failed", "answer": "B"}],
               "model": "m", "source": "model", "tool_calls": 0, "runtime_details": {},
               "reason": "format", "general_adapter": True, "elapsed_seconds": 0.1}
        scored = audit.score_row(row)
        self.assertEqual(scored["content_prediction"], "B")
        self.assertFalse(scored["content_correct"])
        self.assertEqual(scored["failure_kind"], "format_and_content_error")

    def test_withheld_attempt_is_not_counted_as_a_delivered_answer(self):
        row = {"strategy": "coverage", "category": "arithmetic", "question": "2+3", "expected": "5",
               "answer": "", "predicted": "", "status": "rejected", "correct": False,
               "task_format": "integer", "attempts": [{"model": "m", "status": "validation_failed", "answer": "FINAL_ANSWER: 4"}],
               "model": "", "source": "model", "tool_calls": 0, "runtime_details": {},
               "reason": "mismatch", "general_adapter": False, "elapsed_seconds": 0.1}
        scored = audit.score_row(row)
        self.assertIsNone(scored["content_prediction"])
        self.assertEqual(scored["failure_kind"], "withheld_validation_failure")

    def test_token_missing_values_remain_unmeasured(self):
        rows = audit.read_rows(Path(f"{audit.SOURCE}_evaluation.csv"))
        scored = [audit.score_row(r) for r in rows]
        summaries = {s["strategy"]: s for s in audit.summary(rows, scored) if s["category"] == "all"}
        self.assertEqual(summaries["fixed"]["output_tokens_total"], 486)
        self.assertEqual(summaries["tool"]["output_tokens_total"], 300)
        self.assertEqual(summaries["fixed"]["input_token_recorded_calls"], 20)
        self.assertIsNone(summaries["fixed"]["input_tokens_total"])
        self.assertIsNone(summaries["tool"]["total_tokens"])

    def test_original_and_content_scores_are_separate(self):
        rows = audit.read_rows(Path(f"{audit.SOURCE}_evaluation.csv"))
        scored = [audit.score_row(r) for r in rows]
        summaries = {s["strategy"]: s for s in audit.summary(rows, scored) if s["category"] == "all"}
        self.assertEqual([summaries[s]["original_correct"] for s in audit.STRATEGIES], [34, 31, 38, 45])
        self.assertEqual([summaries[s]["content_correct"] for s in audit.STRATEGIES], [72, 69, 76, 83])
        self.assertTrue(all(s["format_only_errors"] == 38 for s in summaries.values()))

    def test_complete_audit_has_no_network_or_inference_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("urllib.request.urlopen", side_effect=AssertionError("No network allowed")), \
             contextlib.redirect_stdout(io.StringIO()):
            prefix = audit.run(output_dir=folder)
            self.assertTrue(Path(f"{prefix}_report.md").is_file())
            self.assertTrue(Path(f"{prefix}_per_question.csv").is_file())
            metadata = json.loads(Path(f"{prefix}_metadata.json").read_text())
            self.assertFalse(metadata["inference_rerun"])


if __name__ == "__main__":
    unittest.main()

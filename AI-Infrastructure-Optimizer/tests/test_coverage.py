import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import coverage_runner as runner
from src import coverage_experiment as experiment
from src.coverage_policy import CoveragePolicy
from src.request_verifier import verify_response
from src.verify_routing200 import PROFILE, problem_key


def policy():
    return CoveragePolicy(json.loads(PROFILE.read_text()))


class TestCoverage(unittest.TestCase):
    def test_request_verifier_rejects_math_errors_without_labels(self):
        self.assertTrue(verify_response("2 + 19", "FINAL_ANSWER: 21")["valid"])
        self.assertEqual(verify_response("2 + 19", "FINAL_ANSWER: 20")["reason"], "arithmetic_mismatch")
        self.assertTrue(verify_response("(123 * 42) - 13", "FINAL_ANSWER: 5153")["valid"])
        self.assertTrue(verify_response("21 * 34", "FINAL_ANSWER: 714")["valid"])

    def test_strict_extraction_and_unsupported_requests(self):
        for text in ["21", "FINAL_ANSWER: 21 extra", "FINAL_ANSWER: 21\nFINAL_ANSWER: 20"]:
            self.assertEqual(verify_response("2 + 19", text)["reason"], "extraction_failed")
        self.assertEqual(verify_response("__import__('os')", "FINAL_ANSWER: 21")["reason"], "unsupported_request")

    def test_no_expected_answer_parameters(self):
        for function in [verify_response, runner.execute, runner.generate_answer, CoveragePolicy.select]:
            self.assertNotIn("expected", inspect.signature(function).parameters)

    def test_small_threshold_unchanged_and_medium_handed_to_large(self):
        p = policy()
        self.assertEqual(p.select("2 + 3")["model"], runner.SMALL)
        self.assertEqual(p.select("21 * 34")["model"], runner.LARGE)
        self.assertTrue(p.select("21 * 34")["requires_runtime_verification"])
        p.metrics["addition_small"][runner.SMALL]["lower_accuracy"] = 0.799
        self.assertEqual(p.select("2 + 3")["model"], runner.LARGE)

    def test_easy_request_uses_one_small_call(self):
        with patch.object(runner, "generate_answer", return_value={"answer": "FINAL_ANSWER: 5"}) as call:
            row = runner.execute("2 + 3", policy())
        call.assert_called_once_with(runner.SMALL, "2 + 3", False)
        self.assertEqual(row["status"], "answered")

    def test_wrong_small_falls_back_to_large(self):
        with patch.object(runner, "generate_answer", side_effect=[
            {"answer": "FINAL_ANSWER: 20"}, {"answer": "FINAL_ANSWER: 21"},
        ]):
            row = runner.execute("2 + 19", policy())
        self.assertEqual(row["model"], runner.LARGE)
        self.assertEqual(row["attempts"][0]["reason"], "arithmetic_mismatch")
        self.assertTrue(row["large_handoff"])
        self.assertGreater(row["fallback_seconds"], 0)

    def test_low_confidence_small_is_not_called(self):
        with patch.object(runner, "generate_answer", return_value={"answer": "FINAL_ANSWER: 714"}) as call:
            row = runner.execute("21 * 34", policy())
        call.assert_called_once_with(runner.LARGE, "21 * 34", False)
        self.assertEqual(row["selection_reason"], "small_confidence_insufficient")

    def test_large_retry_on_error_and_format_failure(self):
        for first in [OSError("offline"), {"answer": "714"}]:
            with self.subTest(first=first), patch.object(runner, "generate_answer", side_effect=[
                first, {"answer": "FINAL_ANSWER: 714"},
            ]):
                row = runner.execute("21 * 34", policy())
            self.assertEqual(row["status"], "answered")
            self.assertTrue(row["large_retry"])
            self.assertEqual(len(row["attempts"]), 2)

    def test_large_failure_is_bounded_and_never_replaced_by_calculator(self):
        with patch.object(runner, "generate_answer", return_value={"answer": "FINAL_ANSWER: 1"}) as call:
            row = runner.execute("21 * 34", policy())
        self.assertEqual(call.call_count, 2)
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["answer"], "")
        self.assertIn("arithmetic_mismatch", row["reason"])

    def test_unsupported_request_no_call(self):
        with patch.object(runner, "generate_answer") as call:
            row = runner.execute("Explain AI", policy())
        call.assert_not_called()
        self.assertEqual(row["reason"], "unsupported_request")

    def test_new_problems_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "old.json").write_text('{"problems": [{"question": "3 + 8"}]}')
            problems = experiment.unseen_problems(path)
            self.assertEqual(problems, experiment.unseen_problems(path))
        keys = [problem_key(p["question"]) for p in problems]
        self.assertEqual(len(keys), 100)
        self.assertEqual(len(set(keys)), 100)
        self.assertNotIn(problem_key("8 + 3"), keys)

    def test_historical_causes_confirm_no_large_calls(self):
        causes = experiment.historical_causes(Path(f"{experiment.HISTORICAL}_evaluation.csv"))
        self.assertEqual(causes["unanswered"], 100)
        self.assertEqual(causes["large_model_calls"], 0)
        self.assertEqual(causes["extraction_failures"], 0)

    def test_summary_counts_all_attempt_costs(self):
        with patch.object(runner, "generate_answer", side_effect=[
            {"answer": "FINAL_ANSWER: 20"}, {"answer": "FINAL_ANSWER: 21"},
        ]):
            row = runner.execute("2 + 19", policy())
        row.update(dataset="test", difficulty="easy", correct=True, audit_mismatches=0)
        summary = experiment.summarize([row])[0]
        self.assertEqual(summary["api_calls"], 2)
        self.assertEqual(summary["fallback_rate"], 1)
        self.assertIsNone(summary["cost"])

    def test_full_pipeline_mocked_in_temporary_directory_only(self):
        # These temporary artifacts never enter the real benchmark output directory.
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(experiment, "unseen_problems", return_value=[{
                 "question": "2 + 3", "expected": 5, "difficulty": "easy", "bucket": "addition_small"}]), \
             patch.object(experiment, "check_ollama"), \
             patch.object(experiment.old, "generate", return_value={"answer": "FINAL_ANSWER: 5"}), \
             patch.object(runner, "generate_answer", return_value={"answer": "FINAL_ANSWER: 5"}), \
             patch.object(experiment.urllib.request, "urlopen") as urlopen, \
             contextlib.redirect_stdout(io.StringIO()):
            urlopen.return_value.__enter__.return_value = io.StringIO('{"models": []}')
            old = Path(folder) / "old.csv"
            old.write_bytes(b"keep")
            prefix = experiment.run(Path(folder))
            self.assertEqual(old.read_bytes(), b"keep")
            self.assertTrue(Path(f"{prefix}_report.md").exists())
            self.assertTrue(Path(f"{prefix}_attempts.csv").exists())


if __name__ == "__main__":
    unittest.main()

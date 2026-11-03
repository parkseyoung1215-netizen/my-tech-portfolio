import contextlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import adaptive_router as router
from src import adaptive_experiment as experiment


def profile(small_correct=20, large_correct=20, samples=20, bucket="addition_small"):
    return router.make_profile([
        {"bucket": bucket, "model": model, "correct": index < correct}
        for model, correct in [(router.SMALL, small_correct), (router.LARGE, large_correct)]
        for index in range(samples)
    ])


class TestAdaptiveRouter(unittest.TestCase):
    def test_conservative_confidence(self):
        self.assertGreater(router.lower_accuracy(20, 20), 0.8)
        self.assertLess(router.lower_accuracy(19, 20), 0.8)
        self.assertEqual(router.lower_accuracy(0, 0), 0)

    def test_small_first_without_expected_answer(self):
        decision = router.plan("2 + 3", profile())
        self.assertEqual(decision["attempts"][0]["model"], router.SMALL)
        self.assertNotIn("expected", inspect.signature(router.execute).parameters)
        self.assertNotIn("expected", inspect.signature(router.plan).parameters)

    def test_bucket_features(self):
        self.assertEqual(router.bucket_for("2 + 3"), "addition_small")
        self.assertEqual(router.bucket_for("11 + 3"), "addition_large")
        self.assertIsNone(router.bucket_for("tell me something"))

    def test_ineligible_small_skipped(self):
        with patch.object(router, "generate", return_value={"answer": "FINAL_ANSWER: 5"}) as call:
            result = router.execute("2 + 3", profile(10, 20))
        self.assertEqual(result["model"], router.LARGE)
        self.assertEqual(call.call_args.args[0], router.LARGE)
        self.assertEqual(len(result["attempts"]), 1)

    def test_malformed_small_falls_back(self):
        with patch.object(router, "generate", side_effect=[
            {"answer": "5"}, {"answer": "FINAL_ANSWER: 5"},
        ]):
            result = router.execute("2 + 3", profile())
        self.assertEqual(result["model"], router.LARGE)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["status"], "extraction_failed")

    def test_api_error_falls_back(self):
        with patch.object(router, "generate", side_effect=[
            OSError("offline"), {"answer": "FINAL_ANSWER: 5"},
        ]):
            result = router.execute("2 + 3", profile())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["attempts"][0]["status"], "error")

    def test_cannot_detect_well_formatted_wrong_answer(self):
        with patch.object(router, "generate", return_value={"answer": "FINAL_ANSWER: 999"}) as call:
            result = router.execute("2 + 3", profile())
        self.assertEqual(result["model"], router.SMALL)
        self.assertEqual(result["answer"], "FINAL_ANSWER: 999")
        call.assert_called_once()

    def test_unknown_request_rejected_without_generation(self):
        with patch.object(router, "generate") as call:
            result = router.execute("summarize", profile())
        call.assert_not_called()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("Neither model", result["reason"])

    def test_unreliable_models_reject_medium(self):
        with patch.object(router, "generate") as call:
            result = router.execute("21 * 34", profile(5, 10, bucket="medium"))
        call.assert_not_called()
        self.assertEqual(result["status"], "rejected")

    def test_legacy_supported_scope_is_preserved(self):
        with patch.object(router, "generate", return_value={"answer": "FINAL_ANSWER: 5"}) as call:
            result = router.execute("2 + 3", profile(0, 0))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(call.call_args.args[2], "legacy")

    def test_insufficient_samples_and_prompt_mismatch(self):
        decision = router.plan("2 + 3", profile(samples=3))
        self.assertTrue(all(a["prompt_version"] == "legacy" for a in decision["attempts"]))
        with self.assertRaises(ValueError):
            router.plan("2 + 3", {"prompt_version": "unknown"})

    def test_splits_exclude_legacy_and_commutative_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old.csv"
            source.write_text("question\n4 + 1\n", encoding="utf-8")
            calibration, evaluation = experiment.build_splits(directory)
            self.assertEqual((calibration, evaluation), experiment.build_splits(directory))
        self.assertEqual(len(calibration), 80)
        self.assertEqual(len(evaluation), 40)
        keys = [experiment.problem_key(p["question"]) for p in calibration + evaluation]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(experiment.problem_key("4 + 1"), keys)
        self.assertEqual(experiment.problem_key("3 + 8"), experiment.problem_key("8 + 3"))

    def test_summary_counts_fallback_work_and_refusals(self):
        def fake_generate(*args):
            return {"answer": "FINAL_ANSWER: 5", "output_tokens": 4, "eval_duration": 0.1}
        with patch.object(router, "generate", side_effect=fake_generate):
            answered = router.execute("2 + 3", profile())
            rejected = router.execute("summarize", profile())
        rows = [dict(answered, bucket="addition_small", correct=True),
                dict(rejected, bucket="hard", correct=False)]
        summary = next(s for s in experiment.summarize(rows) if s["bucket"] == "all")
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["answer_rate"], 0.5)
        self.assertEqual(summary["answered_accuracy"], 1)
        self.assertEqual(summary["output_tokens"], 4)
        self.assertIsNone(summary["cost"])

    def test_pairing_uses_same_questions_not_all_request_average(self):
        rows = [
            {"strategy": strategy, "question": q, "status": status,
             "correct": True, "elapsed_seconds": elapsed}
            for strategy, q, status, elapsed in [
                ("fixed", "a", "answered", 2), ("fixed", "b", "answered", 10),
                ("improved", "a", "answered", 1), ("improved", "b", "rejected", 0.01),
            ]
        ]
        pair = experiment.paired_comparisons(rows)[0]
        self.assertEqual(pair["common_answered"], 1)
        self.assertEqual(pair["lost_answers"], 1)
        self.assertEqual(pair["same_coverage_speedup"], 2)

    def test_preflight_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(experiment, "check_ollama", side_effect=RuntimeError("missing")):
            with self.assertRaises(RuntimeError):
                experiment.run_experiment(Path(folder))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_failed_small_never_calls_unreliable_large(self):
        with patch.object(router, "generate", return_value={"answer": "invalid"}) as call:
            result = router.execute("21 * 34", profile(20, 0, bucket="medium"))
        call.assert_called_once()
        self.assertEqual(call.call_args.args[0], router.SMALL)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("all eligible attempts failed", result["reason"])

    def test_full_experiment_artifacts_and_preservation_without_network(self):
        calibration = [{"question": "2 + 3", "expected": 5, "bucket": "addition_small"}]
        evaluation = [{"question": "4 + 1", "expected": 5, "bucket": "addition_small"}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(experiment, "build_splits", return_value=(calibration, evaluation)), \
             patch.object(experiment, "check_ollama"), \
             patch.object(router, "generate", return_value={
                 "answer": "FINAL_ANSWER: 5", "wall_time": 0.01,
                 "output_tokens": 4, "eval_duration": 0.005,
             }), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            old = Path(directory) / "old.csv"
            old.write_bytes(b"existing data")
            prefix = experiment.run_experiment(Path(directory))
            self.assertEqual(old.read_bytes(), b"existing data")
            self.assertTrue(Path(f"{prefix}_profile.json").is_file())
            self.assertTrue(Path(f"{prefix}_paired.csv").is_file())
            import csv
            with Path(f"{prefix}_evaluation.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r["question"] == "4 + 1" for r in rows))


if __name__ == "__main__":
    unittest.main()

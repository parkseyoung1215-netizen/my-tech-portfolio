import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

from src import routing_runner as runner


class TestRoutingRunner(unittest.TestCase):
    def test_selected_model_generates(self):
        with patch.object(runner, "ask", return_value={"answer": "FINAL_ANSWER: 5", "wall_time": 0.1}) as ask:
            result = runner.execute_request("4 + 1")
        ask.assert_called_once_with("qwen2.5:1.5b", "4 + 1")
        self.assertEqual(result["status"], "generated")
        self.assertIsNone(result["cost"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_rejection_never_calls_ollama(self):
        with patch.object(runner, "ask") as ask:
            for question in ["45 * 93", "(467 * 54) - 87", "Explain AI"]:
                result = runner.execute_request(question)
                self.assertEqual(result["status"], "rejected")
                self.assertIn("최소 정확도", result["reason"])
        ask.assert_not_called()

    def test_generation_error_preserves_model_and_elapsed_time(self):
        with patch.object(runner, "ask", side_effect=OSError("offline")):
            result = runner.execute_request("4 + 1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model"], "qwen2.5:1.5b")
        self.assertIn("offline", result["reason"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_fixed_baseline_bypasses_router(self):
        with patch.object(runner, "route_request") as route, patch.object(
            runner, "ask", return_value={"answer": "FINAL_ANSWER: 4185", "wall_time": 0.1}
        ):
            result = runner.execute_request("45 * 93", fixed_model=runner.FIXED_MODEL)
        route.assert_not_called()
        self.assertEqual(result["status"], "generated")

    def test_comparison_saves_pairs_and_does_not_overwrite(self):
        def fake_ask(model, question):
            return {"answer": "FINAL_ANSWER: 5", "wall_time": 0.1}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "check_ollama"), \
             patch.object(runner, "ask", side_effect=fake_ask), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            old = Path(directory) / "existing.csv"
            old.write_bytes(b"preserve")
            problems = [("easy", "4 + 1", 5), ("medium", "45 * 93", 4185)]
            first, _, summaries = runner.compare(output_dir=directory, problem_set=problems)
            second, _, _ = runner.compare(output_dir=directory, problem_set=problems)
            self.assertNotEqual(first, second)
            self.assertEqual(old.read_bytes(), b"preserve")
            with first.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                sorted(r["question"] for r in rows if r["strategy"] == "fixed"),
                sorted(r["question"] for r in rows if r["strategy"] == "auto"),
            )
            auto = next(s for s in summaries if s["strategy"] == "auto" and s["difficulty"] == "all")
            self.assertEqual(auto["accuracy"], 0.5)
            self.assertEqual(auto["answered_accuracy"], 1)
            self.assertEqual(auto["rejected"], 1)
            self.assertEqual(auto["generated"], 1)

    def test_preflight_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "check_ollama", side_effect=RuntimeError("missing")), \
             patch.object(runner, "ask") as ask:
            with self.assertRaises(RuntimeError):
                runner.compare(output_dir=directory)
            ask.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

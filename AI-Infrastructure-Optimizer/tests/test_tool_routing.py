import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import safe_calculator as calculator
from src import tool_router
from src import tool_experiment as experiment


def calc(text):
    return calculator.calculate(calculator.parse_expression(text))


def profile():
    return json.loads(experiment.PROFILE.read_text())


class TestSafeCalculator(unittest.TestCase):
    def test_precedence_parentheses_and_signed_integers(self):
        for text, value in [("2 + 3 * 4", 14), ("(2 + 3) * 4", 20),
                            ("-(2 + 3) * 4", -20), ("2 - -3", 5),
                            ("0 * 123", 0), ("(123 * 42) - 13", 5153)]:
            with self.subTest(text=text):
                self.assertEqual(calc(text), value)

    def test_rejects_code_and_python_extensions(self):
        for text in ["__import__('os')", "abs(-2)", "[1,2]", "True + 1", "2 ** 1000000",
                     "1 / 0", "2 // 1", "3 % 2", "1 << 2", "0x10 + 1", "2_000 + 1"]:
            with self.subTest(text=text), self.assertRaises(calculator.ToolDeclined):
                calc(text)

    def test_requires_complete_unambiguous_expression(self):
        for text in ["Calculate 2 + 3", "2 + 3 people", "1.5 + 2", "１ + ２", "42", "", None,
                     "2 +", "0002 + 0003", "1 + 2; 3 + 4"]:
            with self.subTest(text=text), self.assertRaises(calculator.ToolDeclined):
                calc(text)

    def test_length_node_depth_and_literal_limits(self):
        cases = [("1 + " * 100 + "1", "length_limit"),
                 ("+".join(["1"] * 22), "node_limit"),
                 ("-" * 20 + "(1+2)", "depth_limit"),
                 ("1000000000000 + 1", "literal_limit")]
        for text, reason in cases:
            with self.subTest(reason=reason), self.assertRaises(calculator.ToolDeclined) as ctx:
                calc(text)
            self.assertEqual(ctx.exception.reason, reason)

    def test_result_limit_checks_intermediates(self):
        with self.assertRaises(calculator.ToolDeclined) as ctx:
            calc(" * ".join(["999999999999"] * 4))
        self.assertEqual(ctx.exception.reason, "result_limit")

    def test_no_eval_or_exec_calls_in_parser(self):
        source = Path(calculator.__file__).read_text()
        calls = [node.func.id for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("eval", calls)
        self.assertNotIn("exec", calls)

    def test_untrusted_tree_rejected(self):
        with self.assertRaises(calculator.ToolDeclined):
            calculator.calculate(ast.parse("danger()", mode="eval"))


class TestToolRouter(unittest.TestCase):
    def test_direct_answer_without_model_or_label_input(self):
        with patch.object(tool_router, "execute_adaptive") as fallback:
            row = tool_router.execute("2 + 19", {})
        fallback.assert_not_called()
        self.assertEqual(row["answer"], "FINAL_ANSWER: 21")
        self.assertEqual(row["source"], "tool")
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(row["attempts"], [])
        self.assertGreater(row["elapsed_seconds"], 0)

    def test_non_arithmetic_goes_to_original_router_unchanged(self):
        question = "Explain AI"
        frozen = {"profile": "sentinel"}
        with patch.object(tool_router, "execute_adaptive", return_value={
            "model": "", "answer": "", "status": "rejected", "attempts": [], "reason": "no model",
        }) as fallback:
            row = tool_router.execute(question, frozen)
        fallback.assert_called_once_with(question, frozen, "improved", 0.8)
        self.assertEqual(row["tool_calls"], 0)
        self.assertTrue(row["model_fallback"])
        self.assertEqual(row["reason"], "no model")

    def test_calculation_limit_uses_model_fallback(self):
        with patch.object(tool_router, "execute_adaptive", return_value={
            "model": "", "answer": "", "status": "rejected", "attempts": [], "reason": "no model",
        }):
            row = tool_router.execute(" * ".join(["999999999999"] * 4), {})
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(row["fallback_reason"], "result_limit")

    def test_unexpected_tool_failure_is_logged_and_falls_back(self):
        with patch.object(tool_router, "calculate", side_effect=RuntimeError("tool offline")), \
             patch.object(tool_router, "execute_adaptive", return_value={
                 "model": "m", "answer": "FINAL_ANSWER: 5", "status": "answered", "attempts": [], "reason": "model",
             }):
            row = tool_router.execute("2 + 3", {})
        self.assertEqual(row["source"], "model_fallback")
        self.assertIn("tool_error", row["fallback_reason"])

    def test_model_fallback_exception_is_not_a_fake_answer(self):
        with patch.object(tool_router, "execute_adaptive", side_effect=OSError("offline")):
            row = tool_router.execute("Explain AI", {})
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["answer"], "")
        self.assertIn("offline", row["reason"])


class TestToolExperiment(unittest.TestCase):
    def test_exact_original_datasets_loaded(self):
        datasets = experiment.load_datasets(experiment.SOURCE_MANIFEST)
        self.assertEqual(len(datasets["replay200"]), 200)
        self.assertEqual(len(datasets["unseen100"]), 100)

    def test_success_requires_both_metrics_to_improve(self):
        summaries = [{"dataset": "d", "strategy": s, "difficulty": "all", "accuracy": acc, "answer_rate": rate}
                     for s, acc, rate in [("fixed", 0.5, 1), ("adaptive", 0.4, 0.5),
                                           ("coverage", 0.8, 0.8), ("tool", 1, 1)]]
        pairs = experiment.comparisons([], summaries)
        # No rows means no datasets. Supply a minimal, explicitly recorded dataset.
        rows = [{"dataset": "d", "strategy": s, "question": "2+3", "status": "answered",
                 "correct": True, "elapsed_seconds": 0.1} for s in experiment.STRATEGIES]
        pairs = experiment.comparisons(rows, summaries)
        self.assertFalse(next(p for p in pairs if p["baseline"] == "fixed")["success_both_improved"])
        self.assertTrue(next(p for p in pairs if p["baseline"] == "coverage")["success_both_improved"])

    def test_empty_case_csv_keeps_header_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "none.csv"
            experiment.write_records(path, [], ["question", "reason"])
            self.assertEqual(path.read_text().strip(), "question,reason")
            with self.assertRaises(FileExistsError):
                experiment.write_records(path, [], ["question"])

    def test_profile_and_labels_are_not_needed_for_direct_arithmetic(self):
        row = experiment.execute_strategy("tool", "21 * 34", {}, None)
        self.assertEqual(row["answer"], "FINAL_ANSWER: 714")
        self.assertEqual(row["verification_arithmetic_calls"], 0)

    def test_coverage_verifier_calls_are_separate_from_direct_tool(self):
        with patch.object(experiment, "execute_coverage", return_value={
            "question": "2+3", "strategy": "coverage", "model": "m", "answer": "FINAL_ANSWER: 5",
            "status": "answered", "attempts": [{"model": "m", "status": "answered", "reason": "verified", "answer": "FINAL_ANSWER: 5"}],
            "reason": "verified", "elapsed_seconds": 0.1,
        }):
            row = experiment.execute_strategy("coverage", "2+3", {}, None)
        self.assertEqual(row["tool_calls"], 0)
        self.assertEqual(row["verification_arithmetic_calls"], 1)

    def test_pipeline_uses_only_temporary_mocked_test_artifacts(self):
        problems = [{"question": "2 + 3", "expected": 5, "difficulty": "easy", "bucket": "addition_small"}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(experiment, "load_datasets", return_value={"replay200": problems, "unseen100": problems}), \
             patch.object(experiment, "check_ollama"), \
             patch.object(experiment, "safety_cases", return_value=experiment.safety_cases()[:2]), \
             patch.object(experiment.adaptive, "generate", return_value={"answer": "FINAL_ANSWER: 5"}), \
             patch("src.coverage_runner.generate_answer", return_value={"answer": "FINAL_ANSWER: 5"}), \
             patch.object(experiment.urllib.request, "urlopen") as urlopen, \
             contextlib.redirect_stdout(io.StringIO()):
            urlopen.return_value.__enter__.return_value = io.StringIO('{"models": []}')
            existing = Path(folder) / "keep.csv"
            existing.write_bytes(b"keep")
            prefix = experiment.run(Path(folder))
            self.assertEqual(existing.read_bytes(), b"keep")
            self.assertTrue(Path(f"{prefix}_report.md").is_file())
            self.assertTrue(Path(f"{prefix}_tool_misapplications.csv").is_file())


if __name__ == "__main__":
    unittest.main()

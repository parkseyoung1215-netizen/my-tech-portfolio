from collections import Counter
import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import mixed_dataset as data
from src import mixed_runtime as runtime
from src import mixed_experiment as experiment
from src.coverage_policy import CoveragePolicy


def profile():
    return json.loads(experiment.PROFILE.read_text())


class TestMixed(unittest.TestCase):
    def test_new_balanced_set_and_no_prior_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "old.csv").write_text("question\n3 + 8\n")
            tasks = data.build_dataset(folder)
            prior = data.prior_questions(folder)
            self.assertEqual(tasks, data.build_dataset(folder))
        self.assertEqual(Counter(t["category"] for t in tasks), {k: 20 for k in experiment.CATEGORIES})
        keys = {data.problem_key(t["question"]) for t in tasks}
        self.assertEqual(len(keys), 100)
        self.assertFalse(keys & prior)

    def test_reference_choice_label_is_unique_and_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            tasks = data.build_dataset(folder)
        for task in tasks:
            if task["task_format"] == "choice":
                self.assertIn(task["expected"], "ABCD")
                self.assertTrue(all(f"\n{c}) " in task["question"] for c in "ABCD"))
            else:
                self.assertEqual(int(task["expected"]), data.reference_answer(task["question"]))

    def test_question_text_classification_and_tool_guard(self):
        cases = [("2 + 3", "arithmetic"),
                 ("What does this code print?\n```python\nprint(2+3)\n```", "coding"),
                 ("Read the passage: '2 + 3' is a password.", "comprehension"),
                 ("Reason about a schedule: X before Y.", "logic"),
                 ("Which gas is oxygen?", "knowledge"), ("Explain this", "unknown")]
        for question, expected in cases:
            self.assertEqual(runtime.infer_category(question), expected)

    def test_runtime_has_no_labels_or_category_inputs(self):
        names = inspect.signature(runtime.execute).parameters
        self.assertNotIn("expected", names)
        self.assertNotIn("category", names)
        self.assertNotIn("task_format", names)

    def test_exact_final_contract_and_independent_parser(self):
        cases = [("FINAL_ANSWER: b", "B"), ("FINAL_ANSWER: +0012", "12"),
                 ("work\nFINAL_ANSWER: D", "D"), ("B", None),
                 ("FINAL_ANSWER: B because...", None),
                 ("FINAL_ANSWER: A\nFINAL_ANSWER: B", None)]
        for text, expected in cases:
            self.assertEqual(runtime.extract_final(text), expected)
            self.assertEqual(experiment.audit_extract(text), expected)

    def test_same_general_model_path_for_all_strategies(self):
        frozen = profile()
        policy = CoveragePolicy(frozen)
        with patch.object(runtime, "generate_general", return_value={"answer": "FINAL_ANSWER: C"}) as call:
            rows = [runtime.execute("What is water?\nA) x\nB) y\nC) H2O\nD) z", s, frozen, policy)
                    for s in experiment.STRATEGIES]
        self.assertEqual(call.call_count, 4)
        self.assertTrue(all(r["model"] == runtime.old.LARGE and r["general_adapter"] for r in rows))
        self.assertTrue(all(r["tool_calls"] == 0 for r in rows))

    def test_no_model_for_explicit_tool_arithmetic(self):
        with patch.object(runtime, "generate_general") as general, \
             patch.object(runtime.old, "generate") as arithmetic:
            row = runtime.execute("27 + 38", "tool", {}, None)
        general.assert_not_called()
        arithmetic.assert_not_called()
        self.assertEqual(row["answer"], "FINAL_ANSWER: 65")

    def test_error_and_invalid_format_preserved(self):
        for response in [OSError("offline"), {"answer": "the answer is A"}]:
            with patch.object(runtime, "generate_general", side_effect=[response]):
                row = runtime.execute("Which option?", "tool", {}, None)
            self.assertNotEqual(row["status"], "answered")
            self.assertEqual(len(row["attempts"]), 1)

    def test_summary_counts_partial_coverage_and_missing_resources(self):
        row = {"strategy": "tool", "category": "knowledge", "status": "answered", "correct": True,
               "attempts": [], "elapsed_seconds": 0.1, "tool_calls": 0, "general_adapter": True,
               "inferred_category": "knowledge", "source": "model", "cpu_seconds": None,
               "boundary_max_rss_mib": None, "extraction_disagreement": False}
        overall = next(r for r in experiment.summarize([row]) if r["category"] == "all")
        self.assertEqual(overall["accuracy"], 1)
        self.assertIsNone(overall["cpu_seconds_total"])
        self.assertIsNone(overall["cost"])

    def test_complete_pipeline_uses_temporary_mocked_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            # One task per category is sufficient for artifact/legacy audit integration.
            tasks = data.build_dataset(folder)
            tiny = [next(t for t in tasks if t["category"] == c) for c in experiment.CATEGORIES]
            snapshot = {"time": 1, "duration": 0.01, "error": "", "processes": {
                "1": {"cpu_seconds": 1, "rss_kib": 1024}}}
            with patch.object(experiment, "build_dataset", return_value=tiny), \
                 patch.object(experiment, "check_ollama"), \
                 patch.object(experiment, "process_snapshot", return_value=snapshot), \
                 patch.object(experiment, "generate_general", return_value={"answer": "FINAL_ANSWER: A"}), \
                 patch.object(runtime, "generate_general", return_value={"answer": "FINAL_ANSWER: A"}), \
                 patch.object(runtime.old, "generate", return_value={"answer": "FINAL_ANSWER: 1"}), \
                 patch("src.coverage_runner.generate_answer", return_value={"answer": "FINAL_ANSWER: 1"}), \
                 patch.object(experiment.urllib.request, "urlopen") as urlopen, \
                 contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value = io.StringIO('{"models": []}')
                old = Path(folder) / "preserve.csv"
                old.write_bytes(b"preserve")
                prefix = experiment.run(folder)
                self.assertEqual(old.read_bytes(), b"preserve")
                self.assertTrue(Path(f"{prefix}_report.md").is_file())
                self.assertTrue(Path(f"{prefix}_legacy_compatibility.csv").is_file())


if __name__ == "__main__":
    unittest.main()

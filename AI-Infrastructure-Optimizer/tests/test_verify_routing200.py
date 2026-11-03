import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import verify_routing200 as verify


class TestVerification200(unittest.TestCase):
    def test_unique_holdout_excludes_prior_csv_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "old.csv").write_text("question\n3 + 8\n", encoding="utf-8")
            (path / "old.json").write_text(json.dumps({"calibration": [{"question": "5 + 7"}]}))
            problems = verify.build_holdout(path)
            again = verify.build_holdout(path)
            keys = [verify.problem_key(p["question"]) for p in problems]
        self.assertEqual(len(keys), 200)
        self.assertEqual(len(set(keys)), 200)
        self.assertNotIn(verify.problem_key("8 + 3"), keys)
        self.assertNotIn(verify.problem_key("7 + 5"), keys)
        self.assertEqual(problems, again)
        for problem in problems:
            self.assertEqual(problem["expected"], verify.reference_answer(problem["question"]))

    def test_exhausted_pool_fails_without_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / "old.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(["question"])
                writer.writerows([[f"{a} + {b}"] for a in range(1, 21) for b in range(a, 21)])
            with self.assertRaises(ValueError):
                verify.build_holdout(folder)

    def test_reference_rejects_execution(self):
        self.assertEqual(verify.reference_answer("(123 * 42) - 13"), 5153)
        with self.assertRaises(ValueError):
            verify.reference_answer("__import__('os').getcwd()")

    def test_independent_extraction_audit(self):
        for text, expected in [
            ("FINAL_ANSWER: 12", 12), ("work\nfinal_answer : -12\n", -12),
            ("FINAL_ANSWER: +0", 0), ("12", None),
            ("FINAL_ANSWER: 12\nFINAL_ANSWER: invalid", None),
            ("FINAL_ANSWER: 12.0", None), ("FINAL_ANSWER:\n12", None),
            ("FINAL_ANSWER: １２", None), ("FINAL_ANSWER: 12 extra", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(verify.audit_extract(text), expected)

    def test_cpu_counter_formats(self):
        self.assertAlmostEqual(verify.cpu_seconds("0:11.53"), 11.53)
        self.assertEqual(verify.cpu_seconds("01:02:03"), 3723)
        self.assertEqual(verify.cpu_seconds("1-00:00:01"), 86401)

    def test_resource_delta_is_real_counter_difference(self):
        before = {"time": 1, "duration": 0.01, "error": "", "processes": {
            "1": {"cpu_seconds": 2.0, "rss_kib": 1024}}}
        after = {"time": 2, "duration": 0.02, "error": "", "processes": {
            "1": {"cpu_seconds": 2.5, "rss_kib": 2048}}}
        result = verify.resource_delta(before, after)
        self.assertEqual(result["cpu_seconds"], 0.5)
        self.assertEqual(result["cpu_percent"], 50)
        self.assertEqual(result["boundary_max_rss_mib"], 2)
        after["processes"] = {}
        self.assertIsNone(verify.resource_delta(before, after)["cpu_seconds"])

    def test_unavailable_resources_not_fabricated(self):
        with patch.object(verify.subprocess, "run", side_effect=OSError("denied")):
            snapshot = verify.process_snapshot()
        self.assertEqual(snapshot["processes"], {})
        self.assertIn("denied", snapshot["error"])
        delta = verify.resource_delta(snapshot, snapshot)
        self.assertIsNone(delta["cpu_seconds"])
        self.assertIsNone(delta["boundary_max_rss_mib"])

    def test_process_snapshot_includes_ollama_descendants_only(self):
        output = "1 0 1024 0:01.00 /bin/ollama\n2 1 2048 0:02.00 /bin/runner\n3 0 4096 0:03.00 /bin/other\n"
        with patch.object(verify.subprocess, "run") as run:
            run.return_value.stdout = output
            snapshot = verify.process_snapshot()
        self.assertEqual(set(snapshot["processes"]), {"1", "2"})

    def test_aggregate_fallback_rate_and_missing_measurements(self):
        row = {"strategy": "improved", "bucket": "medium", "question": "21 * 34",
               "status": "answered", "correct": True, "model": verify.router.LARGE,
               "elapsed_seconds": 1, "attempts": [
                   {"model": verify.router.SMALL, "status": "error"},
                   {"model": verify.router.LARGE, "status": "answered"}],
               "cpu_seconds": None, "cpu_percent": None,
               "boundary_max_rss_mib": None, "audit_mismatches": 0}
        summary = next(s for s in verify.aggregate([row]) if s["bucket"] == "all")
        self.assertEqual(summary["fallback_rate"], 1)
        self.assertEqual(summary["api_calls"], 2)
        self.assertIsNone(summary["cpu_seconds_total"])
        self.assertIsNone(summary["cost"])

    def test_pipeline_writes_report_and_checks_preservation_without_live_calls(self):
        # Temporary mocked integration artifacts are unit-test data only.
        problems = [
            {"question": "2 + 3", "expected": 5, "difficulty": "easy", "bucket": "addition_small"},
            {"question": "21 * 34", "expected": 714, "difficulty": "medium", "bucket": "medium"},
        ]
        snapshot = {"time": 1, "duration": 0.01, "error": "", "processes": {
            "1": {"cpu_seconds": 2, "rss_kib": 1024}}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(verify, "build_holdout", return_value=problems), \
             patch.object(verify, "check_ollama"), \
             patch.object(verify, "process_snapshot", return_value=snapshot), \
             patch.object(verify.router, "generate", return_value={
                 "answer": "FINAL_ANSWER: 5", "output_tokens": 4, "eval_duration": 0.01}), \
             patch.object(verify.urllib.request, "urlopen") as urlopen, \
             contextlib.redirect_stdout(io.StringIO()):
            urlopen.return_value.__enter__.return_value = io.StringIO('{"models": []}')
            old = Path(directory) / "old.csv"
            old.write_bytes(b"preserved")
            prefix = verify.run(Path(directory))
            self.assertEqual(old.read_bytes(), b"preserved")
            self.assertTrue(Path(f"{prefix}_report.md").exists())
            with Path(f"{prefix}_evaluation.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["audit_mismatches"] == "0" for r in rows))


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path

from src.model_router import (
    estimate_difficulty, load_benchmark, route_request, select_model,
)


class TestModelRouter(unittest.TestCase):
    def test_load_current_experiment(self):
        data = load_benchmark()
        small = data["qwen2.5:0.5b"]
        large = data["qwen2.5:1.5b"]
        self.assertEqual(small["easy"]["quality"], 0.1)
        self.assertEqual(large["easy"]["quality"], 1)
        self.assertEqual(large["medium"]["quality"], 0.6)
        self.assertEqual(large["hard"]["quality"], 0)
        self.assertEqual(small["easy"]["samples"], 10)
        self.assertAlmostEqual(small["easy"]["latency"], 0.07474)
        self.assertIsNone(small["easy"]["cost"])

    def test_estimation_stays_within_measured_domain(self):
        for text, expected in [
            ("4 + 1", "easy"), ("45 * 93", "medium"),
            ("(467 * 54) - 87", "hard"), ("Summarize this", None),
            ("1000 + 2", None), ("1 / 2", None), ("", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(estimate_difficulty(text), expected)

    def test_easy_selects_accurate_model_with_unknown_cost(self):
        result = route_request("4 + 1")
        self.assertTrue(result["suitable"])
        self.assertEqual(result["model"], "qwen2.5:1.5b")
        self.assertIsNone(result["cost"])
        self.assertEqual(result["cost_status"], "unmeasured")

    def test_medium_quality_threshold(self):
        self.assertFalse(route_request("45 * 93", min_quality=0.8)["suitable"])
        self.assertTrue(route_request("45 * 93", min_quality=0.6)["suitable"])

    def test_hard_and_unknown_reject(self):
        for text in ["(467 * 54) - 87", "Explain AI", "計算してください"]:
            result = route_request(text)
            self.assertFalse(result["suitable"])
            self.assertIsNone(result["model"])
            self.assertIn("minimum accuracy", result["reason"])
        self.assertTrue(route_request("Explain AI")["uncertain"])

    def test_latency_constraint(self):
        result = route_request("4 + 1", max_latency=0.1)
        self.assertFalse(result["suitable"])

    def test_unknown_cost_uses_latency(self):
        result = route_request("4 + 1", min_quality=0.1)
        self.assertEqual(result["model"], "qwen2.5:0.5b")
        self.assertEqual(result["selection_basis"], "latency")

    def test_measured_cost_ranking_and_partial_costs(self):
        costs = {"qwen2.5:0.5b": 0.02, "qwen2.5:1.5b": 0.01}
        result = route_request("4 + 1", min_quality=0.1, measured_costs=costs)
        self.assertEqual(result["model"], "qwen2.5:1.5b")
        self.assertEqual(result["cost_status"], "measured")
        partial = route_request("4 + 1", min_quality=0.1,
                                measured_costs={"qwen2.5:1.5b": 0.01})
        self.assertEqual(partial["model"], "qwen2.5:0.5b")
        self.assertEqual(partial["selection_basis"], "latency")

    def test_invalid_constraints(self):
        for kwargs in [{"min_quality": 80}, {"min_quality": float("nan")},
                       {"max_latency": -1}, {"measured_costs": {"x": -1}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                route_request("4 + 1", **kwargs)

    def test_missing_file_rejects_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            result = route_request("4 + 1", benchmark_path=Path(directory) / "absent.csv")
        self.assertFalse(result["suitable"])
        self.assertIn("unavailable", result["reason"])

    def test_csv_failures_and_extraction_failure_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            fields = ["model", "difficulty", "question", "correct", "status", "wall_time"]
            rows = [
                ["m", "easy", "1+1", "True", "correct", "0.1"],
                ["m", "easy", "2+2", "False", "extraction_failed", "0.3"],
            ]

            def write(values):
                with path.open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(fields)
                    writer.writerows(values)

            write(rows)
            metrics = load_benchmark(path)["m"]["easy"]
            self.assertEqual(metrics["quality"], 0.5)
            self.assertEqual(metrics["extraction_failure_rate"], 0.5)
            self.assertAlmostEqual(metrics["latency"], 0.2)
            self.assertFalse(route_request("unknown", benchmark_path=path)["suitable"])
            for invalid in [[], [rows[0], rows[0]],
                            [["m", "easy", "1+1", "True", "incorrect", "0.1"]],
                            [["m", "easy", "1+1", "True", "correct", "nan"]]]:
                write(invalid)
                with self.subTest(rows=invalid), self.assertRaises(ValueError):
                    load_benchmark(path)

    def test_legacy_selection_is_preserved(self):
        self.assertEqual(select_model(0.65, 1)["model"], "small")
        self.assertEqual(select_model(0.8, 2)["model"], "medium")
        self.assertIsNone(select_model(1, 1))


if __name__ == "__main__":
    unittest.main()

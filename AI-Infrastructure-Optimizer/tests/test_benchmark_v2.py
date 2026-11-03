import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import benchmark_v2


class TestBenchmarkValidation(unittest.TestCase):
    def test_records_statuses_and_preserves_existing_csv(self):
        def fake_ask(model, question):
            answers = {
                "1 + 1": "FINAL_ANSWER: 2",
                "2 + 2": "FINAL_ANSWER: 5",
                "3 + 3": "6",
            }
            return {"answer": answers[question], "wall_time": 0.1,
                    "output_tokens": 3}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "data"
            output_dir.mkdir()
            existing = output_dir / "benchmark_v2.csv"
            existing.write_bytes(b"existing results\n")
            real_path = Path
            with patch.object(benchmark_v2, "MODELS", ["test-model"]), \
                 patch.object(benchmark_v2, "questions", [
                     ("easy", "1 + 1", 2), ("easy", "2 + 2", 4),
                     ("easy", "3 + 3", 6),
                 ]), \
                 patch.object(benchmark_v2, "ask", side_effect=fake_ask), \
                 patch.object(benchmark_v2, "Path", side_effect=lambda p: real_path(directory) / p), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                benchmark_v2.main()
                benchmark_v2.main()

            self.assertEqual(existing.read_bytes(), b"existing results\n")
            files = list(output_dir.glob("benchmark_v2_*.csv"))
            self.assertEqual(len(files), 2)
            for path in files:
                with path.open(newline="", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([r["status"] for r in rows],
                                 ["correct", "incorrect", "extraction_failed"])
                self.assertEqual([r["correct"] for r in rows], ["True", "False", "False"])
                self.assertEqual([r["predicted"] for r in rows], ["2", "5", ""])
            self.assertIn("Extraction failed: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()

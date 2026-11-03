
# AI Infrastructure Optimizer
# Intelligent Model Router - Version 1

import csv
import math
from pathlib import Path
import re


DEFAULT_BENCHMARK = (
    Path(__file__).resolve().parent.parent / "data" /
    "benchmark_v2_f2be12734ce5444397f2ca02a62c1869.csv"
)
DIFFICULTIES = ("easy", "medium", "hard")

MODELS = {
    "small": {
        "cost": 0.001,
        "quality": 0.70,
        "latency": 0.5,
    },
    "medium": {
        "cost": 0.004,
        "quality": 0.85,
        "latency": 1.2,
    },
    "large": {
        "cost": 0.012,
        "quality": 0.95,
        "latency": 2.5,
    },
}


def select_model(min_quality, max_latency):
    """
    Select the cheapest model that meets
    the required quality and latency.
    """

    candidates = []

    for name, info in MODELS.items():

        if info["quality"] < min_quality:
            continue

        if info["latency"] > max_latency:
            continue

        candidates.append((name, info))

    if not candidates:
        return None

    best = min(
        candidates,
        key=lambda item: item[1]["cost"],
    )

    return {
        "model": best[0],
        "cost": best[1]["cost"],
        "quality": best[1]["quality"],
        "latency": best[1]["latency"],
    }


def load_benchmark(path=DEFAULT_BENCHMARK):
    """Read per-model, per-difficulty metrics. Accuracy uses all response rows.

    CSV contains no measured monetary cost; cost remains None.
    Malformed or duplicate observations are rejected instead of silently ignored.
    """
    groups = {}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"model", "difficulty", "question", "correct", "status", "wall_time"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Benchmark CSV is missing required columns")
        for row in reader:
            model, difficulty = row["model"], row["difficulty"]
            status = row["status"]
            if (not model or difficulty not in DIFFICULTIES
                    or status not in {"correct", "incorrect", "extraction_failed"}
                    or row["correct"] not in {"True", "False"}
                    or (row["correct"] == "True") != (status == "correct")):
                raise ValueError("Invalid benchmark classification")
            seconds = float(row["wall_time"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid benchmark response time")
            key = (model, difficulty, row["question"])
            if key in seen:
                raise ValueError("Duplicate benchmark observation")
            seen.add(key)
            groups.setdefault(model, {}).setdefault(difficulty, []).append(
                (status, seconds)
            )
    if not groups:
        raise ValueError("Benchmark CSV contains no observations")
    return {
        model: {
            difficulty: {
                "quality": sum(s == "correct" for s, _ in rows) / len(rows),
                "latency": sum(t for _, t in rows) / len(rows),
                "extraction_failure_rate": sum(
                    s == "extraction_failed" for s, _ in rows
                ) / len(rows),
                "samples": len(rows),
                "cost": None,
            }
            for difficulty, rows in difficulties.items()
        }
        for model, difficulties in groups.items()
    }


def estimate_difficulty(request):
    """Recognize only arithmetic shapes and ranges covered by this experiment.

    Free text and other operations remain uncertain; never infer that they are easy.
    """
    if not isinstance(request, str):
        return None
    text = request.strip()
    match = re.fullmatch(r"([0-9]+)\s*\+\s*([0-9]+)", text)
    if match and all(1 <= int(n) <= 20 for n in match.groups()):
        return "easy"
    match = re.fullmatch(r"([0-9]+)\s*\*\s*([0-9]+)", text)
    if match:
        a, b = map(int, match.groups())
        if 20 <= a <= 99 and 10 <= b <= 99:
            return "medium"
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*\*\s*([0-9]+)\s*\)\s*-\s*([0-9]+)", text
    )
    if match:
        a, b, c = map(int, match.groups())
        if 100 <= a <= 999 and 10 <= b <= 99 and 10 <= c <= 99:
            return "hard"
    return None


def route_request(request, min_quality=0.8, max_latency=float("inf"),
                  benchmark_path=DEFAULT_BENCHMARK, measured_costs=None):
    """Route using measured accuracy (0..1), mean seconds and optional USD/request.

    measured_costs must be externally measured estimates for this request.
    If any eligible model has unknown cost, rank by latency without inventing costs.
    Unknown difficulty uses minimum quality and maximum latency across all tiers.
    Observed accuracy is not a guarantee for unseen requests or other task domains.
    """
    if not math.isfinite(min_quality) or not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between 0 and 1")
    if math.isnan(max_latency) or max_latency < 0:
        raise ValueError("max_latency must be nonnegative")
    costs = measured_costs or {}
    if any(not math.isfinite(c) or c < 0 for c in costs.values()):
        raise ValueError("Measured costs must be finite and nonnegative")
    difficulty = estimate_difficulty(request)
    result = {
        "suitable": False, "model": None, "difficulty": difficulty,
        "uncertain": difficulty is None, "cost": None, "cost_status": "unmeasured",
        "benchmark_path": str(benchmark_path),
    }
    try:
        metrics = load_benchmark(benchmark_path)
    except (OSError, ValueError, TypeError) as error:
        return {**result, "reason": f"Benchmark unavailable or invalid: {error}"}

    candidates = []
    assessments = {}
    for model, tiers in metrics.items():
        required = (difficulty,) if difficulty else DIFFICULTIES
        if any(tier not in tiers for tier in required):
            assessments[model] = {"reason": "Missing required difficulty measurements"}
            continue
        quality = min(tiers[t]["quality"] for t in required)
        latency = max(tiers[t]["latency"] for t in required)
        info = {"model": model, "quality": quality, "latency": latency,
                "cost": costs.get(model),
                "cost_status": "measured" if model in costs else "unmeasured"}
        assessments[model] = info.copy()
        if quality >= min_quality and latency <= max_latency:
            candidates.append(info)
    result["assessments"] = assessments
    if not candidates:
        return {**result, "reason": (
            "No model meets the minimum accuracy and response-time constraints; "
            "uncertain difficulty uses worst measured accuracy across all tiers."
            if difficulty is None else
            "No model meets the minimum accuracy and response-time constraints "
            "for the estimated difficulty."
        )}
    all_costs_known = all(c["cost"] is not None for c in candidates)
    best = min(candidates, key=lambda c: (
        (c["cost"], c["latency"], c["model"]) if all_costs_known
        else (c["latency"], c["model"])
    ))
    return {**result, **best, "suitable": True,
            "selection_basis": "cost_then_latency" if all_costs_known else "latency",
            "reason": "Meets measured accuracy and response-time constraints"}


if __name__ == "__main__":

    tasks = [
        {"name": "Simple classification", "quality": 0.65, "latency": 1.0},
        {"name": "Text summarization", "quality": 0.80, "latency": 2.0},
        {"name": "Complex reasoning", "quality": 0.92, "latency": 3.0},
    ]

    for task in tasks:

        result = select_model(
            task["quality"],
            task["latency"],
        )

        print(f"\nTask: {task['name']}")

        if result:
            print(f"Selected model: {result['model']}")
            print(f"Estimated cost: ${result['cost']:.4f}")
            print(f"Estimated quality: {result['quality']}")
            print(f"Estimated latency: {result['latency']}s")
        else:
            print("No model meets the requirements.")

"""Frozen calibration/evaluation splits and paired common-coverage comparisons."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import time
from uuid import uuid4

if __package__:
    from . import adaptive_router as router
    from .answer_validator import validate_answer
    from .routing_runner import check_ollama
else:
    import adaptive_router as router
    from answer_validator import validate_answer
    from routing_runner import check_ollama

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ("fixed", "legacy_auto", "improved", "fixed_prompt_control")


def problem_key(question):
    """Canonicalize commutative operations so swapped operands cannot cross splits."""
    expression = re.sub(r"\s+", "", question)
    match = re.fullmatch(r"([0-9]+)([+*])([0-9]+)", expression)
    if match:
        a, operator, b = match.groups()
        return f"{min(int(a), int(b))}{operator}{max(int(a), int(b))}"
    return expression


def build_splits(data_dir, calibration_size=20, evaluation_size=10):
    excluded = {problem_key(q) for q in ["15 * 8", "144 / 12", "17 + 28", "9 ** 2", "100 - 37", "1 + 1"]}
    for path in Path(data_dir).glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("question"):
                    excluded.add(problem_key(row["question"]))
    rng = random.Random(20261104)
    calibration, evaluation = [], []
    for bucket in ("addition_small", "addition_large", "medium", "hard"):
        selected = []
        for _ in range(100000):
            if bucket.startswith("addition"):
                a, b = rng.randint(1, 20), rng.randint(1, 20)
                question, expected = f"{a} + {b}", a + b
            elif bucket == "medium":
                a, b = rng.randint(20, 99), rng.randint(10, 99)
                question, expected = f"{a} * {b}", a * b
            else:
                a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
                question, expected = f"({a} * {b}) - {c}", a * b - c
            key = problem_key(question)
            if key in excluded or router.bucket_for(question) != bucket:
                continue
            excluded.add(key)
            selected.append({"question": question, "expected": expected, "bucket": bucket})
            if len(selected) == calibration_size + evaluation_size:
                break
        else:
            raise ValueError(f"Not enough unseen problems for {bucket}")
        calibration.extend(selected[:calibration_size])
        evaluation.extend(selected[calibration_size:])
    # Fixed shuffle removes easy-to-hard timing bias without looking at model output.
    rng.shuffle(calibration)
    rng.shuffle(evaluation)
    return calibration, evaluation


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def summarize(rows):
    summaries = []
    for strategy in STRATEGIES:
        for bucket in ("all", "addition_small", "addition_large", "medium", "hard"):
            group = [r for r in rows if r["strategy"] == strategy
                     and (bucket == "all" or r["bucket"] == bucket)]
            if not group:
                continue
            answered = [r for r in group if r["status"] == "answered"]
            attempts = [a for r in group for a in r["attempts"]]
            correct = sum(r["correct"] for r in group)
            selected = Counter(r["model"] or "no_selection" for r in group)
            calls = Counter(a["model"] for a in attempts)
            eval_values = [a.get("eval_duration") for a in attempts]
            token_values = [a.get("output_tokens") for a in attempts]
            summaries.append({
                "strategy": strategy, "bucket": bucket, "requests": len(group),
                "answered": len(answered), "correct": correct,
                "accuracy": correct / len(group),
                "answered_accuracy": correct / len(answered) if answered else None,
                "answer_rate": len(answered) / len(group),
                "mean_latency": sum(r["elapsed_seconds"] for r in group) / len(group),
                "mean_answered_latency": sum(r["elapsed_seconds"] for r in answered) / len(answered) if answered else None,
                "total_seconds": sum(r["elapsed_seconds"] for r in group),
                "final_model_ratios": {m: selected[m] / len(group) for m in (router.SMALL, router.LARGE, "no_selection")},
                "api_calls": len(attempts), "model_calls": dict(calls),
                "model_call_ratios": {m: calls[m] / len(attempts) if attempts else 0 for m in (router.SMALL, router.LARGE)},
                "fallback_requests": sum(len(r["attempts"]) > 1 for r in group),
                "api_errors": sum(a["status"] == "error" for a in attempts),
                "format_errors": sum(a["status"] == "extraction_failed" for a in attempts),
                "output_tokens": sum(token_values) if all(v is not None for v in token_values) else None,
                "ollama_eval_seconds": sum(eval_values) if all(v is not None for v in eval_values) else None,
                "cost": None, "cost_status": "unmeasured",
            })
    return summaries


def paired_comparisons(rows):
    indexed = {(r["strategy"], r["question"]): r for r in rows}
    results = []
    for baseline in ("fixed", "legacy_auto", "fixed_prompt_control"):
        common = [(r, indexed[("improved", r["question"])]) for r in rows
                  if r["strategy"] == baseline and r["status"] == "answered"
                  and indexed[("improved", r["question"])]["status"] == "answered"]
        base_answered = {r["question"] for r in rows if r["strategy"] == baseline and r["status"] == "answered"}
        new_answered = {r["question"] for r in rows if r["strategy"] == "improved" and r["status"] == "answered"}
        bt = sum(a["elapsed_seconds"] for a, b in common)
        nt = sum(b["elapsed_seconds"] for a, b in common)
        results.append({
            "baseline": baseline, "common_answered": len(common),
            "lost_answers": len(base_answered - new_answered),
            "new_answers": len(new_answered - base_answered),
            "baseline_correct": sum(a["correct"] for a, b in common),
            "improved_correct": sum(b["correct"] for a, b in common),
            "baseline_total_seconds": bt, "improved_total_seconds": nt,
            "baseline_mean_seconds": bt / len(common) if common else None,
            "improved_mean_seconds": nt / len(common) if common else None,
            "same_coverage_speedup": bt / nt if nt else None,
            "scope": "common answered questions; includes every fallback attempt",
        })
    return results


def run_experiment(data_dir=ROOT / "data", min_quality=0.8):
    data_dir = Path(data_dir)
    # Everything existing is read-only and hashed to check preservation at the end.
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", data_dir) for p in folder.iterdir() if p.is_file()}
    calibration, evaluation = build_splits(data_dir)
    check_ollama([router.SMALL, router.LARGE])
    run_id = uuid4().hex
    prefix = data_dir / f"adaptive_{run_id}"
    write_json(Path(f"{prefix}_splits.json"), {
        "seed": 20261104, "min_quality": min_quality, "prompt_version": router.PROMPT_VERSION,
        "calibration": calibration, "evaluation": evaluation,
        "preserved_sha256": {str(p): digest for p, digest in preserved.items()},
    })
    # Warm up both prompts and models without using calibration or evaluation answers.
    for model in (router.SMALL, router.LARGE):
        for prompt in ("legacy", router.PROMPT_VERSION):
            router.generate(model, "1 + 1", prompt)
    calibration_rows = []
    calibration_start = time.perf_counter()
    for index, problem in enumerate(calibration):
        for model in ((router.SMALL, router.LARGE) if index % 2 == 0 else (router.LARGE, router.SMALL)):
            row = {**problem, "model": model, "answer": "", "wall_time": None,
                   "correct": False, "status": "error", "reason": ""}
            try:
                response = router.generate(model, problem["question"])
                row.update(answer=response["answer"], wall_time=response["wall_time"])
                row.update(validate_answer(response["answer"], problem["expected"]))
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            calibration_rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"Calibration {index+1}/{len(calibration)}", flush=True)
    calibration_seconds = time.perf_counter() - calibration_start
    # Consistent optional validation fields even for failed API calls.
    for row in calibration_rows:
        row.setdefault("predicted", None)
    write_csv(Path(f"{prefix}_calibration.csv"), calibration_rows)
    profile = router.make_profile(calibration_rows)
    profile["calibration_seconds"] = calibration_seconds
    profile["source"] = str(Path(f"{prefix}_calibration.csv"))
    profile_path = Path(f"{prefix}_profile.json")
    write_json(profile_path, profile)
    print(json.dumps(profile, ensure_ascii=False), flush=True)
    # Policy frozen here. Evaluation expected values never enter execute() or plan().
    rows = []
    evaluation_start = time.perf_counter()
    for index, problem in enumerate(evaluation):
        offset = index % len(STRATEGIES)
        for strategy in STRATEGIES[offset:] + STRATEGIES[:offset]:
            row = router.execute(problem["question"], profile, strategy, min_quality)
            predicted = validate_answer(row["answer"], problem["expected"])
            row.update(**problem, predicted=predicted["predicted"],
                       correct=row["status"] == "answered" and predicted["correct"])
            rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"Evaluation {index+1}/{len(evaluation)}", flush=True)
    evaluation_seconds = time.perf_counter() - evaluation_start
    summaries = summarize(rows)
    pairs = paired_comparisons(rows)
    write_csv(Path(f"{prefix}_evaluation.csv"), rows)
    write_csv(Path(f"{prefix}_summary.csv"), summaries)
    write_csv(Path(f"{prefix}_paired.csv"), pairs)
    write_json(Path(f"{prefix}_timing.json"), {
        "calibration_seconds": calibration_seconds, "evaluation_seconds": evaluation_seconds,
        "warmup_and_io_excluded": True, "threshold": min_quality,
    })
    for path, digest in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing file changed: {path}")
    for summary in summaries:
        if summary["bucket"] == "all":
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    for pair in pairs:
        print(json.dumps(pair, ensure_ascii=False), flush=True)
    print(f"Results: {prefix}_*.csv\nProfile: {profile_path}", flush=True)
    return prefix


def main():
    parser = argparse.ArgumentParser(description="독립 보정/평가 기반 fallback 라우팅")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--min-quality", type=float, default=0.8)
    args = parser.parse_args()
    try:
        if args.experiment:
            run_experiment(min_quality=args.min_quality)
        else:
            if args.profile is None:
                parser.error("--profile is required for interactive requests")
            with args.profile.open(encoding="utf-8") as file:
                profile = json.load(file)
            question = args.question if args.question is not None else input("질문: ")
            decision = router.plan(question, profile, args.min_quality)
            if decision["attempts"]:
                check_ollama({a["model"] for a in decision["attempts"]})
            print(json.dumps(router.execute(question, profile, min_quality=args.min_quality),
                             ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"실행 실패: {error}\n")


if __name__ == "__main__":
    main()

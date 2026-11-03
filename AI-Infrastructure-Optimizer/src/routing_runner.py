"""Interactive Ollama routing and paired comparison using measured routing data."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from uuid import uuid4

if __package__:
    from .answer_validator import validate_answer
    from .benchmark_v2 import ask, questions
    from .model_router import DEFAULT_BENCHMARK, load_benchmark, route_request
else:
    from answer_validator import validate_answer
    from benchmark_v2 import ask, questions
    from model_router import DEFAULT_BENCHMARK, load_benchmark, route_request

FIXED_MODEL = "qwen2.5:1.5b"


def check_ollama(models):
    """Read installed model tags only; never download or start a server."""
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        available = {m["name"] for m in json.load(response)["models"]}
    missing = set(models) - available
    if missing:
        raise RuntimeError(f"Ollama 모델을 사용할 수 없습니다: {', '.join(sorted(missing))}")


def execute_request(question, min_quality=0.8, max_latency=float("inf"),
                    benchmark_path=DEFAULT_BENCHMARK, fixed_model=None):
    """Generate only when eligible. Fixed baseline intentionally bypasses routing.

    Elapsed time includes routing and generation, including refusals and errors.
    Unknown ground truth in interactive mode is not marked correct or incorrect.
    """
    start = time.perf_counter()
    row = {"question": question, "model": "", "answer": "", "status": "error",
           "reason": "", "generation_seconds": None, "cost": None,
           "cost_status": "unmeasured"}
    try:
        if fixed_model is None:
            decision = route_request(question, min_quality, max_latency, benchmark_path)
            if not decision["suitable"]:
                row.update(status="rejected", reason=(
                    "최소 정확도 또는 응답 시간 조건을 만족하는 모델이 없습니다. "
                    + decision["reason"]
                ))
                return row
            row["model"] = decision["model"]
        else:
            row["model"] = fixed_model
        response = ask(row["model"], question)
        row.update(answer=response["answer"], status="generated",
                   generation_seconds=response["wall_time"])
    except Exception as error:
        row.update(status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        row["elapsed_seconds"] = time.perf_counter() - start
    return row


def summarize(rows):
    summaries = []
    for strategy in ("fixed", "auto"):
        strategy_rows = [r for r in rows if r["strategy"] == strategy]
        for difficulty in ("all", "easy", "medium", "hard"):
            group = [r for r in strategy_rows
                     if difficulty == "all" or r["difficulty"] == difficulty]
            if not group:
                continue
            generated = [r for r in group if r["generation_seconds"] is not None]
            correct = sum(r["correct"] for r in group)
            counts = Counter(r["model"] or "no_selection" for r in group)
            summaries.append({
                "strategy": strategy, "difficulty": difficulty, "requests": len(group),
                "generated": len(generated), "correct": correct,
                "accuracy": correct / len(group),
                "answered_accuracy": correct / len(generated) if generated else None,
                "rejected": sum(r["status"] == "rejected" for r in group),
                "errors": sum(r["status"] == "error" for r in group),
                "extraction_failed": sum(r["status"] == "extraction_failed" for r in group),
                "mean_request_seconds": sum(r["elapsed_seconds"] for r in group) / len(group),
                "mean_generation_seconds": (
                    sum(r["generation_seconds"] for r in generated) / len(generated)
                    if generated else None
                ),
                "total_processing_seconds": sum(r["elapsed_seconds"] for r in group),
                "model_selection_ratios": json.dumps(
                    {m: n / len(group) for m, n in sorted(counts.items())}, sort_keys=True
                ),
                "cost_status": "unmeasured",
            })
    return summaries


def compare(min_quality=0.8, max_latency=float("inf"),
            benchmark_path=DEFAULT_BENCHMARK, output_dir=None, problem_set=None):
    # Fail before creating artifacts when server, data, or models are unavailable.
    models = load_benchmark(benchmark_path)
    route_request("4 + 1", min_quality, max_latency, benchmark_path)
    check_ollama(set(models) | {FIXED_MODEL})
    for model in sorted(set(models) | {FIXED_MODEL}):
        ask(model, "1 + 1")  # Unmeasured warmup shared by both strategies.
    problems = questions if problem_set is None else problem_set
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    detail_path = output_dir / f"routing_comparison_{run_id}.csv"
    summary_path = output_dir / f"routing_summary_{run_id}.csv"
    source_hash = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
    rows = []
    experiment_start = time.perf_counter()
    # Alternate order to reduce the systematic advantage of always running second.
    for index, (difficulty, question, expected) in enumerate(problems):
        order = ("fixed", "auto") if index % 2 == 0 else ("auto", "fixed")
        for strategy in order:
            row = execute_request(question, min_quality, max_latency, benchmark_path,
                                  FIXED_MODEL if strategy == "fixed" else None)
            row.update(strategy=strategy, difficulty=difficulty, expected=expected,
                       predicted=None, correct=False, min_quality=min_quality,
                       max_latency=max_latency, benchmark_path=str(benchmark_path),
                       benchmark_sha256=source_hash)
            if row["status"] == "generated":
                validation = validate_answer(row["answer"], expected)
                row.update(validation)
            rows.append(row)
            print(f"{strategy:5} | {difficulty:6} | {row['model'] or '-':14} | "
                  f"{row['status']} | {row['elapsed_seconds']:.4f}s", flush=True)
    experiment_seconds = time.perf_counter() - experiment_start
    summaries = summarize(rows)
    for summary in summaries:
        summary["experiment_seconds"] = experiment_seconds
    for path, records in ((detail_path, rows), (summary_path, summaries)):
        if records:
            with path.open("x", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))
    print(f"상세: {detail_path}\n요약: {summary_path}")
    return detail_path, summary_path, summaries


def main():
    parser = argparse.ArgumentParser(description="실측 데이터 기반 Ollama 자동 라우팅")
    parser.add_argument("--compare", action="store_true", help="고정 1.5B 방식과 30문제 비교")
    parser.add_argument("--question", help="자동 라우팅할 산술 질문 (생략하면 직접 입력)")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--max-latency", type=float, default=float("inf"))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    try:
        if args.compare:
            compare(args.min_quality, args.max_latency, args.benchmark)
        else:
            question = args.question if args.question is not None else input("질문: ")
            decision = route_request(question, args.min_quality, args.max_latency, args.benchmark)
            if decision["suitable"]:
                check_ollama([decision["model"]])
            result = execute_request(question, args.min_quality, args.max_latency, args.benchmark)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"실행 실패: {error}\n")


if __name__ == "__main__":
    main()

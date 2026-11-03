
import csv
import json
import random
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

if __package__:
    from .answer_validator import validate_answer
else:
    from answer_validator import validate_answer

MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b"]
URL = "http://localhost:11434/api/generate"

random.seed(42)

# 30 questions: 10 easy, 10 medium, 10 hard
questions = []

for i in range(10):
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    questions.append(("easy", f"{a} + {b}", a + b))

for i in range(10):
    a = random.randint(20, 99)
    b = random.randint(10, 99)
    questions.append(("medium", f"{a} * {b}", a * b))

for i in range(10):
    a = random.randint(100, 999)
    b = random.randint(10, 99)
    c = random.randint(10, 99)
    questions.append(
        ("hard", f"({a} * {b}) - {c}", a * b - c)
    )


def ask(model, question):
    payload = json.dumps({
        "model": model,
        "prompt": (
            f"Calculate: {question}. "
            "Return your answer in exactly this format: FINAL_ANSWER: 123 (replace 123 with your calculated integer)."
        ),
        "stream": False,
        "options": {"temperature": 0},
    }).encode()

    request = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    wall_time = time.perf_counter() - start

    return {
        "answer": result["response"].strip(),
        "wall_time": wall_time,
        "total_duration": result.get("total_duration", 0) / 1e9,
        "load_duration": result.get("load_duration", 0) / 1e9,
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "output_tokens": result.get("eval_count", 0),
        "eval_duration": result.get("eval_duration", 0) / 1e9,
    }


def main():
    results = []

    # Warm up both models before measurement
    for model in MODELS:
        print(f"Warming up {model}...")
        ask(model, "1 + 1")

    for difficulty, question, expected in questions:
        for model in MODELS:
            try:
                r = ask(model, question)
                validation = validate_answer(r["answer"], expected)

                row = {
                    "difficulty": difficulty,
                    "question": question,
                    "model": model,
                    "expected": expected,
                    "answer": r["answer"],
                    "correct": validation["correct"],
                    "predicted": validation["predicted"],
                    "status": validation["status"],
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in r.items() if k != "answer"},
                }

                results.append(row)

                print(
                    f"{difficulty:6} | {model:14} | "
                    f"{validation['status']} | "
                    f"{r['wall_time']:.3f}s"
                )

            except Exception as e:
                print(f"ERROR: {model}: {e}")

    path = Path("data") / f"benchmark_v2_{uuid4().hex}.csv"
    path.parent.mkdir(exist_ok=True)

    if results:
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("\n===== RESULTS =====")

    for model in MODELS:
        for difficulty in ["easy", "medium", "hard"]:
            rows = [
                r for r in results
                if r["model"] == model
                and r["difficulty"] == difficulty
            ]

            if not rows:
                continue

            accuracy = sum(r["correct"] for r in rows) / len(rows) * 100
            avg_time = sum(r["wall_time"] for r in rows) / len(rows)
            counts = {
                status: sum(r["status"] == status for r in rows)
                for status in ("correct", "incorrect", "extraction_failed")
            }

            print(
                f"{model} | {difficulty} | "
                f"Accuracy: {accuracy:.1f}% | "
                f"Avg time: {avg_time:.3f}s | "
                f"Correct: {counts['correct']} | "
                f"Incorrect: {counts['incorrect']} | "
                f"Extraction failed: {counts['extraction_failed']}"
            )

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

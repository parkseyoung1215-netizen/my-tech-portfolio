
import csv
import json
import time
import urllib.request
from pathlib import Path

MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b"]

QUESTIONS = [
    ("What is 15 multiplied by 8? Answer with only the number.", "120"),
    ("What is 144 divided by 12? Answer with only the number.", "12"),
    ("What is 17 plus 28? Answer with only the number.", "45"),
    ("What is 9 squared? Answer with only the number.", "81"),
    ("What is 100 minus 37? Answer with only the number.", "63"),
]


def ask_model(model, question):
    payload = json.dumps({
        "model": model,
        "prompt": question,
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    elapsed = time.perf_counter() - start

    return {
        "answer": result["response"].strip(),
        "seconds": round(elapsed, 3),
        "tokens": result.get("eval_count", 0),
        "generation_seconds": result.get("eval_duration", 0) / 1e9,
    }


def main():
    results = []

    for model in MODELS:
        print(f"\n===== Testing {model} =====")

        for question, expected in QUESTIONS:
            try:
                result = ask_model(model, question)
                correct = result["answer"] == expected

                row = {
                    "model": model,
                    "question": question,
                    "expected": expected,
                    "answer": result["answer"],
                    "correct": correct,
                    "seconds": result["seconds"],
                    "output_tokens": result["tokens"],
                    "generation_seconds": result["generation_seconds"],
                }
                results.append(row)

                print(
                    f"{'PASS' if correct else 'FAIL'} | "
                    f"{result['seconds']}s | "
                    f"{result['answer']}"
                )

            except Exception as error:
                print(f"ERROR: {error}")

    output = Path("data/benchmark_results.csv")
    output.parent.mkdir(exist_ok=True)

    if results:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("\n===== SUMMARY =====")

    for model in MODELS:
        rows = [r for r in results if r["model"] == model]

        if not rows:
            print(f"{model}: No successful requests")
            continue

        accuracy = sum(r["correct"] for r in rows) / len(rows) * 100
        avg_time = sum(r["seconds"] for r in rows) / len(rows)

        print(
            f"{model}: Accuracy {accuracy:.1f}% "
            f"| Average time {avg_time:.3f}s"
        )

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()

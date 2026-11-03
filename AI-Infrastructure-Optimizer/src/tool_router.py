"""Tool-first routing, direct calculator output and original Adaptive fallback."""

import time

if __package__:
    from .safe_calculator import parse_expression, calculate, ToolDeclined
    from .adaptive_router import execute as execute_adaptive
else:
    from safe_calculator import parse_expression, calculate, ToolDeclined
    from adaptive_router import execute as execute_adaptive

VERSION = "safe-integer-tool-routing-v1"


def execute(question, profile):
    start = time.perf_counter()
    row = {"question": question, "strategy": "tool", "model": "", "answer": "",
           "status": "rejected", "reason": "", "attempts": [], "source": "none",
           "tool_checks": 1, "tool_calls": 0, "tool_seconds": 0.0,
           "model_fallback": False, "fallback_reason": "", "fallback_seconds": 0.0,
           "tool_events": [], "cost": None, "cost_status": "unmeasured"}
    try:
        try:
            tree = parse_expression(question)
            row["tool_calls"] = 1
            value = calculate(tree)
            row.update(status="answered", answer=f"FINAL_ANSWER: {value}",
                       source="tool", reason="direct_calculator_result")
            row["tool_events"].append({"status": "answered", "value": value, "reason": "admitted"})
            return row
        except ToolDeclined as error:
            row["fallback_reason"] = error.reason
            row["tool_events"].append({"status": "declined", "reason": error.reason})
        except Exception as error:
            row["fallback_reason"] = f"tool_error:{type(error).__name__}"
            row["tool_events"].append({"status": "error", "reason": row["fallback_reason"], "error": str(error)})
        finally:
            row["tool_seconds"] = time.perf_counter() - start
        row["model_fallback"] = True
        row["source"] = "model_fallback"
        fallback_start = time.perf_counter()
        try:
            result = execute_adaptive(question, profile, "improved", 0.8)
            row.update(model=result["model"], answer=result["answer"], status=result["status"],
                       attempts=result["attempts"], reason=result["reason"])
        except Exception as error:
            row.update(status="error", reason=f"model_fallback_error:{type(error).__name__}: {error}")
        finally:
            row["fallback_seconds"] = time.perf_counter() - fallback_start
        return row
    finally:
        row["elapsed_seconds"] = time.perf_counter() - start


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="명확한 정수 산술식은 계산기로 직접 처리")
    parser.add_argument("--question")
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parent.parent /
                        "data/adaptive_a8a1ef3394824a7cb65faf5707152986_profile.json")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("질문: ")
    # Loading a calibration profile does not contact or start Ollama.
    with args.profile.open(encoding="utf-8") as file:
        profile = json.load(file)
    print(json.dumps(execute(question, profile), ensure_ascii=False, indent=2))

"""Generation and bounded fallback orchestration, independent of scoring labels."""

import json
import time
import urllib.request

if __package__:
    from .adaptive_router import SMALL, LARGE, generate
    from .request_verifier import verify_response
else:
    from adaptive_router import SMALL, LARGE, generate
    from request_verifier import verify_response

VERSION = "verified-coverage-v1"


def generate_answer(model, question, retry=False):
    if not retry:
        return generate(model, question)
    # Feedback has no corrected integer, reference label or computed intermediates.
    prompt = (
        f"Calculate: {question}.\n"
        "A previous attempt failed validation. Recalculate independently and carefully. "
        "For multiplication, split an operand into hundreds, tens and units and combine "
        "the partial products. For a composite expression, perform the subtraction last. "
        "Return exactly one line starting with FINAL_ANSWER: followed by the calculated "
        "integer. Do not include explanations or examples."
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0}}).encode("utf-8")
    request = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return {"answer": result["response"].strip(), "wall_time": time.perf_counter() - start,
            "output_tokens": result.get("eval_count"),
            "eval_duration": result["eval_duration"] / 1e9 if "eval_duration" in result else None}


def execute(question, policy):
    start = time.perf_counter()
    row = {"question": question, "strategy": "coverage", "model": "", "answer": "",
           "status": "rejected", "reason": "", "attempts": [],
           "cost": None, "cost_status": "unmeasured", "routing_seconds": 0.0,
           "validation_seconds": 0.0, "fallback_seconds": 0.0,
           "large_handoff": False, "large_retry": False}
    try:
        decision = policy.select(question)
        row["routing_seconds"] = time.perf_counter() - start
        row["selection_reason"] = decision["reason"]
        if decision["model"] is None:
            row["reason"] = decision["reason"]
            return row
        candidates = ([(SMALL, False)] if decision["model"] == SMALL else []) + [(LARGE, False), (LARGE, True)]
        for model, retry in candidates:
            attempt_start = time.perf_counter()
            attempt = {"model": model, "retry": retry, "status": "error", "error": "",
                       "reason": "api_error", "validation_seconds": 0.0}
            if model == LARGE:
                row["large_handoff"] = True
                row["large_retry"] |= retry
            try:
                result = generate_answer(model, question, retry)
                attempt.update(result)
                verify_start = time.perf_counter()
                validation = verify_response(question, result["answer"])
                attempt["validation_seconds"] = time.perf_counter() - verify_start
                row["validation_seconds"] += attempt["validation_seconds"]
                attempt.update(status="answered" if validation["valid"] else "validation_failed",
                               reason=validation["reason"], predicted=validation["predicted"])
            except Exception as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
            attempt["elapsed_seconds"] = time.perf_counter() - attempt_start
            row["attempts"].append(attempt)
            if len(row["attempts"]) > 1:
                row["fallback_seconds"] += attempt["elapsed_seconds"]
            if attempt["status"] == "answered":
                row.update(model=model, answer=attempt["answer"], status="answered", reason="runtime_verified")
                break
        else:
            row["reason"] = "all_candidates_failed: " + "; ".join(
                f"{a['model']}{':retry' if a['retry'] else ''}={a['reason']}" for a in row["attempts"]
            )
    finally:
        row["elapsed_seconds"] = time.perf_counter() - start
    return row


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    if __package__:
        from .coverage_policy import CoveragePolicy
        from .routing_runner import check_ollama
    else:
        from coverage_policy import CoveragePolicy
        from routing_runner import check_ollama
    parser = argparse.ArgumentParser(description="산술 검증 기반 Adaptive 실행")
    parser.add_argument("--question")
    parser.add_argument("--profile", type=Path, default=(Path(__file__).resolve().parent.parent /
        "data/adaptive_a8a1ef3394824a7cb65faf5707152986_profile.json"))
    args = parser.parse_args()
    try:
        with args.profile.open(encoding="utf-8") as file:
            policy = CoveragePolicy(json.load(file))
        question = args.question if args.question is not None else input("질문: ")
        if policy.select(question)["model"] is not None:
            check_ollama([SMALL, LARGE])
        print(json.dumps(execute(question, policy), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"실행 실패: {error}\n")

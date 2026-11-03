"""Conservative, calibrated small-first Ollama routing, without answer oracles."""

import json
import math
import re
import time
import urllib.request

if __package__:
    from .answer_validator import extract_answer
    from .benchmark_v2 import ask as legacy_ask
    from .model_router import route_request, estimate_difficulty
else:
    from answer_validator import extract_answer
    from benchmark_v2 import ask as legacy_ask
    from model_router import route_request, estimate_difficulty

SMALL = "qwen2.5:0.5b"
LARGE = "qwen2.5:1.5b"
PROMPT_VERSION = "integer-no-example-v1"
MIN_SAMPLES = 20


def bucket_for(question):
    """Features are syntax and operand sizes only, never a computed answer."""
    difficulty = estimate_difficulty(question)
    if difficulty == "easy":
        operands = list(map(int, re.findall(r"[0-9]+", question)))
        return "addition_small" if max(operands) <= 10 else "addition_large"
    return difficulty


def lower_accuracy(correct, samples):
    """95% Wilson lower bound. Small samples cannot imply perfect confidence."""
    if samples <= 0 or not 0 <= correct <= samples:
        return 0.0
    z = 1.96
    p = correct / samples
    return (p + z*z / (2*samples) - z * math.sqrt(
        p*(1-p)/samples + z*z/(4*samples*samples)
    )) / (1 + z*z/samples)


def generate(model, question, prompt_version=PROMPT_VERSION):
    if prompt_version == "legacy":
        return legacy_ask(model, question)
    if prompt_version != PROMPT_VERSION:
        raise ValueError("Unsupported prompt version")
    payload = json.dumps({
        "model": model,
        "prompt": (
            f"Calculate: {question}.\n"
            "Return exactly one line starting with FINAL_ANSWER: followed by "
            "the calculated integer. Do not include explanations or examples."
        ),
        "stream": False, "options": {"temperature": 0},
    }).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return {
        "answer": result["response"].strip(),
        "wall_time": time.perf_counter() - start,
        "output_tokens": result.get("eval_count"),
        "eval_duration": result["eval_duration"] / 1e9 if "eval_duration" in result else None,
    }


def make_profile(rows):
    """Use calibration rows only. API/format failures count as incorrect."""
    groups = {}
    for row in rows:
        groups.setdefault(row["bucket"], {}).setdefault(row["model"], []).append(row)
    return {
        "prompt_version": PROMPT_VERSION, "min_samples": MIN_SAMPLES,
        "buckets": {
            bucket: {
                model: {"samples": len(rs), "correct": sum(r["correct"] for r in rs),
                        "accuracy": sum(r["correct"] for r in rs) / len(rs),
                        "lower_accuracy": lower_accuracy(sum(r["correct"] for r in rs), len(rs))}
                for model, rs in models.items()
            } for bucket, models in groups.items()
        },
    }


def plan(question, profile, min_quality=0.8):
    if not math.isfinite(min_quality) or not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between 0 and 1")
    if profile.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Calibration prompt version mismatch")
    bucket = bucket_for(question)
    attempts, reasons = [], []
    for model in (SMALL, LARGE):
        metric = profile.get("buckets", {}).get(bucket, {}).get(model)
        if (metric and metric["samples"] >= MIN_SAMPLES
                and metric["lower_accuracy"] >= min_quality):
            attempts.append({"model": model, "prompt_version": PROMPT_VERSION})
        else:
            reasons.append(f"{model}: insufficient calibration confidence for {bucket}")
    # Keep previously supported requests using the unchanged, measured legacy prompt.
    legacy = route_request(question, min_quality=min_quality)
    if legacy["suitable"]:
        attempts.append({"model": legacy["model"], "prompt_version": "legacy"})
    return {"bucket": bucket, "attempts": attempts, "reasons": reasons,
            "reason": "Eligible calibrated models, then supported legacy fallback"
            if attempts else "Neither model is sufficiently reliable for this request"}


def execute(question, profile=None, strategy="improved", min_quality=0.8):
    """No expected answer parameter: fallback uses eligibility, errors and format only."""
    start = time.perf_counter()
    row = {"question": question, "strategy": strategy, "model": "", "answer": "",
           "status": "rejected", "reason": "", "attempts": [],
           "cost": None, "cost_status": "unmeasured"}
    try:
        if strategy in ("fixed", "fixed_prompt_control"):
            candidates = [{"model": LARGE, "prompt_version":
                           "legacy" if strategy == "fixed" else PROMPT_VERSION}]
        elif strategy == "legacy_auto":
            decision = route_request(question, min_quality=min_quality)
            candidates = ([{"model": decision["model"], "prompt_version": "legacy"}]
                          if decision["suitable"] else [])
            row["reason"] = decision["reason"]
        elif strategy == "improved":
            decision = plan(question, profile, min_quality)
            candidates = decision["attempts"]
            row["reason"] = decision["reason"] + "; " + "; ".join(decision["reasons"])
        else:
            raise ValueError("Unknown strategy")
        for candidate in candidates:
            attempt = {**candidate, "status": "error", "error": ""}
            attempt_start = time.perf_counter()
            try:
                response = generate(candidate["model"], question, candidate["prompt_version"])
                attempt.update(response)
                attempt["status"] = (
                    "answered" if extract_answer(response["answer"]) is not None
                    else "extraction_failed"
                )
            except Exception as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
            attempt["elapsed_seconds"] = time.perf_counter() - attempt_start
            row["attempts"].append(attempt)
            if attempt["status"] == "answered":
                row.update(model=candidate["model"], answer=response["answer"], status="answered")
                break
        else:
            if candidates:
                row["reason"] += "; all eligible attempts failed or returned invalid FINAL_ANSWER"
    finally:
        row["elapsed_seconds"] = time.perf_counter() - start
    return row

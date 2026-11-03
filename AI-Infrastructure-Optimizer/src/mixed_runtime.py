"""Mixed-task adapter. No labels, category metadata or evaluation data access."""

import json
import re
import time
import urllib.request

if __package__:
    from . import adaptive_router as old
    from .coverage_runner import execute as execute_coverage
    from .tool_router import execute as execute_tool
    from .safe_calculator import parse_expression, ToolDeclined
else:
    import adaptive_router as old
    from coverage_runner import execute as execute_coverage
    from tool_router import execute as execute_tool
    from safe_calculator import parse_expression, ToolDeclined


def infer_category(question):
    try:
        parse_expression(question)
        return "arithmetic"
    except ToolDeclined:
        pass
    if "```python" in question.lower():
        return "coding"
    if question.lower().startswith("read the passage:"):
        return "comprehension"
    if question.lower().startswith("reason "):
        return "logic"
    if question.lower().startswith(("what ", "which ", "how many ")):
        return "knowledge"
    return "unknown"


def extract_final(text):
    declarations = [line for line in text.splitlines()
                    if re.match(r"^[ \t]*FINAL_ANSWER\b", line, re.I)]
    if len(declarations) != 1:
        return None
    match = re.fullmatch(r"[ \t]*FINAL_ANSWER[ \t]*:[ \t]*([A-D]|[+-]?[0-9]+)[ \t]*",
                         declarations[0], re.I)
    if not match:
        return None
    answer = match.group(1).upper()
    return str(int(answer)) if answer.lstrip("+-").isdigit() else answer


def generate_general(question):
    payload = json.dumps({"model": old.LARGE, "prompt": question +
        "\nChoose the best option. Return exactly one line: FINAL_ANSWER: followed by "
        "the option letter A, B, C or D. Do not include explanations.",
        "stream": False, "options": {"temperature": 0, "num_predict": 64}}).encode("utf-8")
    request = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return {"answer": result["response"].strip(), "wall_time": time.perf_counter()-start,
            "output_tokens": result.get("eval_count"),
            "eval_duration": result["eval_duration"] / 1e9 if "eval_duration" in result else None,
            "done_reason": result.get("done_reason")}


def execute(question, strategy, profile, coverage_policy):
    start = time.perf_counter()
    inferred = infer_category(question)
    if inferred == "arithmetic":
        if strategy == "fixed":
            row = old.execute(question, profile, "fixed", 0.8)
        elif strategy == "adaptive":
            row = old.execute(question, profile, "improved", 0.8)
        elif strategy == "coverage":
            row = execute_coverage(question, coverage_policy)
        elif strategy == "tool":
            row = execute_tool(question, profile)
        else:
            raise ValueError("Unknown strategy")
        row.setdefault("source", "model" if row["attempts"] else "none")
        row.setdefault("tool_calls", 0)
        row["general_adapter"] = False
    else:
        if strategy not in ("fixed", "adaptive", "coverage", "tool"):
            raise ValueError("Unknown strategy")
        row = {"question": question, "model": old.LARGE, "answer": "", "status": "error",
               "reason": "", "attempts": [], "source": "model", "tool_calls": 0,
               "general_adapter": True}
        attempt_start = time.perf_counter()
        attempt = {"model": old.LARGE, "status": "error", "error": ""}
        try:
            result = generate_general(question)
            attempt.update(result)
            valid = extract_final(result["answer"]) in ("A", "B", "C", "D")
            attempt["status"] = "answered" if valid else "extraction_failed"
            row.update(answer=result["answer"], status="answered" if valid else "rejected",
                       reason="common_1.5B_general_adapter" if valid else "extraction_failed")
        except Exception as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
            row["reason"] = attempt["error"]
        attempt["elapsed_seconds"] = time.perf_counter() - attempt_start
        row["attempts"].append(attempt)
    # Equal wrapper boundary includes classification, rejected routes and fallbacks.
    result = {"question": question, "strategy": strategy, "model": row["model"],
              "answer": row["answer"], "status": row["status"], "reason": row["reason"],
              "attempts": row["attempts"], "source": row["source"], "tool_calls": row["tool_calls"],
              "general_adapter": row["general_adapter"], "inferred_category": inferred,
              "elapsed_seconds": time.perf_counter()-start, "cost": None, "cost_status": "unmeasured"}
    result["runtime_details"] = {key: value for key, value in row.items() if key not in result}
    return result

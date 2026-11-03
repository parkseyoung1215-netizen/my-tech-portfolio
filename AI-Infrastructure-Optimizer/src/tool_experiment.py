"""Live four-way comparison on the unchanged Coverage 200/100 datasets."""

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
    from . import adaptive_router as adaptive
    from .coverage_policy import CoveragePolicy
    from .coverage_runner import execute as execute_coverage
    from .tool_router import execute as execute_tool, VERSION
    from .verify_routing200 import ROOT, PROFILE, reference_answer, audit_extract, problem_key
    from .adaptive_experiment import write_json
    from .answer_validator import validate_answer
    from .routing_runner import check_ollama
else:
    import adaptive_router as adaptive
    from coverage_policy import CoveragePolicy
    from coverage_runner import execute as execute_coverage
    from tool_router import execute as execute_tool, VERSION
    from verify_routing200 import ROOT, PROFILE, reference_answer, audit_extract, problem_key
    from adaptive_experiment import write_json
    from answer_validator import validate_answer
    from routing_runner import check_ollama

SOURCE_MANIFEST = ROOT / "data/coverage_951686cca01344ed8b732fc1e66d7cab_manifest.json"
STRATEGIES = ("fixed", "adaptive", "coverage", "tool")
LABELS = {"fixed": "고정 1.5B", "adaptive": "Adaptive", "coverage": "Coverage", "tool": "Tool-Assisted"}


def load_datasets(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = source["datasets"]
    for name, count in (("replay200", 200), ("unseen100", 100)):
        problems = datasets[name]
        if len(problems) != count or len({problem_key(p["question"]) for p in problems}) != count:
            raise ValueError("Dataset size or uniqueness mismatch")
        if any(p["expected"] != reference_answer(p["question"]) for p in problems):
            raise ValueError("Reference label mismatch")
    if ({problem_key(p["question"]) for p in datasets["replay200"]}
            & {problem_key(p["question"]) for p in datasets["unseen100"]}):
        raise ValueError("Dataset overlap")
    return datasets


def write_records(path, rows, fields=None):
    fields = list(rows[0]) if rows else fields
    with Path(path).open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def execute_strategy(strategy, question, profile, policy):
    if strategy == "tool":
        row = execute_tool(question, profile)
    elif strategy == "coverage":
        row = execute_coverage(question, policy)
    elif strategy in ("fixed", "adaptive"):
        row = adaptive.execute(question, profile, "fixed" if strategy == "fixed" else "improved", 0.8)
    else:
        raise ValueError("Unknown strategy")
    # Normalize schema without changing existing runtime code or inventing sub-timings.
    row.update(strategy=strategy)
    row.setdefault("source", "model" if row["attempts"] else "none")
    for name, default in {
        "tool_checks": 0, "tool_calls": 0, "tool_seconds": None, "model_fallback": False,
        "fallback_reason": "", "fallback_seconds": None, "tool_events": [],
        "routing_seconds": None, "validation_seconds": None, "large_handoff": None,
        "large_retry": None, "selection_reason": "original_policy",
    }.items():
        row.setdefault(name, default)
    row["verification_checks"] = sum("answer" in a for a in row["attempts"]) if strategy == "coverage" else 0
    row["verification_arithmetic_calls"] = sum(a.get("reason") in ("verified", "arithmetic_mismatch")
                                                for a in row["attempts"]) if strategy == "coverage" else 0
    return row


def summarize(rows):
    summaries = []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        for strategy in STRATEGIES:
            for difficulty in ("all", "easy", "medium", "hard"):
                group = [r for r in rows if r["dataset"] == dataset and r["strategy"] == strategy
                         and (difficulty == "all" or r["difficulty"] == difficulty)]
                if not group:
                    continue
                answered = [r for r in group if r["status"] == "answered"]
                correct = sum(r["correct"] for r in group)
                attempts = [a for r in group for a in r["attempts"]]
                summaries.append({
                    "dataset": dataset, "strategy": strategy, "difficulty": difficulty,
                    "requests": len(group), "answered": len(answered), "correct": correct,
                    "accuracy": correct / len(group), "answer_rate": len(answered) / len(group),
                    "answered_accuracy": correct / len(answered) if answered else None,
                    "mean_latency_seconds": sum(r["elapsed_seconds"] for r in group) / len(group),
                    "total_seconds": sum(r["elapsed_seconds"] for r in group),
                    "small_calls": sum(a["model"] == adaptive.SMALL for a in attempts),
                    "large_calls": sum(a["model"] == adaptive.LARGE for a in attempts),
                    "model_calls": len(attempts),
                    "direct_calculator_calls": sum(r["tool_calls"] for r in group),
                    "calculator_checks": sum(r["tool_checks"] for r in group),
                    "verification_checks": sum(r["verification_checks"] for r in group),
                    "verification_arithmetic_calls": sum(r["verification_arithmetic_calls"] for r in group),
                    "tool_fallback_requests": sum(r["model_fallback"] for r in group),
                    "tool_misapplications": sum(r["tool_misapplication"] for r in group),
                    "extraction_audit_mismatches": sum(r["audit_mismatches"] for r in group),
                    "api_errors": sum(a["status"] == "error" for a in attempts),
                    "cost": None, "cost_status": "unmeasured",
                })
    return summaries


def comparisons(rows, summaries):
    index = {(r["dataset"], r["strategy"], r["question"]): r for r in rows}
    aggregate = {(s["dataset"], s["strategy"]): s for s in summaries if s["difficulty"] == "all"}
    result = []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        tool = aggregate[(dataset, "tool")]
        for baseline in STRATEGIES[:-1]:
            base = aggregate[(dataset, baseline)]
            original = [r for r in rows if r["dataset"] == dataset and r["strategy"] == baseline]
            common = [(r, index[(dataset, "tool", r["question"])]) for r in original
                      if r["status"] == "answered" and index[(dataset, "tool", r["question"])]["status"] == "answered"]
            bt = sum(a["elapsed_seconds"] for a, b in common)
            tt = sum(b["elapsed_seconds"] for a, b in common)
            result.append({
                "dataset": dataset, "baseline": baseline,
                "accuracy_strictly_improved": tool["accuracy"] > base["accuracy"],
                "answer_rate_strictly_improved": tool["answer_rate"] > base["answer_rate"],
                "success_both_improved": tool["accuracy"] > base["accuracy"] and tool["answer_rate"] > base["answer_rate"],
                "common_answered": len(common), "baseline_common_correct": sum(a["correct"] for a, b in common),
                "tool_common_correct": sum(b["correct"] for a, b in common),
                "baseline_common_seconds": bt, "tool_common_seconds": tt,
                "common_speed_ratio": bt / tt if tt else None,
                "lost_answers": sum(r["status"] == "answered" and index[(dataset, "tool", r["question"])]["status"] != "answered" for r in original),
                "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in common),
            })
    return result


def safety_cases():
    return [
        {"id": "precedence", "question": "2 + 3 * 4", "expected_tool": True, "value": 14},
        {"id": "signed_parentheses", "question": "-(2 + 3) * 4", "expected_tool": True, "value": -20},
        {"id": "natural_language", "question": "Explain AI", "expected_tool": False, "value": None},
        {"id": "text_with_expression", "question": "Calculate 2 + 3", "expected_tool": False, "value": None},
        {"id": "units", "question": "2 + 3 people", "expected_tool": False, "value": None},
        {"id": "function_call", "question": "__import__('os')", "expected_tool": False, "value": None},
        {"id": "power", "question": "2 ** 1000000", "expected_tool": False, "value": None},
        {"id": "division", "question": "1 / 0", "expected_tool": False, "value": None},
        {"id": "decimal", "question": "1.5 + 2", "expected_tool": False, "value": None},
        {"id": "plain_number", "question": "42", "expected_tool": False, "value": None},
        {"id": "leading_zero_model_fallback", "question": "0002 + 0003", "expected_tool": False, "value": 5},
        {"id": "depth_limit", "question": "-" * 20 + "(1 + 2)", "expected_tool": False, "value": None},
        {"id": "result_limit", "question": " * ".join(["999999999999"] * 4), "expected_tool": False, "value": None},
        {"id": "length_limit", "question": "1 + " * 100 + "1", "expected_tool": False, "value": None},
    ]


def make_report(summaries, pairs, probes, metadata):
    def pct(v):
        return f"{v:.2%}" if v is not None else "해당 없음"
    lines = ["# Tool-Assisted Routing 실제 비교", "", f"실행 ID: `{metadata['run_id']}`.", "",
             "기존 고정 1.5B·Adaptive·Coverage 코드는 변경하지 않고 같은 실행에서 실제 재측정했습니다. "
             "Coverage 실험의 기존 200문제와 당시 신규 100문제를 그대로 사용합니다. "
             "이번에 처음 만든 홀드아웃은 아니며 모델 재보정·임계값 변경·평가 정답의 런타임 전달은 없습니다.", "",
             "도구는 입력 전체가 명확한 ASCII 정수 +,-,* 및 괄호 수식인 경우에만 직접 반환합니다. "
             "AST 노드 허용 목록과 길이·깊이·노드·리터럴·중간 결과 제한을 사용하며 Python 코드를 실행하지 않습니다. "
             "계산값은 모델로 재생성하지 않습니다. 그 외 요청은 원래 Adaptive 모델 라우터로 전달하며, "
             "그 라우터가 적합한 모델을 찾지 못하면 거절할 수 있습니다."]
    for dataset in ("replay200", "unseen100"):
        lines += ["", f"## {dataset}", "",
                  "| 방식 | 전체 정확도 | 답변 비율 | 답변 정확도 | 평균 지연(ms) | 처리 시간 합계(초) | 0.5B 호출 | 1.5B 호출 | 직접 계산 호출 | 검증용 계산 호출 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in summaries:
            if s["dataset"] == dataset and s["difficulty"] == "all":
                lines.append(f"| {LABELS[s['strategy']]} | {pct(s['accuracy'])} | {pct(s['answer_rate'])} | "
                             f"{pct(s['answered_accuracy'])} | {s['mean_latency_seconds']*1000:.6f} | {s['total_seconds']:.6f} | "
                             f"{s['small_calls']} | {s['large_calls']} | {s['direct_calculator_calls']} | {s['verification_arithmetic_calls']} |")
    lines += ["", "평균 지연과 처리 시간 합계에는 선택·계산·검증·실패·거절·fallback이 포함됩니다. "
              "워밍업과 사후 채점·파일 저장은 제외합니다. 호출은 재시도를 포함한 실제 횟수입니다. "
              "Coverage의 기존 산술 검증도 계산을 수행하므로 직접 답변 도구와 별도 열로 기록했습니다. "
              "검증용 계산 열은 추출 실패로 계산까지 진행하지 못한 검증을 제외합니다.", "",
              "## 성공 판정 및 같은 답변 범위의 속도", "",
              "성공은 각 세트·각 기준선 대비 **전체 정확도와 답변 비율이 모두 엄격하게 증가한 경우에만** 판정합니다.", "",
              "| 세트 | 기준선 | 정확도 증가 | 답변 비율 증가 | 성공 | 공통 답변 | 같은 범위 시간: 기준선 / 도구(초) | 잃은 답변 |",
              "|---|---|---|---|---|---:|---|---:|"]
    for p in pairs:
        lines.append(f"| {p['dataset']} | {LABELS[p['baseline']]} | {p['accuracy_strictly_improved']} | "
                     f"{p['answer_rate_strictly_improved']} | {p['success_both_improved']} | {p['common_answered']} | "
                     f"{p['baseline_common_seconds']:.6f} / {p['tool_common_seconds']:.6f} | {p['lost_answers']} |")
    success = all(p["success_both_improved"] for p in pairs)
    lines += ["", f"사전 정의한 두 지표의 동시 개선 조건: **{'충족' if success else '미충족'}**. "
              "거절로 인한 시간 감소를 성공 근거로 삼지 않습니다.", "",
              "## 도구 오적용·fallback 점검", "",
              f"주 평가의 도구 오답/오적용 {metadata['main_tool_misapplications']}건, 도구→모델 라우터 fallback "
              f"{metadata['main_tool_fallbacks']}건. 주 평가가 모두 지원 산술식이므로 별도 점검을 함께 실행했습니다.", "",
              "| 점검 사례 | 도구 허용 예상 | 실제 답변 출처 | fallback 원인 | 실제 모델 호출 | 오적용 |",
              "|---|---|---|---|---:|---|"]
    for r in probes:
        lines.append(f"| {r['id']} | {r['expected_tool']} | {r['source']} | {r['fallback_reason']} | {len(r['attempts'])} | {r['tool_misapplication']} |")
    lines += ["", "점검 사례는 주 평가의 정확도·지연·호출 합계에 넣지 않았습니다. "
              "구문/안전 제한으로 도구가 거부한 경우에는 기존 라우터를 실제 호출했습니다. "
              "자연어·미지원 수식을 원래 라우터도 처리하지 못하면 모델 호출 없이 거절합니다. "
              "선행 0이 있는 식은 Python 정수 구문으로 허용하지 않아 원래 모델 경로를 점검하는 사례로 사용했습니다.", "",
              "## 무결성·비용·한계", "",
              f"원본 소스·데이터 해시 보존: {metadata['preserved_files_unchanged']}. "
              f"독립 추출기 불일치: {metadata['extraction_audit_mismatches']}건. "
              f"전체 루프 경과 시간: {metadata['evaluation_wall_seconds']:.6f}초(진행 출력·채점 포함, 워밍업·저장 제외).", "",
              "문제 명세와 프로필·소스 해시를 실행 전 고정했고, 외부 AST 채점기로 기준 정답을 검사했습니다. "
              "계산 도구·런타임에는 기대 정답을 전달하지 않았습니다. 실제 모델 응답과 직접 도구 결과를 출처별로 기록했습니다.", "",
              "금전 비용·전력·CPU·메모리·FLOPs는 이번 실험에서 미측정입니다. 호출 수 감소를 금전 절감률로 바꾸지 않았습니다. "
              "Ollama 서버는 계속 실행 중이므로 모델 호출이 없어도 상주 메모리나 유휴 비용이 사라졌다는 뜻은 아닙니다.", "",
              "이 결과는 지원하는 산술식 영역에 한정됩니다. 소수·나눗셈·거듭제곱·단위·일반 문장·일반 AI 작업의 성능은 "
              "개선됐다고 주장하지 않습니다. 100문제 세트도 이전에 평가한 데이터의 재실행이며 완전히 새로운 홀드아웃은 아닙니다. "
              "단일 실행의 짧은 도구 지연은 반복 실험·부하 검증이 필요하고 모델과의 속도 차이를 모든 작업에 일반화할 수 없습니다.", "",
              "## 파일", "",
              *[f"- [{suffix}]({metadata['prefix_name']}_{suffix})" for suffix in
                ("manifest.json", "evaluation.csv", "summary.csv", "paired.csv", "tool_events.csv", "tool_misapplications.csv", "fallbacks.csv", "probes.csv", "metadata.json")], ""]
    return "\n".join(lines)


def run(data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", data_dir) for p in folder.iterdir() if p.is_file()}
    datasets = load_datasets(SOURCE_MANIFEST)
    profile = json.loads(PROFILE.read_text())
    policy = CoveragePolicy(profile)
    check_ollama([adaptive.SMALL, adaptive.LARGE])
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = json.load(response)
    run_id = uuid4().hex
    prefix = data_dir / f"tool_routing_{run_id}"
    manifest = {"version": VERSION, "datasets": datasets, "safety_cases": safety_cases(),
                "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
                "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                "preserved_sha256": {str(p): digest for p, digest in preserved.items()},
                "success_rule": "accuracy AND answer_rate strictly improve vs every baseline on each dataset"}
    manifest_path = Path(f"{prefix}_manifest.json")
    write_json(manifest_path, manifest)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for model in (adaptive.SMALL, adaptive.LARGE):
        for prompt in ("legacy", adaptive.PROMPT_VERSION):
            adaptive.generate(model, "1 + 1", prompt)
    rows = []
    start = time.perf_counter()
    for dataset, problems in datasets.items():
        for index, problem in enumerate(problems):
            offset = index % len(STRATEGIES)
            for strategy in STRATEGIES[offset:] + STRATEGIES[:offset]:
                row = execute_strategy(strategy, problem["question"], profile, policy)
                validation = validate_answer(row["answer"], problem["expected"])
                texts = [a.get("answer", "") for a in row["attempts"]] + [row["answer"]]
                audit = sum(audit_extract(text) != validate_answer(text, problem["expected"])["predicted"] for text in texts)
                row.update(dataset=dataset, **problem, predicted=validation["predicted"],
                           correct=row["status"] == "answered" and validation["correct"], audit_mismatches=audit,
                           tool_misapplication=row["source"] == "tool" and not validation["correct"])
                rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"Actual {dataset}: {index+1}/{len(problems)} (four strategies)", flush=True)
    wall = time.perf_counter() - start
    probes = []
    for case in safety_cases():
        row = execute_tool(case["question"], profile)
        actual_tool = row["source"] == "tool"
        predicted = audit_extract(row["answer"])
        row.update(**case, tool_misapplication=actual_tool != case["expected_tool"] or (
            actual_tool and predicted != case["value"]))
        probes.append(row)
    summaries = summarize(rows)
    pairs = comparisons(rows, summaries)
    for path, digest in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing file changed: {path}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("Manifest changed")
    metadata = {"run_id": run_id, "prefix_name": prefix.name, "evaluation_wall_seconds": wall,
                "ollama_tags": tags, "manifest_sha256": manifest_hash, "preserved_files_unchanged": True,
                "real_inference": True, "cost_status": "unmeasured",
                "main_tool_misapplications": sum(r["tool_misapplication"] for r in rows),
                "main_tool_fallbacks": sum(r["model_fallback"] for r in rows),
                "probe_misapplications": sum(r["tool_misapplication"] for r in probes),
                "extraction_audit_mismatches": sum(r["audit_mismatches"] for r in rows),
                "success_both_metrics_all_baselines": all(p["success_both_improved"] for p in pairs)}
    write_records(Path(f"{prefix}_evaluation.csv"), rows)
    write_records(Path(f"{prefix}_summary.csv"), summaries)
    write_records(Path(f"{prefix}_paired.csv"), pairs)
    write_records(Path(f"{prefix}_probes.csv"), probes)
    events = [{"dataset": r.get("dataset", "safety_probe"), "question": r["question"],
               "source": r["source"], "tool_calls": r["tool_calls"], "events": r["tool_events"],
               "model_fallback": r["model_fallback"], "fallback_reason": r["fallback_reason"],
               "model_calls": len(r["attempts"]), "status": r["status"], "misapplication": r["tool_misapplication"]}
              for r in rows + probes if r["strategy"] == "tool"]
    fields = list(events[0])
    write_records(Path(f"{prefix}_tool_events.csv"), events)
    write_records(Path(f"{prefix}_tool_misapplications.csv"), [r for r in events if r["misapplication"]], fields)
    write_records(Path(f"{prefix}_fallbacks.csv"), [r for r in events if r["model_fallback"]], fields)
    write_json(Path(f"{prefix}_metadata.json"), metadata)
    with Path(f"{prefix}_report.md").open("x", encoding="utf-8") as file:
        file.write(make_report(summaries, pairs, probes, metadata))
    for summary in summaries:
        if summary["difficulty"] == "all":
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"REPORT: {prefix}_report.md", flush=True)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Tool-Assisted Routing 실제 비교")
    parser.parse_args()
    try:
        run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"실험 중단: {error}\n")

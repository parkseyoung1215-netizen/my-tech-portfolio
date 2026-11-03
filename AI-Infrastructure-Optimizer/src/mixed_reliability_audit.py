"""Offline audit of frozen mixed-task results; never calls inference or changes routing."""

import ast
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import re
from uuid import uuid4

if __package__:
    from . import mixed_runtime
    from .mixed_experiment import audit_extract, CATEGORIES, STRATEGIES
    from .verify_routing200 import ROOT, problem_key, reference_answer
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import mixed_runtime
    from mixed_experiment import audit_extract, CATEGORIES, STRATEGIES
    from verify_routing200 import ROOT, problem_key, reference_answer
    from tool_experiment import write_records
    from adaptive_experiment import write_json

SOURCE = ROOT / "data/mixed_5f6833b806564466821eb9f81873848f"
VERSION = "posthoc-content-exact-v1"


def content_extract(response, task_format):
    """Gold-independent: original final line OR entire bare answer, nothing fuzzy."""
    strict = audit_extract(response)
    if strict is None:
        token = response.strip()
        pattern = r"[A-D]" if task_format == "choice" else r"[+-]?[0-9]+"
        if re.fullmatch(pattern, token, re.I) is None:
            return None, "unresolved"
        strict, mode = token.upper(), "bare_answer"
    else:
        mode = "final_line"
    if task_format == "choice":
        return (strict, mode) if strict in "ABCD" and len(strict) == 1 else (None, "unresolved")
    if task_format != "integer" or re.fullmatch(r"[+-]?[0-9]+", strict) is None:
        return None, "unresolved"
    try:
        return str(int(strict)), mode
    except ValueError:
        return None, "unresolved"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for r in rows:
        r["attempts"] = json.loads(r["attempts"])
        r["runtime_details"] = json.loads(r["runtime_details"])
        for k in ("correct", "general_adapter"):
            r[k] = r[k] == "True"
        for k in ("elapsed_seconds", "cpu_seconds", "cpu_percent", "boundary_max_rss_mib"):
            r[k] = float(r[k]) if r[k] else None
        r["tool_calls"] = int(r["tool_calls"])
    return rows


def score_row(row):
    strict = mixed_runtime.extract_final(row["answer"])
    if strict != audit_extract(row["answer"]):
        raise ValueError("Strict parsers disagree")
    if strict != (row["predicted"] or None):
        raise ValueError("Stored strict prediction mismatch")
    original_correct = row["status"] == "answered" and strict == row["expected"]
    if original_correct != row["correct"]:
        raise ValueError("Stored original score mismatch")
    # The parser receives response and answer type, never the expected answer.
    content, mode = content_extract(row["answer"], row["task_format"])
    content_correct = content is not None and content == row["expected"]
    if row["status"] == "answered":
        failure = "format_valid_content_correct" if content_correct else "format_valid_content_wrong"
    elif content is not None:
        failure = "format_only_error" if content_correct else "format_and_content_error"
    elif not row["attempts"]:
        failure = "policy_rejection"
    elif row["attempts"][-1]["status"] == "error":
        failure = "api_error"
    elif audit_extract(row["attempts"][-1].get("answer", "")) is not None:
        failure = "withheld_validation_failure"
    else:
        failure = "unresolved_format_error"
    models = [a["model"] for a in row["attempts"]]
    return {
        "strategy": row["strategy"], "category": row["category"], "question": row["question"],
        "expected": row["expected"], "raw_response": row["answer"], "original_status": row["status"],
        "original_prediction": strict, "original_correct": original_correct,
        "content_prediction": content, "content_parse_mode": mode, "content_correct": content_correct,
        "content_resolvable": content is not None, "failure_kind": failure,
        "primary_selected_model": models[0] if models else None,
        "model_sequence": models, "model_calls": len(models), "final_recorded_model": row["model"] or None,
        "answer_source": row["source"], "tool_calls": row["tool_calls"],
        "last_attempt_response": row["attempts"][-1].get("answer", "") if models else "",
        "selection_reason": row["runtime_details"].get("selection_reason", row["reason"]),
        "general_adapter": row["general_adapter"],
        "elapsed_seconds": row["elapsed_seconds"],
    }


def summary(rows, scored):
    result = []
    for strategy in STRATEGIES:
        for category in ("all",) + CATEGORIES:
            pairs = [(r, s) for r, s in zip(rows, scored) if r["strategy"] == strategy
                     and (category == "all" or r["category"] == category)]
            if not pairs:
                continue
            raw = [r for r, s in pairs]
            scores = [s for r, s in pairs]
            attempts = [a for r in raw for a in r["attempts"]]
            output = [a.get("output_tokens") for a in attempts]
            inputs = [a.get("prompt_tokens") for a in attempts]
            evaluations = [a.get("eval_duration") for a in attempts]
            cpu = [r["cpu_seconds"] for r in raw]
            rss = [r["boundary_max_rss_mib"] for r in raw if r["boundary_max_rss_mib"] is not None]
            original_answered = sum(r["status"] == "answered" for r in raw)
            resolved = sum(s["content_resolvable"] for s in scores)
            original_correct = sum(s["original_correct"] for s in scores)
            content_correct = sum(s["content_correct"] for s in scores)
            result.append({
                "strategy": strategy, "category": category, "requests": len(pairs),
                "original_correct": original_correct, "original_accuracy": original_correct / len(pairs),
                "original_answered": original_answered, "original_answer_rate": original_answered / len(pairs),
                "original_answered_accuracy": original_correct / original_answered if original_answered else None,
                "content_correct": content_correct, "content_accuracy": content_correct / len(pairs),
                "content_resolvable": resolved, "content_resolvable_rate": resolved / len(pairs),
                "content_resolved_accuracy": content_correct / resolved if resolved else None,
                "format_only_errors": sum(s["failure_kind"] == "format_only_error" for s in scores),
                "format_and_content_errors": sum(s["failure_kind"] == "format_and_content_error" for s in scores),
                "failure_counts": dict(Counter(s["failure_kind"] for s in scores)),
                "mean_latency_seconds": sum(r["elapsed_seconds"] for r in raw) / len(pairs),
                "total_seconds": sum(r["elapsed_seconds"] for r in raw),
                "small_calls": sum(a["model"] == "qwen2.5:0.5b" for a in attempts),
                "large_calls": sum(a["model"] == "qwen2.5:1.5b" for a in attempts),
                "model_calls": len(attempts), "direct_tool_calls": sum(r["tool_calls"] for r in raw),
                "output_tokens_total": sum(output) if all(v is not None for v in output) else None,
                "output_token_recorded_calls": sum(v is not None for v in output),
                "input_tokens_total": sum(inputs) if all(v is not None for v in inputs) else None,
                "input_token_recorded_calls": sum(v is not None for v in inputs),
                "input_tokens_recorded_subtotal": sum(v for v in inputs if v is not None),
                "total_tokens": sum(output) + sum(inputs) if all(v is not None for v in output+inputs) else None,
                "ollama_eval_seconds": sum(evaluations) if all(v is not None for v in evaluations) else None,
                "cpu_seconds_total": sum(cpu) if all(v is not None for v in cpu) else None,
                "cpu_samples": sum(v is not None for v in cpu),
                "rss_boundary_max_mib": max(rss) if rss else None, "rss_samples": len(rss),
                "cost": None, "cost_status": "unmeasured",
            })
    return result


def identity_audit(rows):
    result = []
    for question in dict.fromkeys(r["question"] for r in rows):
        group = [r for r in rows if r["question"] == question]
        if len(group) != 4 or {r["strategy"] for r in group} != set(STRATEGIES):
            raise ValueError("Missing strategy for a question")
        non_arithmetic = group[0]["category"] != "arithmetic"
        # Reconstructed from frozen source, not a captured HTTP wire log.
        body = {"model": "qwen2.5:1.5b", "prompt": question +
                "\nChoose the best option. Return exactly one line: FINAL_ANSWER: followed by "
                "the option letter A, B, C or D. Do not include explanations.",
                "stream": False, "options": {"temperature": 0, "num_predict": 64}}
        result.append({
            "question": question, "category": group[0]["category"],
            "identical_final_responses": len({r["answer"] for r in group}) == 1,
            "common_general_adapter": all(r["general_adapter"] for r in group),
            "actual_models": {r["strategy"]: [a["model"] for a in r["attempts"]] for r in group},
            "observed_model_calls": sum(len(r["attempts"]) for r in group),
            "reconstructed_common_payload_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() if non_arithmetic else None,
            "responses": {r["strategy"]: r["answer"] for r in group},
        })
    return result


FACT_REVIEW = {
    "What is the chemical formula of water?": "H2O", "Which symbol represents the element oxygen?": "O",
    "What kind of astronomical object is the Sun?": "A star", "Which planet is the largest in the Solar System?": "Jupiter",
    "Which process allows green plants to use light to make sugars?": "Photosynthesis",
    "Which gas do plants take in during photosynthesis?": "Carbon dioxide", "How many sides does a hexagon have?": "Six",
    "Which instrument measures temperature?": "Thermometer", "Which instrument indicates magnetic north?": "Compass",
    "Which state of matter has a fixed volume and a fixed shape?": "Solid", "What is the SI unit of electric current?": "Ampere",
    "Which material is generally a good electrical conductor?": "Copper", "What does a CPU primarily do in a computer?": "Execute instructions",
    "Which data structure follows first-in, first-out order?": "Queue", "Which protocol is used to transfer web pages?": "HTTP",
    "Which number system uses only the digits zero and one?": "Binary", "Which ocean is the largest by area?": "Pacific Ocean",
    "What causes day and night on Earth?": "Earth rotating on its axis", "Which change turns a liquid into a gas?": "Evaporation",
    "Which force attracts objects with mass toward one another?": "Gravity",
}


def review_gold(tasks, coding_reference):
    reviewed = []
    for t in tasks:
        q, category = t["question"], t["category"]
        options = dict(re.findall(r"(?m)^([A-D])\) (.+)$", q))
        if category != "arithmetic" and set(options) != set("ABCD"):
            raise ValueError("Invalid choices")
        intended = options.get(t["expected"], t["expected"])
        reason, reference = "", None
        if category == "arithmetic":
            reference, reason = str(reference_answer(q)), "independent_AST_arithmetic"
        elif category == "knowledge":
            reference, reason = FACT_REVIEW[q.splitlines()[0]], "manual_elementary_fact_review_no_external_citation"
        elif category == "coding":
            reference, reason = coding_reference[q], "previously_recorded_isolated_Python_reference_output"
        elif category == "logic":
            reason = "independent_template_logic_review"
            if "Who finishes last?" in q:
                reference = re.search(r"; [A-Za-z]+ finishes before ([A-Za-z]+)\.", q).group(1)
            elif "Who arrives first?" in q:
                reference = re.search(r"\. ([A-Za-z]+) arrives before", q).group(1)
            elif "Must it be a zork?" in q:
                reference = "No; being blue does not establish it is a zork"
            elif "The lamp is off." in q:
                reference = "The sensor is not active"
            elif "Can a coin be in that box?" in q:
                reference = "No"
        elif category == "comprehension":
            reason = "passage_entailment_review"
            if "originally owned" in q:
                reason = "ambiguous_ownership_lending_does_not_prove_ownership"
            elif "When is the event?" in q:
                reference = "Wednesday"
            elif "What did" in q and "order?" in q:
                reference = "Nothing"
            elif "What text is on the sign?" in q:
                reference = "2 + 3"
            elif "Why was the umbrella packed?" in q:
                reference = "Rain was forecast"
        ambiguous = reason.startswith("ambiguous")
        reviewed.append({"question": q, "category": category, "stored_label": t["expected"],
                         "stored_answer_text": intended, "review_reference": reference,
                         "review_status": "ambiguous_label" if ambiguous else "agrees" if reference == intended else "mismatch_or_unreviewed",
                         "review_reason": reason})
    return reviewed


def verify_integrity(prefix, rows):
    prefix = Path(prefix)
    manifest = json.loads(Path(f"{prefix}_manifest.json").read_text())
    metadata = json.loads(Path(f"{prefix}_metadata.json").read_text())
    if hashlib.sha256(Path(f"{prefix}_manifest.json").read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    keys = {problem_key(t["question"]) for t in manifest["tasks"]}
    if len(keys) != 100 or len(rows) != 400:
        raise ValueError("Expected 100 tasks / 400 rows")
    for name, digest in manifest["preserved_sha256"].items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source/data changed: {name}")
        prior = set()
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as file:
                prior = {problem_key(r["question"]) for r in csv.DictReader(file) if r.get("question")}
        elif path.suffix == ".json":
            def walk(obj):
                if isinstance(obj, dict):
                    if isinstance(obj.get("question"), str):
                        yield problem_key(obj["question"])
                    for value in obj.values():
                        yield from walk(value)
                elif isinstance(obj, list):
                    for value in obj:
                        yield from walk(value)
            prior = set(walk(json.loads(path.read_text())))
        if keys & prior:
            raise ValueError(f"Evaluation overlaps prior dataset: {name}")
    tasks = {t["question"]: t for t in manifest["tasks"]}
    for strategy in STRATEGIES:
        group = [r for r in rows if r["strategy"] == strategy]
        if len(group) != 100 or {r["question"] for r in group} != set(tasks):
            raise ValueError("Strategies used different tasks")
        for r in group:
            t = tasks[r["question"]]
            if any(r[k] != t[k] for k in ("expected", "category", "task_format")):
                raise ValueError("Evaluation row does not match frozen task")
    if "expected" in inspect.signature(mixed_runtime.execute).parameters:
        raise ValueError("Runtime unexpectedly accepts labels")
    # Inspect the exact call site instead of inferring runtime inputs from score rows.
    source = ROOT / "src/mixed_experiment.py"
    tree = ast.parse(source.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "execute"]
    if len(calls) != 1 or ast.unparse(calls[0]) != "execute(task['question'], strategy, profile, policy)":
        raise ValueError("Runtime call site changed or includes extra evaluation data")
    return {"manifest_hash_verified": True, "frozen_source_data_verified": True,
            "prior_project_overlap": 0, "task_rows_verified": 400,
            "runtime_call_verified": "question, strategy, frozen profile, coverage policy only",
            "pretraining_overlap": "not_measured", "raw_HTTP_payload_capture": "not_recorded"}, manifest


def make_report(summaries, identities, gold, metadata):
    overall = {s["strategy"]: s for s in summaries if s["category"] == "all"}
    def pct(v):
        return f"{v:.2%}" if v is not None else "해당 없음"
    def num(v, digits=3):
        return f"{v:.{digits}f}" if v is not None else "미측정"
    lines = ["# 혼합 평가 신뢰성 감사", "", f"감사 버전 `{VERSION}`, 실행 ID `{metadata['run_id']}`.", "",
             "모델 재실행·라우팅 변경 없이 기존400행의 응답과 실제 계측 로그만 재분석했습니다. "
             "원래 점수·채점기·원본 CSV/보고서는 보존했습니다. 내용 지표는 **사후 보조 지표**이며 새 배포 버전의 실측 결과가 아닙니다.", "",
             "## 평가 전후 — 동일한 응답, 다른 지표", "",
             "내용 파서는 기존 FINAL_ANSWER 줄 또는 응답 전체가 단일 A~D/정수인 경우만 인정합니다. "
             "문장 속 숫자·정답 후보를 찾거나 여러 답 중 정답을 선택하지 않습니다. 정답 라벨은 파서 인수가 아닙니다. "
             "Coverage가 일부러 반환하지 않은 실패 후보를 답변으로 되살리지 않습니다.", "",
             "| 방식 | 기존 정확도 | 내용 정확도 | 기존 답변율 | 내용 해석 가능률 | 기존 답변 정확도 | 해석된 내용 정확도 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in overall.values():
        lines.append(f"| {s['strategy']} | {pct(s['original_accuracy'])} | {pct(s['content_accuracy'])} | "
                     f"{pct(s['original_answer_rate'])} | {pct(s['content_resolvable_rate'])} | "
                     f"{pct(s['original_answered_accuracy'])} | {pct(s['content_resolved_accuracy'])} |")
    lines += ["", "내용 해석 가능률은 원래 서비스가 승인·반환한 답변율이 아닙니다. "
              "저장된 응답을 형식에 덜 의존해 읽을 수 있었는지를 뜻합니다. 실행 시간과 비용을 새 채점기로 재측정하지 않았습니다.", "",
              "## 유형별 정확도: 기존 → 내용", "",
              "| 유형 | fixed | adaptive | coverage | tool |", "|---|---|---|---|---|"]
    ix = {(s["strategy"], s["category"]): s for s in summaries}
    for category in CATEGORIES:
        cells = [f"{pct(ix[(st,category)]['original_accuracy'])} → {pct(ix[(st,category)]['content_accuracy'])}" for st in STRATEGIES]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "## 형식과 내용 오류", "",
              "각 방식의 비산술80문제에는 접두어 없는 단일 문자45건이 있습니다. 그중38건은 정답,7건은 오답입니다. "
              "형식이 맞았던35건 중에도10건은 오답입니다. 따라서 형식만 개선해도 모든 내용 오류가 해결되지는 않습니다. "
              "API 장애나 출력 토큰 제한으로 인한 절단과도 구분해야 합니다.", "",
              "원래 요약의 format_failures 필드는 모델 호출 후 미답변을 합산해 Coverage의 산술 검증 실패7건도 포함했습니다. "
              "새 문항별 감사 CSV는 정책 거절·형식만 오류·형식+내용 오류·내용 오류·검증 실패를 분리합니다.", "",
              "## 실제 측정 효율", "",
              "| 방식 | 평균 지연(ms) | 처리 시간(초) | 0.5B/1.5B 호출 | 직접 도구 | 출력 토큰 | 입력 토큰 전체 | CPU(초) | RSS 경계최대(MiB) |",
              "|---|---:|---:|---|---:|---:|---|---:|---:|"]
    for s in overall.values():
        lines.append(f"| {s['strategy']} | {s['mean_latency_seconds']*1000:.3f} | {s['total_seconds']:.3f} | "
                     f"{s['small_calls']}/{s['large_calls']} | {s['direct_tool_calls']} | {s['output_tokens_total']} | "
                     f"{num(s['input_tokens_total'],0)} | {num(s['cpu_seconds_total'])} | {num(s['rss_boundary_max_mib'],1)} |")
    lines += ["", "출력 토큰은 모든 실제 모델 시도(실패·재시도 포함)의 기록을 합산했습니다. "
              "입력은 fixed의 산술20회에만1359토큰이 기록됐고 나머지는 미기록입니다. 전체 입력·총 토큰을 추정하지 않았습니다. "
              "산술 도구 경로의 모델 토큰0은 모델 호출이 없다는 뜻이며 계산 자체가 무비용이라는 뜻은 아닙니다.", "",
              "요청 시간에는 라우팅·거절·검증·fallback이 포함되고 ps 수집·채점·워밍업은 제외됩니다. "
              "CPU는 Ollama 프로세스군의 경계 누적 차이며 도구/어댑터 Python CPU와 GPU는 제외됩니다. "
              "RSS는 공유·상주 모델을 포함할 수 있는 경계 표본으로 전략별 전용/피크 메모리가 아닙니다. "
              "금전 비용·전력·FLOPs·GPU 자원은 미측정입니다. 토큰·호출 감소를 실제 금전 절감률로 바꾸지 않았습니다.", "",
              "## 비산술 응답이 같은 이유", "",
              f"비산술80문제 중 네 응답이 모두 같은 문항은 {sum(r['identical_final_responses'] for r in identities if r['category']!='arithmetic')}개입니다. "
              "원래 산술 라우터는 일반 작업을 거절하므로 mixed_runtime의 공통 어댑터가 모든 비산술 요청을 동일한 "
              "qwen2.5:1.5b·프롬프트·온도0·출력64토큰 상한으로 보냈습니다. 각 방식에80회, 총320회 모델 시도가 기록됐습니다. "
              "현재 코드는 비산술 모델을 차별적으로 선택하지 않으며 별도 결과 캐시나 앞 방식의 답을 복사하는 경로도 없습니다. "
              "같은 입력/설정이 동일 응답의 직접적인 구조적 이유이며 온도0도 일관성에 기여할 수 있습니다.", "",
              "실제 HTTP 요청 본문·서버 요청ID·seed·캐시 hit 로그는 저장하지 않았습니다. "
              "identity.csv의 요청 해시는 고정된 소스에서 재구성한 값이며 네트워크 캡처를 뜻하지 않습니다. "
              "서버 내부 캐시나 비결정성을 완전히 배제했다고 주장하지 않습니다.", "",
              "## 개선의 범위와 최종 판단", "",
              "내용 기준 Tool은 fixed보다 정확도72%→83%, 호출100→80, 출력486→300토큰, "
              "처리 시간9.153→5.902초로 개선됐습니다. 내용 해석 가능률은 둘 다100%이므로 답변 범위가 늘어난 것은 아닙니다. "
              "정확도와 답변 비율이 모두 엄격하게 증가했다는 성공 선언은 하지 않습니다.", "",
              "비산술 정확도·모델 호출·출력 토큰은 네 방식 모두 같습니다. fixed와 Tool의 정확도 차이11개, "
              "모델20회·출력186토큰 차이는 모두 산술에서 나옵니다. 다양한 작업의 모델 선택이 개선됐다는 증거는 없습니다. "
              "평가된 산술 요청에서 품질을 유지/개선하며 모델 추론 작업을 줄였다는 증거는 있지만, "
              "유료 청구나 전체 하드웨어 비용 절감·일반 서비스 성능을 입증한 것은 아닙니다.", "",
              "## 정답·누출·무결성", "",
              f"정답 검토 결과: {dict(Counter(g['review_status'] for g in gold))}.", "",
              "산술은 별도 AST 계산, 코딩은 기존에 기록한 격리 Python 기준 출력20건, 논리는 규칙 추론, "
              "지식은 기초 사실 수작업 검토, 지문은 명시적 내용으로 점검했습니다. 지식 검토는 외부 출처의 독립 사실 검증을 뜻하지 않습니다. "
              "소유권 지문4개는 '빌려주었다'만으로 소유자를 단정할 수 없어 모호한 라벨로 표시했습니다. "
              "원래 라벨과 주 점수를 바꾸지 않았고 gold_review.csv에 검토 근거를 남겼습니다. "
              "따라서 정답 판정 문제를 전부 해소했다고 주장할 수 없습니다.", "",
              "원본400행의 저장된 예측/점수와 독립 추출기를 대조했고, 동일 문제 집합·과거 프로젝트 평가 중복0·소스/프로필 해시를 검사했습니다. "
              "런타임 호출부는 질문·전략·기존 프로필·정책만 전달하고 정답/유형은 이후 채점에만 사용합니다. "
              "감사 범위 안에서 라벨 누출 정황은 없었지만 사전학습 중복이나 저장하지 않은 서버 상태까지 증명한 것은 아닙니다.", "",
              "수작업 객관식100개, 유형별20개, 반복 템플릿과 쉬운 분류 단서라는 한계가 있습니다. "
              "다음 신뢰성 검증은 명시적 답변 계약·모호한 라벨 정리·독립 새 문제·반복 측정·입출력 토큰/요청ID 기록이 필요합니다. "
              "이번에는 채점기·라우터·모델을 바꾸거나 추가 추론을 실행하지 않았습니다.", "",
              "## 파일", "",
              *[f"- [{s}]({metadata['prefix_name']}_{s})" for s in
                ("per_question.csv", "summary.csv", "identity.csv", "gold_review.csv", "integrity.json", "metadata.json")], ""]
    return "\n".join(lines)


def run(source=SOURCE, output_dir=ROOT / "data"):
    source, output_dir = Path(source), Path(output_dir)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", ROOT / "data") for p in folder.iterdir() if p.is_file()}
    rows = read_rows(Path(f"{source}_evaluation.csv"))
    integrity, manifest = verify_integrity(source, rows)
    scored = [score_row(r) for r in rows]
    summaries = summary(rows, scored)
    # Confirm that the 'before' table exactly reproduces the saved original summary.
    with Path(f"{source}_summary.csv").open() as file:
        original = {(r["strategy"], r["category"]): r for r in csv.DictReader(file)}
    for s in summaries:
        old = original[(s["strategy"], s["category"])]
        if s["original_correct"] != int(old["correct"]) or s["original_answered"] != int(old["answered"]):
            raise ValueError("Before metrics do not match original results")
    identities = identity_audit(rows)
    with Path(f"{source}_coding_reference_audit.csv").open() as file:
        code_review = list(csv.DictReader(file))
    if len(code_review) != 20 or any(r["matches"] != "True" for r in code_review):
        raise ValueError("Missing/mismatched coding reference execution")
    gold = review_gold(manifest["tasks"], {r["question"]: r["reference_output"] for r in code_review})
    run_id = uuid4().hex
    prefix = output_dir / f"mixed_reliability_{run_id}"
    metadata = {"run_id": run_id, "prefix_name": prefix.name, "audit_version": VERSION,
                "source": str(source), "inference_rerun": False, "posthoc_metric": True,
                "scoring_rule": "unique final-line answer or whole-response bare answer; no label-dependent extraction",
                "preserved_sha256": {str(p): d for p, d in preserved.items()}}
    write_records(Path(f"{prefix}_per_question.csv"), scored)
    write_records(Path(f"{prefix}_summary.csv"), summaries)
    write_records(Path(f"{prefix}_identity.csv"), identities)
    write_records(Path(f"{prefix}_gold_review.csv"), gold)
    write_json(Path(f"{prefix}_integrity.json"), integrity)
    write_json(Path(f"{prefix}_metadata.json"), metadata)
    with Path(f"{prefix}_report.md").open("x", encoding="utf-8") as file:
        file.write(make_report(summaries, identities, gold, metadata))
    for p, digest in preserved.items():
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing file changed during audit: {p}")
    for s in summaries:
        if s["category"] == "all":
            print(json.dumps(s, ensure_ascii=False))
    print(f"REPORT: {prefix}_report.md")
    return prefix


if __name__ == "__main__":
    run()

"""Replay and unseen evaluation of verifier-gated large-model fallback.

Scoring labels stay in this experiment module, outside the runtime components.
"""

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import urllib.request
from uuid import uuid4

if __package__:
    from . import adaptive_router as old
    from .coverage_policy import CoveragePolicy
    from .coverage_runner import execute as execute_coverage, VERSION
    from .verify_routing200 import ROOT, PROFILE, prior_questions, problem_key, reference_answer, audit_extract
    from .adaptive_experiment import write_json, write_csv
    from .answer_validator import validate_answer
    from .routing_runner import check_ollama
else:
    import adaptive_router as old
    from coverage_policy import CoveragePolicy
    from coverage_runner import execute as execute_coverage, VERSION
    from verify_routing200 import ROOT, PROFILE, prior_questions, problem_key, reference_answer, audit_extract
    from adaptive_experiment import write_json, write_csv
    from answer_validator import validate_answer
    from routing_runner import check_ollama

HISTORICAL = ROOT / "data/verification200_faeda373c1e6443f9775a4adb75be465"
STRATEGIES = ("fixed", "adaptive_old", "coverage")
SEED = 2026110601


def unseen_problems(data_dir):
    seen = prior_questions(data_dir)
    rng = random.Random(SEED)
    problems = []
    for difficulty, count in (("easy", 20), ("medium", 40), ("hard", 40)):
        if difficulty == "easy":
            pool = [f"{a} + {b}" for a in range(1, 21) for b in range(a, 21)]
        elif difficulty == "medium":
            pool = [f"{a} * {b}" for a in range(20, 100) for b in range(10, 100)]
        else:
            pool = [f"({rng.randint(100,999)} * {rng.randint(10,99)}) - {rng.randint(10,99)}" for _ in range(10000)]
        rng.shuffle(pool)
        selected = 0
        for question in pool:
            key = problem_key(question)
            if key in seen:
                continue
            seen.add(key)
            problems.append({"question": question, "expected": reference_answer(question),
                             "difficulty": difficulty, "bucket": old.bucket_for(question)})
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(f"Insufficient unused {difficulty} problems; refusing reuse")
    rng.shuffle(problems)
    return problems


def historical_causes(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["strategy"] == "improved"]
    calls = [a for r in rows for a in json.loads(r["attempts"])]
    return {
        "requests": len(rows), "unanswered": sum(r["status"] != "answered" for r in rows),
        "small_confidence_and_large_policy_rejection": sum(not json.loads(r["attempts"]) for r in rows),
        "large_model_calls": sum(a["model"] == old.LARGE for a in calls),
        "fallback_requests": sum(len(json.loads(r["attempts"])) > 1 for r in rows),
        "api_errors": sum(a["status"] == "error" for a in calls),
        "extraction_failures": sum(a["status"] == "extraction_failed" for a in calls),
        "rejection_buckets": dict(Counter(r["bucket"] for r in rows if r["status"] != "answered")),
    }


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
                attempts = [a for r in group for a in r["attempts"]]
                selected = Counter(r["attempts"][0]["model"] if r["attempts"] else "none" for r in group)
                final = Counter(r["model"] or "none" for r in group)
                correct = sum(r["correct"] for r in group)
                summaries.append({
                    "dataset": dataset, "strategy": strategy, "difficulty": difficulty,
                    "requests": len(group), "answered": len(answered), "correct": correct,
                    "accuracy": correct / len(group),
                    "answered_accuracy": correct / len(answered) if answered else None,
                    "answer_rate": len(answered) / len(group),
                    "mean_latency": sum(r["elapsed_seconds"] for r in group) / len(group),
                    "total_seconds": sum(r["elapsed_seconds"] for r in group),
                    "primary_selection_ratios": {m: selected[m] / len(group) for m in (old.SMALL, old.LARGE, "none")},
                    "final_answer_ratios": {m: final[m] / len(group) for m in (old.SMALL, old.LARGE, "none")},
                    "model_calls": dict(Counter(a["model"] for a in attempts)),
                    "model_seconds": {m: sum(a["elapsed_seconds"] for a in attempts if a["model"] == m)
                                      for m in (old.SMALL, old.LARGE)},
                    "api_calls": len(attempts),
                    "fallback_requests": sum(len(r["attempts"]) > 1 for r in group),
                    "fallback_rate": sum(len(r["attempts"]) > 1 for r in group) / len(group),
                    "fallback_seconds": sum(r["fallback_seconds"] for r in group),
                    "validation_seconds": sum(r["validation_seconds"] for r in group),
                    "routing_seconds": (sum(r["routing_seconds"] for r in group)
                                        if all(r["routing_seconds"] is not None for r in group) else None),
                    "large_handoff_requests": sum(r["large_handoff"] for r in group),
                    "large_retry_requests": sum(r["large_retry"] for r in group),
                    "policy_refusals": sum(not r["attempts"] for r in group),
                    "unanswered_reasons": dict(Counter(r["reason"] for r in group if r["status"] != "answered")),
                    "attempt_failure_reasons": dict(Counter(a.get("reason", a["status"]) for a in attempts if a["status"] != "answered")),
                    "audit_mismatches": sum(r["audit_mismatches"] for r in group),
                    "target_accuracy_met": correct / len(group) >= 0.55,
                    "target_answer_rate_met": len(answered) / len(group) >= 0.95,
                    "cost": None, "cost_status": "unmeasured",
                })
    return summaries


def paired(rows):
    index = {(r["dataset"], r["strategy"], r["question"]): r for r in rows}
    pairs = []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        for strategy in ("fixed", "adaptive_old"):
            original = [r for r in rows if r["dataset"] == dataset and r["strategy"] == strategy]
            common = [(r, index[(dataset, "coverage", r["question"])]) for r in original
                      if r["status"] == "answered" and index[(dataset, "coverage", r["question"])]["status"] == "answered"]
            bt = sum(a["elapsed_seconds"] for a, b in common)
            nt = sum(b["elapsed_seconds"] for a, b in common)
            pairs.append({
                "dataset": dataset, "baseline": strategy, "common_answered": len(common),
                "baseline_correct": sum(a["correct"] for a, b in common),
                "coverage_correct": sum(b["correct"] for a, b in common),
                "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in common),
                "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in common),
                "lost_answers": sum(r["status"] == "answered" and index[(dataset, "coverage", r["question"])]["status"] != "answered" for r in original),
                "lost_correct_answers": sum(r["correct"] and index[(dataset, "coverage", r["question"])]["status"] != "answered" for r in original),
                "baseline_common_seconds": bt, "coverage_common_seconds": nt,
                "common_speed_ratio": bt / nt if nt else None,
            })
    return pairs


def make_report(summaries, pairs, metadata):
    labels = {"fixed": "고정 1.5B", "adaptive_old": "기존 Adaptive", "coverage": "개선 Adaptive"}
    def pct(v):
        return f"{v:.2%}" if v is not None else "해당 없음"
    lines = ["# 답변 범위 개선 실제 실험", "", f"실행 ID: `{metadata['run_id']}`.", "",
             "## 기존 200문제 미답변 원인", "", "```json", json.dumps(metadata["historical_causes"], ensure_ascii=False, indent=2), "```", "",
             "기존 Adaptive는 중간 50개·어려움 50개를 두 모델 모두 신뢰 기준 미달이라는 이유로 호출 전 거절했습니다. "
             "1.5B로 전달되지 않았으므로 fallback 실패나 추출 실패가 거절 원인은 아니었습니다.", "",
             "## 변경 범위와 누출 방지", "",
             "생성(coverage_runner), 검증(request_verifier), 선택(coverage_policy)을 분리했습니다. "
             "20표본·Wilson 하한 80%라는 작은 모델 기준을 유지하고, 보정 프로필은 한 번만 읽어 메모리에 고정했습니다.", "",
             "신뢰도가 낮은 요청은 1.5B를 검증 대상 후보로 호출합니다. 산술식에 한해 입력 식을 계산해 모델 출력과 비교하는 "
             "**결정적 산술 검증기**를 추가했습니다. 이 기능은 평가 정답을 읽지 않지만 실행 중 입력 식의 올바른 값을 계산합니다. "
             "따라서 순수 LLM 신뢰도 라우팅과 다른, 도구 검증을 결합한 시스템입니다. 계산값을 답변으로 대신 반환하거나 "
             "모델의 프롬프트에 정답·수정값을 제공하지 않습니다. 검증된 모델 응답만 반환합니다.", "",
             "큰 모델의 사전 신뢰도가 좋아졌다고 가정하지 않습니다. 오류·잘못된 정수·추출 실패 시 1.5B를 최대 한 번 재시도합니다. "
             "단순히 답변 비율을 높이기 위해 실패한 답변을 승인하지 않습니다. 일반 자연어 요청에는 이 검증을 적용할 수 없습니다.", "",
             "정답 라벨은 모델 선택·생성·실행 함수에 전달하지 않고 모든 모델 호출이 끝난 뒤 채점에만 사용했습니다. "
             "기존 200문제는 명시적인 재평가이며 독립 홀드아웃으로 주장하지 않습니다. 신규 100문제는 기존 CSV·JSON 및 재평가 세트와 "
             "교환법칙 중복까지 제외했습니다. 평가 결과에 따라 프롬프트·재시도 예산·기준을 변경하지 않았습니다."]
    for dataset in ("replay200", "unseen100"):
        lines += ["", f"## {dataset}", "",
                  "| 방식 | 정답/요청 | 전체 정확도 | 답변 정확도 | 답변 비율 | 평균 지연 | 처리 시간 합계 | 첫 선택 0.5B / 1.5B / 거절 | 재호출 fallback |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
        for s in summaries:
            if s["dataset"] != dataset or s["difficulty"] != "all":
                continue
            ratios = s["primary_selection_ratios"]
            lines.append(f"| {labels[s['strategy']]} | {s['correct']}/{s['requests']} | {pct(s['accuracy'])} | "
                         f"{pct(s['answered_accuracy'])} | {pct(s['answer_rate'])} | {s['mean_latency']:.4f}초 | {s['total_seconds']:.3f}초 | "
                         f"{pct(ratios[old.SMALL])} / {pct(ratios[old.LARGE])} / {pct(ratios['none'])} | {pct(s['fallback_rate'])} |")
        s = next(s for s in summaries if s["dataset"] == dataset and s["strategy"] == "coverage" and s["difficulty"] == "all")
        lines += ["", f"목표 판정: 전체 정확도 55% 이상 **{'달성' if s['target_accuracy_met'] else '미달'}**, "
                  f"답변 비율 95% 이상 **{'달성' if s['target_answer_rate_met'] else '미달'}**.", "",
                  f"큰 모델 실제 전달 {s['large_handoff_requests']}건, 큰 모델 재시도 {s['large_retry_requests']}건, "
                  f"fallback 시간 합계 {s['fallback_seconds']:.3f}초. 모델별 시도 시간: `{s['model_seconds']}`.", "",
                  f"미답변 사유: `{s['unanswered_reasons']}`."]
    lines += ["", "## 같은 문제에 답한 경우의 시간·품질", "",
              "| 세트 | 대상 | 공통 답변 | 공통 정답: 대상 / 개선 | 대상 시간 | 개선 시간 | 속도 비 | 잃은 답변 / 그중 정답 | 정답→오답 |",
              "|---|---|---:|---|---:|---:|---:|---|---:|"]
    for p in pairs:
        ratio = f"{p['common_speed_ratio']:.2f}" if p["common_speed_ratio"] else "해당 없음"
        lines.append(f"| {p['dataset']} | {labels[p['baseline']]} | {p['common_answered']} | "
                     f"{p['baseline_correct']} / {p['coverage_correct']} | {p['baseline_common_seconds']:.3f}초 | "
                     f"{p['coverage_common_seconds']:.3f}초 | {ratio} | {p['lost_answers']} / {p['lost_correct_answers']} | {p['correct_to_wrong']} |")
    lines += ["", "거절을 포함한 평균 감소를 속도 개선의 증거로 사용하지 않습니다. "
              "같은 질문의 모든 모델 호출·검증·재시도 시간이 공통 범위 비교에 포함됩니다. "
              "더 많은 요청에 답하는 시스템과 절반을 거절하는 시스템의 전체 처리량 차이는 별도로 해석해야 합니다.", "",
              "## 한계와 감사", "",
              f"신규 문제 분포: `{metadata['unseen_distribution']}`. 기존 200문제와 분포가 다르므로 두 세트의 전체 점수 차이를 "
              "과적합 효과만으로 해석할 수 없습니다. 난이도별 CSV도 함께 확인해야 합니다.", "",
              f"독립 정답 추출 파서 불일치 {sum(s['audit_mismatches'] for s in summaries if s['difficulty']=='all')}건. "
              "원문과 모델별 오류·검증 실패·시도 시간을 상세 CSV에 남겼습니다. "
              "형식 실패에 대한 기존 엄격한 추출 정책은 유지했습니다.", "",
              "큰 모델의 틀린 답을 검출할 수 있어도 올바른 모델 답변을 항상 얻을 수 있는 것은 아닙니다. "
              "제한된 재시도에도 틀린 계산이 반복되면 미답변으로 남습니다. 목표 미달을 신뢰 기준 하향으로 숨기지 않습니다.", "",
              f"워밍업 제외 전체 루프 경과 시간 {metadata['evaluation_wall_seconds']:.3f}초. "
              "요청별 시간은 라우팅·오류·거절·검증·fallback을 포함하고, 채점·CSV 저장은 제외합니다. "
              "반복·장시간 측정은 없으며 캐시·발열·백그라운드 부하를 완전히 통제하지 못했습니다.", "",
              "금전 비용·전력·FLOPs는 미측정이며 추정 비용 절감률을 만들지 않았습니다. "
              "상대 처리 시간은 이번 동시 비교 기준이며 과거 실행의 지연과 직접 비교하면 환경 차이가 섞일 수 있습니다.", "",
              "## 다음 단계 제안 — 승인 전 미실행", "",
              "1. 산술 도구의 계산 결과를 직접 반환하는 명시적 도구 실행 모드와, 검증된 모델 답변만 반환하는 현 모드를 구분해 비교합니다.",
              "2. 답변 범위를 유지하는 더 강한 모델 또는 도구 지원을 검토하고 별도 보정·홀드아웃으로 평가합니다.",
              "3. 같은 답변 범위에서 반복 지연 측정과 캐시·모델 로딩 정책을 검증합니다.", "",
              "추가 구현·모델 설치·추가 실험은 사용자 승인 후 진행합니다.", "",
              "## 결과 파일", "",
              *[f"- [{suffix}]({metadata['prefix_name']}_{suffix})" for suffix in
                ("manifest.json", "evaluation.csv", "summary.csv", "paired.csv", "attempts.csv", "metadata.json")], ""]
    return "\n".join(lines)


def run(data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", data_dir) for p in folder.iterdir() if p.is_file()}
    profile = json.loads(PROFILE.read_text())
    policy = CoveragePolicy(profile)
    replay = json.loads(Path(f"{HISTORICAL}_splits.json").read_text())["problems"]
    unseen = unseen_problems(data_dir)
    causes = historical_causes(Path(f"{HISTORICAL}_evaluation.csv"))
    check_ollama([old.SMALL, old.LARGE])
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = json.load(response)
    run_id = uuid4().hex
    prefix = data_dir / f"coverage_{run_id}"
    manifest = {"policy_version": VERSION, "seed": SEED,
                "calibration_profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                "small_threshold": 0.8, "max_small_calls": 1, "max_large_calls": 2,
                "runtime_verifier": "request-derived exact arithmetic; no label inputs",
                "datasets": {"replay200": replay, "unseen100": unseen},
                "preserved_sha256": {str(p): digest for p, digest in preserved.items()}}
    manifest_path = Path(f"{prefix}_manifest.json")
    write_json(manifest_path, manifest)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for model in (old.SMALL, old.LARGE):
        for prompt in ("legacy", old.PROMPT_VERSION):
            old.generate(model, "1 + 1", prompt)
    rows = []
    orders = list(itertools.permutations(STRATEGIES))
    start = time.perf_counter()
    for dataset, problems in (("replay200", replay), ("unseen100", unseen)):
        for index, problem in enumerate(problems):
            for strategy in orders[index % len(orders)]:
                if strategy == "coverage":
                    row = execute_coverage(problem["question"], policy)
                else:
                    row = old.execute(problem["question"], profile,
                                      "fixed" if strategy == "fixed" else "improved", 0.8)
                    row.update(strategy=strategy, routing_seconds=None, validation_seconds=0.0,
                               fallback_seconds=sum(a["elapsed_seconds"] for a in row["attempts"][1:]),
                               large_handoff=any(a["model"] == old.LARGE for a in row["attempts"]),
                               large_retry=sum(a["model"] == old.LARGE for a in row["attempts"]) > 1,
                               selection_reason="original_policy")
                    # Existing runtime did not separately time routing; keep that missing.
                validation = validate_answer(row["answer"], problem["expected"])
                mismatches = sum(audit_extract(a.get("answer", "")) != validate_answer(
                    a.get("answer", ""), problem["expected"])["predicted"] for a in row["attempts"])
                row.update(dataset=dataset, **problem, predicted=validation["predicted"],
                           correct=row["status"] == "answered" and validation["correct"], audit_mismatches=mismatches)
                rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"Actual {dataset}: {index+1}/{len(problems)}", flush=True)
    wall = time.perf_counter() - start
    # Missing old routing timing is not fabricated from total-minus-generation.
    summaries = summarize(rows)
    comparisons = paired(rows)
    for path, digest in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing file changed: {path}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("Manifest changed")
    metadata = {"run_id": run_id, "prefix_name": prefix.name, "evaluation_wall_seconds": wall,
                "unseen_distribution": dict(Counter(p["difficulty"] for p in unseen)),
                "historical_causes": causes, "ollama_tags": tags, "manifest_sha256": manifest_hash,
                "real_inference": True, "preserved_files_unchanged": True, "cost_status": "unmeasured"}
    write_csv(Path(f"{prefix}_evaluation.csv"), rows)
    write_csv(Path(f"{prefix}_summary.csv"), summaries)
    write_csv(Path(f"{prefix}_paired.csv"), comparisons)
    attempts = [{"dataset": r["dataset"], "strategy": r["strategy"], "question": r["question"],
                 "attempt_index": i, "model": a["model"], "status": a["status"],
                 "reason": a.get("reason", a["status"]), "elapsed_seconds": a["elapsed_seconds"],
                 "answer": a.get("answer", ""), "error": a.get("error", "")}
                for r in rows for i, a in enumerate(r["attempts"], 1)]
    write_csv(Path(f"{prefix}_attempts.csv"), attempts)
    write_json(Path(f"{prefix}_metadata.json"), metadata)
    with Path(f"{prefix}_report.md").open("x", encoding="utf-8") as file:
        file.write(make_report(summaries, comparisons, metadata))
    for s in summaries:
        if s["difficulty"] == "all":
            print(json.dumps(s, ensure_ascii=False), flush=True)
    print(f"REPORT: {prefix}_report.md", flush=True)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="거절 원인 분석 및 검증 기반 fallback 실측")
    parser.parse_args()
    try:
        run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"실험 중단: {error}\n")

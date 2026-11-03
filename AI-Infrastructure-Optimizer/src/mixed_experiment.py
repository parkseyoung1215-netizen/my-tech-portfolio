"""Live mixed-task benchmark with preserved arithmetic policies and common general adapter."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request
from uuid import uuid4

if __package__:
    from .mixed_dataset import build_dataset, SEED
    from .mixed_runtime import execute, extract_final, generate_general
    from . import adaptive_router as old
    from .coverage_policy import CoveragePolicy
    from .coverage_runner import execute as legacy_coverage
    from .tool_router import execute as legacy_tool
    from .verify_routing200 import ROOT, PROFILE, process_snapshot, resource_delta, problem_key
    from .routing_runner import check_ollama
    from .adaptive_experiment import write_json
    from .tool_experiment import write_records
else:
    from mixed_dataset import build_dataset, SEED
    from mixed_runtime import execute, extract_final, generate_general
    import adaptive_router as old
    from coverage_policy import CoveragePolicy
    from coverage_runner import execute as legacy_coverage
    from tool_router import execute as legacy_tool
    from verify_routing200 import ROOT, PROFILE, process_snapshot, resource_delta, problem_key
    from routing_runner import check_ollama
    from adaptive_experiment import write_json
    from tool_experiment import write_records

STRATEGIES = ("fixed", "adaptive", "coverage", "tool")
CATEGORIES = ("arithmetic", "knowledge", "logic", "coding", "comprehension")


def audit_extract(text):
    """Independent string parser for the same final-line contract."""
    lines = []
    marker = "final_answer"
    for line in text.splitlines():
        stripped = line.lstrip(" \t")
        if stripped[:len(marker)].lower() != marker:
            continue
        tail = stripped[len(marker):]
        if tail and (tail[0].isalnum() or tail[0] == "_"):
            continue
        lines.append(tail)
    if len(lines) != 1:
        return None
    tail = lines[0].lstrip(" \t")
    if not tail.startswith(":"):
        return None
    answer = tail[1:].strip(" \t").upper()
    if answer in ("A", "B", "C", "D"):
        return answer
    digits = answer[1:] if answer[:1] in ("+", "-") else answer
    if not digits or any(c not in "0123456789" for c in digits):
        return None
    try:
        return str(int(answer))
    except ValueError:
        return None


def summarize(rows):
    result = []
    for strategy in STRATEGIES:
        for category in ("all",) + CATEGORIES:
            group = [r for r in rows if r["strategy"] == strategy
                     and (category == "all" or r["category"] == category)]
            if not group:
                continue
            answered = [r for r in group if r["status"] == "answered"]
            calls = [a for r in group for a in r["attempts"]]
            correct = sum(r["correct"] for r in group)
            cpu = [r["cpu_seconds"] for r in group if r["cpu_seconds"] is not None]
            rss = [r["boundary_max_rss_mib"] for r in group if r["boundary_max_rss_mib"] is not None]
            result.append({
                "strategy": strategy, "category": category, "requests": len(group),
                "answered": len(answered), "correct": correct, "accuracy": correct / len(group),
                "answer_rate": len(answered) / len(group),
                "answered_accuracy": correct / len(answered) if answered else None,
                "mean_latency_seconds": sum(r["elapsed_seconds"] for r in group) / len(group),
                "total_seconds": sum(r["elapsed_seconds"] for r in group),
                "small_calls": sum(a["model"] == old.SMALL for a in calls),
                "large_calls": sum(a["model"] == old.LARGE for a in calls),
                "model_calls": len(calls), "tool_calls": sum(r["tool_calls"] for r in group),
                "common_general_adapter_requests": sum(r["general_adapter"] for r in group),
                "classification_errors": sum(r["inferred_category"] != r["category"] for r in group),
                "tool_wrong_category": sum(r["source"] == "tool" and r["category"] != "arithmetic" for r in group),
                "tool_wrong_answer": sum(r["source"] == "tool" and not r["correct"] for r in group),
                "format_failures": sum(r["status"] != "answered" and bool(r["attempts"]) for r in group),
                "cpu_samples": len(cpu), "cpu_seconds_total": sum(cpu) if len(cpu) == len(group) else None,
                "rss_samples": len(rss), "rss_boundary_max_mib": max(rss) if rss else None,
                "rss_boundary_mean_mib": statistics.mean(rss) if rss else None,
                "extraction_disagreements": sum(r["extraction_disagreement"] for r in group),
                "cost": None, "cost_status": "unmeasured",
            })
    return result


def paired(rows):
    ix = {(r["strategy"], r["question"]): r for r in rows}
    result = []
    for baseline in STRATEGIES[:-1]:
        for category in ("all",) + CATEGORIES:
            group = [r for r in rows if r["strategy"] == baseline
                     and (category == "all" or r["category"] == category)]
            common = [(r, ix[("tool", r["question"])]) for r in group
                      if r["status"] == "answered" and ix[("tool", r["question"])]["status"] == "answered"]
            if not group:
                continue
            bt = sum(a["elapsed_seconds"] for a, b in common)
            tt = sum(b["elapsed_seconds"] for a, b in common)
            result.append({"baseline": baseline, "category": category, "common_answered": len(common),
                           "baseline_correct": sum(a["correct"] for a, b in common),
                           "tool_correct": sum(b["correct"] for a, b in common),
                           "baseline_seconds": bt, "tool_seconds": tt,
                           "speed_ratio": bt / tt if tt else None,
                           "lost_answers": sum(r["status"] == "answered" and ix[("tool", r["question"])]["status"] != "answered" for r in group),
                           "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in common)})
    return result


def report(summaries, pairs, legacy, metadata):
    def fmt(v, digits=3):
        return f"{v:.{digits}f}" if v is not None else "미측정"
    def pct(v):
        return f"{v:.1%}" if v is not None else "해당 없음"
    overall = {s["strategy"]: s for s in summaries if s["category"] == "all"}
    lines = ["# 혼합 작업 실제 비교", "", f"실행 ID: `{metadata['run_id']}`.", "",
             "## 평가 조건과 기존 코드의 한계", "",
             "기존 코드의 모델 선택·정답 검증은 산술 전용입니다. 기존 코드를 보존하고 네 방식에 공통 혼합 어댑터를 추가했습니다. "
             "산술은 원래 고정/Adaptive/Coverage/Tool 정책 그대로 실행하고, 비산술은 네 방식 모두 같은 1.5B·프롬프트·온도0·출력64토큰 상한으로 실행합니다. "
             "따라서 본 결과는 원래 프로그램을 그대로 실행한 성능이 아니라 **공통 비산술 어댑터를 적용한 시스템 비교**입니다. "
             "비산술 0.5B의 품질을 보정한 근거가 없어 임의로 사용하지 않았고, 1.5B의 정확도 보장도 주장하지 않습니다.", "",
             "기존 비산술 거절을 확인하는 별도 점검에서는 원래 Adaptive/Coverage/Tool을 유형별 한 문제에 실행했습니다. "
             f"그 {len(legacy)}건 중 답변은 {sum(r['status']=='answered' for r in legacy)}건입니다. 주 평가 지표에 합산하지 않았습니다.", "",
             "새 100문제는 산술·일반 지식·논리·코딩·문장 이해 각20개입니다. 기존 CSV·JSON 문제와 정규화/교환 중복을 제외했습니다. "
             "산술 범위는 기존 난이도 규칙 안에 유지했습니다. 비산술은 객관식이며 코딩은 Python 출력 예측, 문장 이해는 짧은 지문 질문입니다. "
             "자유 코드 생성·테스트 실행·긴 문서·외부 검색·다국어 품질을 평가한 것은 아닙니다.", "",
             "## 전체 결과", "",
             "| 방식 | 전체 정확도 | 답변 비율 | 답변 정확도 | 평균 지연(ms) | 처리 시간 합계(초) | 0.5B 호출 | 1.5B 호출 | 직접 도구 호출 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy in STRATEGIES:
        s = overall[strategy]
        lines.append(f"| {strategy} | {pct(s['accuracy'])} | {pct(s['answer_rate'])} | {pct(s['answered_accuracy'])} | "
                     f"{s['mean_latency_seconds']*1000:.3f} | {s['total_seconds']:.3f} | {s['small_calls']} | {s['large_calls']} | {s['tool_calls']} |")
    lines += ["", "전체 정확도는 거절도 분모에 포함합니다. 지연은 유형 판정·도구 적용·검증·거절·fallback을 포함합니다. "
              "워밍업·사후 채점·ps 수집은 요청 지연에서 제외합니다.", "",
              "## 유형별 정확도와 답변 비율", "",
              "| 유형 | fixed 정확도/답변율 | adaptive 정확도/답변율 | coverage 정확도/답변율 | tool 정확도/답변율 |",
              "|---|---|---|---|---|"]
    ix = {(s["strategy"], s["category"]): s for s in summaries}
    for category in CATEGORIES:
        cells = [f"{pct(ix[(s,category)]['accuracy'])} / {pct(ix[(s,category)]['answer_rate'])}" for s in STRATEGIES]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "## 실제 프로세스 자원", "",
              "| 방식 | CPU 시간 합계(초) | 유효 CPU 표본 | RSS 경계 최대(MiB) | 유효 RSS 표본 |",
              "|---|---:|---:|---:|---:|"]
    for strategy in STRATEGIES:
        s = overall[strategy]
        lines.append(f"| {strategy} | {fmt(s['cpu_seconds_total'])} | {s['cpu_samples']}/100 | "
                     f"{fmt(s['rss_boundary_max_mib'],1)} | {s['rss_samples']}/100 |")
    lines += ["", "CPU는 ps가 보고한 Ollama 및 자식 프로세스 누적 시간의 요청 전후 차입니다. "
              "RSS는 두 경계 시점의 합계 최댓값이며 연속 피크나 전략별 전용 메모리가 아닙니다. "
              "상주한 다른 모델·공유 페이지·다른 클라이언트 영향을 포함할 수 있고 GPU 연산·GPU 메모리는 측정하지 않습니다. "
              "도구 계산과 어댑터 Python 프로세스의 CPU는 이 Ollama 프로세스 수치에 포함되지 않습니다. "
              "PID가 바뀌거나 접근할 수 없으면 미측정으로 둡니다. 금전 비용·전력·FLOPs는 미측정이며 절감률을 추정하지 않았습니다.", "",
              "## 공통 답변 범위의 비교", "",
              "| 기준선 | 공통 답변 | 정답: 기준선/tool | 시간: 기준선/tool(초) | 속도 비 | 잃은 답변 | 정답→오답 |",
              "|---|---:|---|---|---:|---:|---:|"]
    for p in pairs:
        if p["category"] == "all":
            lines.append(f"| {p['baseline']} | {p['common_answered']} | {p['baseline_correct']} / {p['tool_correct']} | "
                         f"{p['baseline_seconds']:.3f} / {p['tool_seconds']:.3f} | {fmt(p['speed_ratio'],2)} | "
                         f"{p['lost_answers']} | {p['correct_to_wrong']} |")
    lines += ["", "## 판정", ""]
    tool = overall["tool"]
    for baseline in STRATEGIES[:-1]:
        b = overall[baseline]
        both = tool["accuracy"] > b["accuracy"] and tool["answer_rate"] > b["answer_rate"]
        quality = tool["accuracy"] >= b["accuracy"] and tool["answer_rate"] >= b["answer_rate"]
        faster = tool["total_seconds"] < b["total_seconds"]
        lines.append(f"- vs {baseline}: 정확도·답변율 동시 엄격 개선={both}; "
                     f"집계 품질/범위 비열화={quality}; 전체 처리 시간 감소={faster}.")
    lines += ["", "위 판정은 이번 단일 측정에 한정합니다. 답변율이 같으면 동시에 증가했다고 표현하지 않습니다. "
              "비산술 80문제는 네 방식 모두 같은 모델 경로이므로 비산술 자체의 선택 효율 향상을 입증하지 않습니다. "
              "개선이 산술 20문제에만 집중되면 그 범위를 넘어 성공으로 일반화하지 않습니다. "
              "새 보정 근거 없이 0.5B를 비산술로 확장하거나 평가 결과를 보고 기준을 바꾸지 않았습니다.", "",
              "## 유형 오분류와 누출 감사", "",
              f"Tool의 유형 분류 오류 {tool['classification_errors']}건, 비산술에 계산 도구를 적용한 사례 {tool['tool_wrong_category']}건, "
              f"직접 도구 오답 {tool['tool_wrong_answer']}건, 독립 추출 파서 불일치 "
              f"{sum(s['extraction_disagreements'] for s in overall.values())}건입니다.", "",
              "분류기는 질문 원문의 수식 문법·코드 블록·지문·논리 지시어만 읽습니다. 정답/유형/선택지 정답 위치를 API에 전달하지 않습니다. "
              "유형 라벨은 분류 후 오류 집계에만 사용하며, 정답은 응답 후 채점에만 사용합니다. "
              "객관식의 모든 선택지는 일반 요청의 일부이며 어느 선택지가 정답인지는 라우터에 주지 않습니다. "
              "실행 전 문제·소스·프로필 해시를 고정했고 원본 파일 보존을 검사했습니다.", "",
              "지문에는 계산처럼 보이는 비밀번호 문자열도 포함했습니다. 다만 템플릿 지시어가 뚜렷해 유형 분류가 쉬운 데이터이며, "
              "유형별 20개·수작업 사실 문항·반복된 추론/지문 템플릿의 한계를 가집니다. "
              "분류 오류가 없더라도 일반 서비스에서의 무오류를 뜻하지 않습니다. 여러 시드·독립 데이터·자유형 채점·동시 부하 검증은 남아 있습니다.", "",
              f"전체 루프 경과 시간 {metadata['evaluation_wall_seconds']:.3f}초(모니터·채점·진행 출력 포함). "
              "기존 산술 방식의 프롬프트·재시도 정책은 보존했으므로 그 차이와 모델 선택 효과를 완전히 분리할 수 없습니다.", "",
              "## 파일", "",
              *[f"- [{suffix}]({metadata['prefix_name']}_{suffix})" for suffix in
                ("manifest.json", "evaluation.csv", "summary.csv", "paired.csv", "classification.csv", "tool_misapplications.csv", "resources.csv", "legacy_compatibility.csv", "metadata.json")], ""]
    return "\n".join(lines)


def run(data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", data_dir) for p in folder.iterdir() if p.is_file()}
    tasks = build_dataset(data_dir)
    profile = json.loads(PROFILE.read_text())
    policy = CoveragePolicy(profile)
    check_ollama([old.SMALL, old.LARGE])
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = json.load(response)
    run_id = uuid4().hex
    prefix = data_dir / f"mixed_{run_id}"
    manifest = {"seed": SEED, "tasks": tasks, "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                "policy": "unchanged arithmetic policies; common non-arithmetic 1.5B adapter",
                "preserved_sha256": {str(p): d for p, d in preserved.items()}}
    manifest_path = Path(f"{prefix}_manifest.json")
    write_json(manifest_path, manifest)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for model in (old.SMALL, old.LARGE):
        for prompt in ("legacy", old.PROMPT_VERSION):
            old.generate(model, "1 + 1", prompt)
    generate_general("Which option is a color?\nA) Red\nB) Chair\nC) Table\nD) Door")
    rows, resources = [], []
    start = time.perf_counter()
    for index, task in enumerate(tasks):
        offset = index % len(STRATEGIES)
        for strategy in STRATEGIES[offset:] + STRATEGIES[:offset]:
            before = process_snapshot()
            # Only question text and frozen calibration enter runtime.
            row = execute(task["question"], strategy, profile, policy)
            after = process_snapshot()
            predicted = extract_final(row["answer"])
            row.update(**task, predicted=predicted,
                       correct=row["status"] == "answered" and predicted == task["expected"],
                       extraction_disagreement=predicted != audit_extract(row["answer"]),
                       **resource_delta(before, after))
            rows.append(row)
            resources.append({"question": task["question"], "strategy": strategy, "before": before, "after": after})
        if (index+1) % 10 == 0:
            print(f"Actual mixed tasks {index+1}/{len(tasks)}", flush=True)
    wall = time.perf_counter()-start
    legacy = []
    for category in CATEGORIES[1:]:
        question = next(t["question"] for t in tasks if t["category"] == category)
        for strategy in ("adaptive", "coverage", "tool"):
            if strategy == "adaptive":
                row = old.execute(question, profile, "improved", 0.8)
            elif strategy == "coverage":
                row = legacy_coverage(question, policy)
            else:
                row = legacy_tool(question, profile)
            legacy.append({"strategy": strategy, "category": category, "question": question,
                           "status": row["status"], "reason": row["reason"], "model_calls": len(row["attempts"])})
    summaries, pairs = summarize(rows), paired(rows)
    for path, digest in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Original changed: {path}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("Manifest changed")
    metadata = {"run_id": run_id, "prefix_name": prefix.name, "evaluation_wall_seconds": wall,
                "ollama_tags": tags, "preserved_files_unchanged": True, "real_inference": True,
                "cost_status": "unmeasured", "manifest_sha256": manifest_hash}
    write_records(Path(f"{prefix}_evaluation.csv"), rows)
    write_records(Path(f"{prefix}_summary.csv"), summaries)
    write_records(Path(f"{prefix}_paired.csv"), pairs)
    classifications = [{"strategy": r["strategy"], "question": r["question"], "category": r["category"],
                        "inferred_category": r["inferred_category"], "source": r["source"],
                        "classification_error": r["category"] != r["inferred_category"],
                        "tool_misapplication": r["source"] == "tool" and (r["category"] != "arithmetic" or not r["correct"])} for r in rows]
    write_records(Path(f"{prefix}_classification.csv"), classifications)
    write_records(Path(f"{prefix}_tool_misapplications.csv"), [r for r in classifications if r["tool_misapplication"]], list(classifications[0]))
    write_records(Path(f"{prefix}_resources.csv"), resources)
    write_records(Path(f"{prefix}_legacy_compatibility.csv"), legacy)
    write_json(Path(f"{prefix}_metadata.json"), metadata)
    with Path(f"{prefix}_report.md").open("x", encoding="utf-8") as file:
        file.write(report(summaries, pairs, legacy, metadata))
    for s in summaries:
        if s["category"] == "all":
            print(json.dumps(s, ensure_ascii=False), flush=True)
    print(f"REPORT: {prefix}_report.md", flush=True)
    return prefix


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"혼합 평가 중단: {error}")

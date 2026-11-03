"""Frozen 200-question holdout verification. No tuning or simulated inference."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import urllib.request
from uuid import uuid4

if __package__:
    from . import adaptive_router as router
    from .adaptive_experiment import problem_key, paired_comparisons, summarize, write_json, write_csv
    from .answer_validator import validate_answer
    from .routing_runner import check_ollama
else:
    import adaptive_router as router
    from adaptive_experiment import problem_key, paired_comparisons, summarize, write_json, write_csv
    from answer_validator import validate_answer
    from routing_runner import check_ollama

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "data/adaptive_a8a1ef3394824a7cb65faf5707152986_profile.json"
STRATEGIES = ("fixed", "legacy_auto", "improved")
SEED = 20261104200


def reference_answer(question):
    """Independent AST arithmetic oracle, used outside routing only."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError("Unsupported reference expression")
    return visit(ast.parse(question, mode="eval").body)


def audit_extract(text):
    """Independent line parser checks the production regex without relaxing policy."""
    if not isinstance(text, str):
        return None
    declarations = []
    for line in text.splitlines():
        line = line.lstrip(" \t")
        marker = "final_answer"
        if line[:len(marker)].lower() != marker:
            continue
        suffix = line[len(marker):]
        if suffix and (suffix[0].isalnum() or suffix[0] == "_"):
            continue
        declarations.append(suffix)
    if len(declarations) != 1:
        return None
    suffix = declarations[0].lstrip(" \t")
    if not suffix.startswith(":"):
        return None
    value = suffix[1:].strip(" \t")
    digits = value[1:] if value[:1] in ("+", "-") else value
    if not digits or any(c not in "0123456789" for c in digits):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def prior_questions(data_dir):
    seen = {problem_key(q) for q in ("15 * 8", "144 / 12", "17 + 28", "9 ** 2", "100 - 37", "1 + 1")}
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("question"), str):
                seen.add(problem_key(value["question"]))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for path in Path(data_dir).iterdir():
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    walk(row)
        elif path.suffix == ".json":
            walk(json.loads(path.read_text(encoding="utf-8")))
    return seen


def build_holdout(data_dir):
    seen = prior_questions(data_dir)
    rng = random.Random(SEED)
    pools = {
        "easy": [f"{a} + {b}" for a in range(1, 21) for b in range(a, 21)],
        "medium": [f"{a} * {b}" for a in range(20, 100) for b in range(10, 100)],
    }
    problems = []
    for difficulty, target in (("easy", 100), ("medium", 50), ("hard", 50)):
        pool = pools.get(difficulty)
        if pool is not None:
            rng.shuffle(pool)
        chosen = 0
        for index in range(100000):
            if pool is None:
                question = f"({rng.randint(100,999)} * {rng.randint(10,99)}) - {rng.randint(10,99)}"
            else:
                if index == len(pool):
                    break
                question = pool[index]
            key = problem_key(question)
            if key in seen:
                continue
            seen.add(key)
            problems.append({"question": question, "expected": reference_answer(question),
                             "difficulty": difficulty, "bucket": router.bucket_for(question)})
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise ValueError(f"Only {chosen} unused {difficulty} questions; need {target}. No reuse allowed.")
    rng.shuffle(problems)
    return problems


def cpu_seconds(value):
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    seconds = 0.0
    for part in value.split(":"):
        seconds = seconds * 60 + float(part)
    return days * 86400 + seconds


def process_snapshot():
    """Actual ps counters for Ollama and its descendants; unavailable stays missing."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,time=,comm="],
            capture_output=True, text=True, timeout=5, check=True,
        )
        records = {}
        for line in result.stdout.splitlines():
            fields = line.split(None, 4)
            if len(fields) != 5:
                continue
            pid, ppid, rss, cpu, command = fields
            records[int(pid)] = {"ppid": int(ppid), "rss_kib": int(rss),
                                 "cpu_seconds": cpu_seconds(cpu), "command": command}
        selected = {pid for pid, rec in records.items()
                    if "ollama" in Path(rec["command"]).name.lower()}
        while True:
            children = {pid for pid, rec in records.items() if rec["ppid"] in selected}
            expanded = selected | children
            if expanded == selected:
                break
            selected = expanded
        if not selected:
            raise RuntimeError("No Ollama process visible")
        return {"time": time.perf_counter(), "duration": time.perf_counter() - start,
                "processes": {str(pid): records[pid] for pid in selected}, "error": ""}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return {"time": time.perf_counter(), "duration": time.perf_counter() - start,
                "processes": {}, "error": f"{type(error).__name__}: {error}"}


def resource_delta(before, after):
    bp, ap = before["processes"], after["processes"]
    stable = bool(bp and ap and bp.keys() == ap.keys())
    cpu = sum(max(0, ap[p]["cpu_seconds"] - bp[p]["cpu_seconds"]) for p in bp) if stable else None
    elapsed = after["time"] - before["time"]
    rss = max(sum(p["rss_kib"] for p in bp.values()),
              sum(p["rss_kib"] for p in ap.values())) / 1024 if bp and ap else None
    return {"cpu_seconds": cpu,
            "cpu_percent": cpu / elapsed * 100 if cpu is not None and elapsed > 0 else None,
            "boundary_max_rss_mib": rss, "resource_pid_stable": stable,
            "resource_error": before["error"] or after["error"] or ("PID set changed" if not stable else ""),
            "monitor_seconds": before["duration"] + after["duration"]}


def aggregate(rows):
    summaries = [r for r in summarize(rows) if r["strategy"] in STRATEGIES]
    for summary in summaries:
        group = [r for r in rows if r["strategy"] == summary["strategy"]
                 and (summary["bucket"] == "all" or r["bucket"] == summary["bucket"])]
        summary["fallback_rate"] = summary["fallback_requests"] / len(group)
        summary["fallback_rate_called"] = summary["fallback_requests"] / sum(bool(r["attempts"]) for r in group) if any(r["attempts"] for r in group) else None
        for name in ("cpu_seconds", "cpu_percent", "boundary_max_rss_mib"):
            values = [r[name] for r in group if r[name] is not None]
            summary[f"{name}_samples"] = len(values)
            summary[f"{name}_mean"] = statistics.mean(values) if values else None
            summary[f"{name}_max"] = max(values) if values else None
        summary["cpu_seconds_total"] = sum(r["cpu_seconds"] for r in group) if all(r["cpu_seconds"] is not None for r in group) else None
        summary["audit_mismatches"] = sum(r["audit_mismatches"] for r in group)
    return summaries


def report_markdown(summaries, pairs, metadata):
    overall = {r["strategy"]: r for r in summaries if r["bucket"] == "all"}
    labels = {"fixed": "고정 1.5B", "legacy_auto": "기존 자동", "improved": "Adaptive"}
    def num(value, digits=3):
        return f"{value:.{digits}f}" if value is not None else "미측정"
    lines = ["# 신규 200문제 실제 검증", "",
             f"실행 ID: `{metadata['run_id']}`. 기존 프로필·프롬프트·80% 임계값 고정, 재학습 없음.", "",
             "## 전체 결과", "",
             "전체 정확도는 거절도 분모에 포함하며 답변 정확도는 유효한 최종 정수를 반환한 요청만 집계합니다.", "",
             "| 방식 | 정답 | 전체 정확도 | 답변 정확도 | 답변 비율 | 평균 지연 | 처리 시간 합계 | 0.5B / 1.5B / 미선택 | fallback |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for strategy in STRATEGIES:
        s = overall[strategy]
        ratios = s["final_model_ratios"]
        lines.append(f"| {labels[strategy]} | {s['correct']}/200 | {s['accuracy']:.1%} | "
                     f"{s['answered_accuracy']:.1%} | {s['answer_rate']:.1%} | {s['mean_latency']:.3f}초 | "
                     f"{s['total_seconds']:.3f}초 | {ratios[router.SMALL]:.1%} / {ratios[router.LARGE]:.1%} / {ratios['no_selection']:.1%} | {s['fallback_rate']:.1%} |")
    lines += ["", "fallback은 실제 후보를 2회 이상 호출한 요청 / 전체 요청입니다. 신뢰 기준으로 작은 모델을 건너뛰는 것은 재호출 fallback에 포함하지 않습니다.",
              "지연은 라우팅과 모든 fallback을 포함하고 ps 수집은 제외합니다. 거절도 포함한 평균이므로 이것만으로 효율 우위를 판단하지 않습니다.",
              "", "## 동일 질문·동일 답변 범위 비교", "",
              "| 비교 대상 | 공통 답변 | 잃은 답변 | 공통 정답: 대상 / Adaptive | 대상 평균 | Adaptive 평균 | 관측 속도 비 |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {labels[p['baseline']]} | {p['common_answered']} | {p['lost_answers']} | "
                     f"{p['baseline_correct']} / {p['improved_correct']} | {num(p['baseline_mean_seconds'])}초 | "
                     f"{num(p['improved_mean_seconds'])}초 | {num(p['same_coverage_speedup'], 2)}배 |")
    lines += ["", "## 실제 프로세스 자원 계측", "",
              "| 방식 | CPU 누적 시간 차 합계 | 평균 CPU % | 경계 샘플 최대 RSS | 유효 CPU 샘플 |",
              "|---|---:|---:|---:|---:|"]
    for strategy in STRATEGIES:
        s = overall[strategy]
        lines.append(f"| {labels[strategy]} | {num(s['cpu_seconds_total'])}초 | {num(s['cpu_percent_mean'],1)}% | "
                     f"{num(s['boundary_max_rss_mib_max'],1)} MiB | {s['cpu_seconds_samples']}/200 |")
    fixed, old, new = (overall[s] for s in STRATEGIES)
    old_pair = next(p for p in pairs if p["baseline"] == "legacy_auto")
    preserved = (old_pair["lost_answers"] == 0 and new["accuracy"] >= old["accuracy"]
                 and new["answered_accuracy"] >= old["answered_accuracy"])
    lines += ["", "## 객관적 판단", "",
              f"전체 정답 수가 가장 많은 방식: **{labels[max(overall, key=lambda s: overall[s]['correct'])]}**.",
              f"기존 자동 대비 평가 집합에서 답변 범위·정확도 유지: **{'확인' if preserved else '미충족'}**."]
    if old_pair["same_coverage_speedup"]:
        lines.append(f"공통 답변의 Adaptive 관측 속도 비는 {old_pair['same_coverage_speedup']:.2f}배입니다. "
                     "1보다 크면 빠르고 작으면 느립니다. 단일 실행의 관측값이며 통계적 우위 확정은 아닙니다.")
    if new["answer_rate"] < fixed["answer_rate"]:
        lines.append("Adaptive는 고정 1.5B의 전체 답변 범위를 유지하지 못합니다. "
                     "거절로 줄어든 총 시간을 동일 서비스의 성능 향상으로 해석하지 않습니다.")
    lines += ["", "## 데이터 누출·추출 감사", "",
              "- 문제 세트를 추론 전에 저장하고 해시로 고정했습니다. 기존 모든 CSV와 JSON의 question 필드를 제외했습니다.",
              "- 덧셈·곱셈의 교환 중복도 제거했으며 200개 모두 고유합니다. 정답은 별도 AST 계산으로 검증했습니다.",
              "- 정답은 execute/plan에 전달하지 않으며 모든 응답이 끝난 뒤 채점에만 사용했습니다.",
              f"- 독립 파서와 기존 검증기 불일치: {sum(s['audit_mismatches'] for s in overall.values())}건.",
              f"- 시도별 형식 오류: {sum(s['format_errors'] for s in overall.values())}건. API 오류: {sum(s['api_errors'] for s in overall.values())}건.",
              "- 원본 응답과 추출 결과, 시도·오류 내역은 상세 CSV에 저장했습니다. 형식이 틀린 답을 느슨하게 재채점하지 않았습니다.",
              "", "## 측정 범위와 한계", "",
              "- 실제 Ollama 호출 결과입니다. 테스트의 모의 응답은 평가 데이터에 포함하지 않았습니다.",
              "- 덧셈 100개·곱셈 50개·복합식 50개이며 일반 자연어 작업의 대표 표본은 아닙니다.",
              "- 이전 실험에서 작은 덧셈 영역을 많이 사용해, 새로운 세트의 작은/큰 덧셈 비율은 균등하지 않습니다. splits.json의 구성을 참조하십시오.",
              "- CPU는 ps의 Ollama 및 자식 프로세스 누적 CPU 시간 차입니다. GPU 연산 시간은 포함하지 않습니다.",
              "- RSS는 요청 전후 두 경계 샘플 합계의 최댓값이며 연속 측정 피크가 아닙니다. 상주한 다른 모델 메모리와 공유 페이지 중복이 포함될 수 있습니다.",
              "- PID가 바뀌거나 프로세스를 읽지 못하면 CPU는 미측정입니다. CPU %는 여러 코어 합산으로 100%를 넘을 수 있습니다.",
              "- 다른 Ollama 클라이언트와 백그라운드 작업을 완전히 통제하지 못했습니다. 거절 요청의 짧은 구간 CPU 수치는 특히 해상도 영향을 받습니다.",
              "- 프로세스 모니터 실행 비용은 요청 지연에서 제외하고 전체 루프 경과 시간에는 포함합니다.",
              "- 전력·FLOPs·GPU 메모리·금전 비용은 미측정입니다. 금전 추정치도 만들지 않았습니다. 토큰·생성 시간은 측정된 작업 지표입니다.",
              "- 기존 프롬프트와 Adaptive 프롬프트가 다릅니다. 이번 세 방식 비교만으로 프롬프트 효과와 모델 크기 효과를 완전히 분리할 수 없습니다.",
              "- 프로필의 이전 보정 비용은 이번 요청 처리 시간에서 제외됐습니다. 반복·장시간·독립 장비 검증은 수행하지 않았습니다.",
              f"- 워밍업 제외 전체 평가 루프: {metadata['evaluation_wall_seconds']:.3f}초. 프로세스 수집·채점·진행 출력 포함.",
              "", "## 파일", "",
              *[f"- [{suffix}]({metadata['prefix_name']}_{suffix})" for suffix in
                ("splits.json", "evaluation.csv", "summary.csv", "paired.csv", "resources.csv", "metadata.json")], ""]
    return "\n".join(lines)


def run(data_dir=ROOT / "data", profile_path=PROFILE):
    data_dir = Path(data_dir)
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (ROOT / "src", data_dir) for p in folder.iterdir() if p.is_file()}
    profile_bytes = Path(profile_path).read_bytes()
    profile = json.loads(profile_bytes)
    router.plan("2 + 3", profile, 0.8)  # Validate frozen profile before network or artifacts.
    problems = build_holdout(data_dir)
    check_ollama([router.SMALL, router.LARGE])
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = json.load(response)
    run_id = uuid4().hex
    prefix = data_dir / f"verification200_{run_id}"
    dataset = {"seed": SEED, "problems": problems, "min_quality": 0.8,
               "profile": str(profile_path), "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
               "prior_sha256": {str(p): digest for p, digest in preserved.items()}}
    write_json(Path(f"{prefix}_splits.json"), dataset)
    dataset_hash = hashlib.sha256(Path(f"{prefix}_splits.json").read_bytes()).hexdigest()
    for model in (router.SMALL, router.LARGE):
        for prompt in ("legacy", router.PROMPT_VERSION):
            router.generate(model, "1 + 1", prompt)
    rows, resource_rows = [], []
    orders = list(itertools.permutations(STRATEGIES))
    start = time.perf_counter()
    for index, problem in enumerate(problems):
        for strategy in orders[index % len(orders)]:
            before = process_snapshot()
            row = router.execute(problem["question"], profile, strategy, 0.8)
            after = process_snapshot()
            # Scoring and audit happen after generation; no expected value enters execute.
            validation = validate_answer(row["answer"], problem["expected"])
            mismatches = 0
            for attempt in row["attempts"]:
                text = attempt.get("answer", "")
                production = validate_answer(text, problem["expected"])["predicted"]
                mismatches += audit_extract(text) != production
            row.update(**problem, predicted=validation["predicted"],
                       correct=row["status"] == "answered" and validation["correct"],
                       audit_mismatches=mismatches, **resource_delta(before, after))
            rows.append(row)
            resource_rows.append({"question": problem["question"], "strategy": strategy,
                                  "before": before, "after": after})
        if (index + 1) % 20 == 0:
            print(f"Actual evaluation: {index+1}/200 ({(index+1)*3} strategy requests)", flush=True)
    wall = time.perf_counter() - start
    summaries = aggregate(rows)
    pairs = [p for p in paired_comparisons(rows) if p["baseline"] in ("fixed", "legacy_auto")]
    for path, digest in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing file changed: {path}")
    if hashlib.sha256(Path(f"{prefix}_splits.json").read_bytes()).hexdigest() != dataset_hash:
        raise RuntimeError("Holdout manifest changed")
    metadata = {"run_id": run_id, "prefix_name": prefix.name,
                "evaluation_wall_seconds": wall, "python": platform.python_version(),
                "platform": platform.platform(), "cpu_count": os.cpu_count(), "ollama_tags": tags,
                "profile_sha256": dataset["profile_sha256"], "dataset_sha256": dataset_hash,
                "bucket_counts": dict(Counter(p["bucket"] for p in problems)),
                "prior_files_unchanged": True, "real_inference": True, "cost_status": "unmeasured"}
    write_csv(Path(f"{prefix}_evaluation.csv"), rows)
    write_csv(Path(f"{prefix}_summary.csv"), summaries)
    write_csv(Path(f"{prefix}_paired.csv"), pairs)
    write_csv(Path(f"{prefix}_resources.csv"), resource_rows)
    write_json(Path(f"{prefix}_metadata.json"), metadata)
    with Path(f"{prefix}_report.md").open("x", encoding="utf-8") as file:
        file.write(report_markdown(summaries, pairs, metadata))
    for summary in summaries:
        if summary["bucket"] == "all":
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    for pair in pairs:
        print(json.dumps(pair, ensure_ascii=False), flush=True)
    print(f"REPORT: {prefix}_report.md", flush=True)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실제 Ollama 신규 200문제 고정 정책 검증")
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    try:
        run(profile_path=args.profile)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"검증 중단: {error}\n")

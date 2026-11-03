"""Read-only post-run audit; distinguish model selection, format failure and refusal."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

if __package__:
    from . import verify_routing200 as verification
else:
    import verify_routing200 as verification


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["attempts"] = json.loads(row["attempts"])
        row["correct"] = row["correct"] == "True"
        row["expected"] = int(row["expected"])
        row["predicted"] = int(row["predicted"]) if row["predicted"] else None
        row["audit_mismatches"] = int(row["audit_mismatches"])
        for key in ("elapsed_seconds", "cpu_seconds", "cpu_percent", "boundary_max_rss_mib"):
            row[key] = float(row[key]) if row[key] else None
    return rows


def enrich(summaries, pairs, rows):
    indexed = {(r["strategy"], r["question"]): r for r in rows}
    for summary in summaries:
        group = [r for r in rows if r["strategy"] == summary["strategy"]
                 and (summary["bucket"] == "all" or r["bucket"] == summary["bucket"])]
        counts = Counter(r["attempts"][0]["model"] if r["attempts"] else "no_selection" for r in group)
        summary["primary_selection_ratios"] = {m: counts[m] / len(group) for m in
                                              (verification.router.SMALL, verification.router.LARGE, "no_selection")}
        summary["policy_refusals"] = sum(not r["attempts"] for r in group)
        summary["failed_selected_requests"] = sum(bool(r["attempts"]) and r["status"] != "answered" for r in group)
    for pair in pairs:
        common = [(r, indexed[("improved", r["question"])]) for r in rows
                  if r["strategy"] == pair["baseline"] and r["status"] == "answered"
                  and indexed[("improved", r["question"])]["status"] == "answered"]
        pair["correct_to_wrong"] = sum(a["correct"] and not b["correct"] for a, b in common)
        pair["wrong_to_correct"] = sum(not a["correct"] and b["correct"] for a, b in common)
        for label, position in (("baseline", 0), ("improved", 1)):
            values = [p[position]["cpu_seconds"] for p in common]
            pair[f"{label}_common_cpu_seconds"] = sum(values) if all(v is not None for v in values) else None
    return summaries, pairs


def audit(prefix):
    prefix = Path(prefix)
    dataset = json.loads(Path(f"{prefix}_splits.json").read_text())
    metadata = json.loads(Path(f"{prefix}_metadata.json").read_text())
    assert hashlib.sha256(Path(f"{prefix}_splits.json").read_bytes()).hexdigest() == metadata["dataset_sha256"]
    for name, digest in dataset["prior_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prior file changed: {name}")
    keys = {verification.problem_key(p["question"]) for p in dataset["problems"]}
    if len(keys) != 200:
        raise ValueError("Holdout must contain 200 unique problems")
    for name in dataset["prior_sha256"]:
        path = Path(name)
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as file:
                previous = {verification.problem_key(r["question"]) for r in csv.DictReader(file) if r.get("question")}
            if keys & previous:
                raise ValueError("Holdout overlaps prior CSV")
    rows = load_rows(Path(f"{prefix}_evaluation.csv"))
    if len(rows) != 600:
        raise ValueError("Expected exactly 600 strategy records")
    for strategy in verification.STRATEGIES:
        group = [r for r in rows if r["strategy"] == strategy]
        if len(group) != 200 or {verification.problem_key(r["question"]) for r in group} != keys:
            raise ValueError("Strategies used different question sets")
    format_cases = []
    for row in rows:
        if row["expected"] != verification.reference_answer(row["question"]):
            raise ValueError("Invalid reference answer")
        actual = verification.validate_answer(row["answer"], row["expected"])
        if actual["predicted"] != row["predicted"] or row["correct"] != (row["status"] == "answered" and actual["correct"]):
            raise ValueError("Incorrect stored score")
        for attempt in row["attempts"]:
            text = attempt.get("answer", "")
            if verification.audit_extract(text) != verification.validate_answer(text, row["expected"])["predicted"]:
                raise ValueError("Independent extraction disagreement")
            if attempt["status"] == "extraction_failed":
                format_cases.append({"strategy": row["strategy"], "question": row["question"],
                                     "expected": row["expected"], "raw_response": text})
    summaries = verification.aggregate(rows)
    pairs = [p for p in verification.paired_comparisons(rows) if p["baseline"] in ("fixed", "legacy_auto")]
    enrich(summaries, pairs, rows)
    report = verification.report_markdown(summaries, pairs, metadata)
    report = report.replace("| 0.5B / 1.5B / 미선택 |", "| 최종 유효 답변: 0.5B / 1.5B / 미답변 |")
    report = report.replace("기존 자동 대비 평가 집합에서 답변 범위·정확도 유지:", "기존 자동 대비 집계 정확도·답변 범위 유지:")
    report += "\n## 추가 감사: 실제 선택과 문항별 품질\n\n"
    report += "상단 모델 비율은 최종 유효 답변 기준입니다. 형식 실패는 미선택이 아니라 호출 후 미답변입니다. 실제 첫 모델 선택은 다음과 같습니다.\n\n"
    report += "| 방식 | 0.5B 선택 | 1.5B 선택 | 정책 거절 | 선택 후 실패 |\n|---|---:|---:|---:|---:|\n"
    for summary in summaries:
        if summary["bucket"] == "all":
            ratios = summary["primary_selection_ratios"]
            report += f"| {summary['strategy']} | {ratios[verification.router.SMALL]:.1%} | {ratios[verification.router.LARGE]:.1%} | {summary['policy_refusals']} | {summary['failed_selected_requests']} |\n"
    report += "\n공통 답변의 정답 수가 같아도 같은 문항을 모두 맞혔다는 뜻은 아닙니다.\n\n"
    for pair in pairs:
        report += (f"- {pair['baseline']} 대비 정답→오답 {pair['correct_to_wrong']}건, 오답→정답 {pair['wrong_to_correct']}건. "
                   f"공통 답변 CPU 시간: 대상 {pair['baseline_common_cpu_seconds']:.3f}초, Adaptive {pair['improved_common_cpu_seconds']:.3f}초.\n")
    for row in rows:
        if row["strategy"] == "improved" and row["status"] == "answered" and not row["correct"]:
            report += f"\nAdaptive 오답: `{row['question']}`, 정답 `{row['expected']}`, 실제 응답 `{row['answer']}`. "
            report += "형식은 올바르므로 실행 시 정답을 모르는 라우터는 이를 탐지하지 못했습니다.\n"
    report += "\n형식 오류 원문(임의 복구 없이 기존 엄격한 기준을 유지):\n\n"
    for case in format_cases:
        report += f"- {case['strategy']}, `{case['question']}`: `{case['raw_response']}`\n"
    report += ("\n따라서 Adaptive는 기존 자동 대비 집계 품질·응답 범위를 유지하면서 빠른 선택지였지만, "
               "개별 요청 품질의 무손실 대체나 모든 지표에서의 우위를 증명한 것은 아닙니다. "
               "전체 정답 수·응답 범위를 우선하면 이번 실험에서는 고정 1.5B가 우수합니다. "
               "금전 비용은 실제·추정 모두 미측정입니다.\n")
    report += f"\n감사 후 CSV: `{prefix.name}_audited_summary.csv`, `{prefix.name}_audited_pairs.csv`. "
    report += "원시 결과와 최초 자동 보고서는 보존했습니다.\n"
    verification.write_csv(Path(f"{prefix}_audited_summary.csv"), summaries)
    verification.write_csv(Path(f"{prefix}_audited_pairs.csv"), pairs)
    with Path(f"{prefix}_audited_report.md").open("x", encoding="utf-8") as file:
        file.write(report)
    print(f"AUDITED REPORT: {prefix}_audited_report.md")
    return summaries, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실제 200문제 결과 사후 감사; 모델 재실행 없음")
    parser.add_argument("prefix", type=Path)
    audit(parser.parse_args().prefix)

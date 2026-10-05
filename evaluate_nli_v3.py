import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_nli import PREFIX, auc, run_nli

THRESHOLDS = [0.3, 0.5, 0.7]
HARMLESS = ["identical", "paraphrase"]
VIOLATIONS = ["negation", "number", "entity", "scope"]


def nli(premises, hypotheses):
    probs, labels = run_nli(premises, hypotheses)
    p = pd.DataFrame(probs, columns=labels)
    return p["entailment"].values, p["contradiction"].values


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("testset_v2.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results_v2")
    df = pd.read_csv(data_path)

    ent_f, con_f = nli(PREFIX + df["state"], PREFIX + df["intent"])
    ent_r, _ = nli(PREFIX + df["intent"], PREFIX + df["state"])

    df["p_entail_fwd"] = ent_f
    df["p_contra_fwd"] = con_f
    df["p_entail_rev"] = ent_r
    df["score_a"] = 1 - ent_f                      # 기존 방식
    df["score_b"] = con_f                          # 모순 확률만
    df["score_c"] = 1 - np.minimum(ent_f, ent_r)   # 양방향 중 약한 쪽

    scorers = {"A: 1-entail": "score_a", "B: contradiction": "score_b", "C: 1-min(fwd,rev)": "score_c"}
    harmless = df[df["kind"].isin(HARMLESS)]
    para = df[df["kind"] == "paraphrase"]
    viol = df[df["kind"].isin(VIOLATIONS)]

    rows = []
    for name, col in scorers.items():
        r = {"scorer": name, "AUC(violation vs harmless)": round(auc(viol[col], harmless[col]), 3)}
        for t in THRESHOLDS:
            r[f"false_alarm@{t}"] = round(float((para[col] > t).mean()), 2)
        for t in THRESHOLDS:
            r[f"detect@{t}"] = round(float((viol[col] > t).mean()), 2)
        rows.append(r)
    summary = pd.DataFrame(rows).set_index("scorer")

    kinds = HARMLESS + VIOLATIONS
    by_kind = pd.DataFrame(
        {name: (df[col] > 0.5).groupby(df["kind"]).mean() for name, col in scorers.items()}
    ).reindex([k for k in kinds if k in set(df["kind"])]).round(2)

    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "scores_nli_v3.csv", index=False)
    summary.to_csv(out_dir / "experiment4_summary.csv")
    by_kind.to_csv(out_dir / "experiment4_by_kind.csv")

    print(f"\nRows: {len(df)}  harmless: {len(harmless)}  violations: {len(viol)}")
    print("\n== Experiment 4: scoring rules ==")
    print("(false_alarm = paraphrase flagged, want low / detect = violations flagged, want high)")
    print(summary.to_string())
    print("\n== Share flagged at 0.5, by kind ==")
    print(by_kind.to_string())
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()

import sys

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results_v2/scores_nli_v3.csv"
d = pd.read_csv(path)

DRIFT = ["negation", "number", "entity", "scope", "different_task", "part_dropped", "unrelated"]
SCORERS = {"A": "score_a", "B": "score_b", "C": "score_c"}
GRID = np.round(np.arange(0.1, 0.96, 0.05), 2)
N_SPLITS = 200


def metrics(df, col, t):
    """False alarm on paraphrase rows; detection averaged over the 7 drift kinds."""
    fa = (df.loc[df["kind"] == "paraphrase", col] > t).mean()
    det = np.mean([(df.loc[df["kind"] == k, col] > t).mean() for k in DRIFT])
    return fa, det


def pick_threshold(train, col):
    best, best_t = -9.0, None
    for t in GRID:
        fa, det = metrics(train, col, t)
        score = det - fa
        if score > best + 1e-12:
            best, best_t = score, t
    return best_t


rng = np.random.default_rng(0)
rows = []
for _ in range(N_SPLITS):
    train_idx = []
    for _, g in d.groupby("kind"):
        idx = rng.permutation(g.index.values)
        train_idx += list(idx[: len(idx) // 2])
    train, test = d.loc[train_idx], d.drop(train_idx)
    for name, col in SCORERS.items():
        t = pick_threshold(train, col)
        fa, det = metrics(test, col, t)
        fa5, det5 = metrics(test, col, 0.5)
        rows.append({"rule": name, "thr": t, "fa": fa, "det": det, "fa05": fa5, "det05": det5})

r = pd.DataFrame(rows)
summary = r.groupby("rule").agg(
    thr_mean=("thr", "mean"),
    thr_std=("thr", "std"),
    test_false_alarm=("fa", "mean"),
    test_detection=("det", "mean"),
    false_alarm_at_0_5=("fa05", "mean"),
    detection_at_0_5=("det05", "mean"),
).round(3)

a = r[r["rule"] == "A"].reset_index(drop=True)
b = r[r["rule"] == "B"].reset_index(drop=True)
wins = ((a["det"] - a["fa"]) > (b["det"] - b["fa"])).mean()

print(f"\nRows: {len(d)}   splits: {N_SPLITS}   (threshold chosen on one half, measured on the other)")
print("detection = mean over 7 drift kinds; false alarm = paraphrase rows only\n")
print(summary.to_string())
print(f"\nA beats B (detection - false alarm) in {wins:.0%} of splits")
summary.to_csv("results_v2/experiment4_heldout.csv")
print("Saved: results_v2/experiment4_heldout.csv")

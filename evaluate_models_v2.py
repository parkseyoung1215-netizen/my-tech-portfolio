import numpy as np
import pandas as pd

import evaluate_nli as E
from evaluate_nli import PREFIX

MODELS = {
    "MiniLM": "cross-encoder/nli-MiniLM2-L6-H768",
    "DeBERTa": "cross-encoder/nli-deberta-v3-base",
}
DRIFT = ["negation", "number", "entity", "scope", "different_task", "part_dropped", "unrelated"]
GRID = np.round(np.arange(0.1, 0.96, 0.05), 2)
N_SPLITS = 200

base = pd.read_csv("testset_v2.csv")
swap = pd.read_csv("testset_swap.csv")
both = pd.concat([base, swap], ignore_index=True)

for name, m in MODELS.items():
    print(f">>> scoring with {name} ...", flush=True)
    E.NLI_MODEL = m
    probs, labels = E.run_nli(PREFIX + both["state"], PREFIX + both["intent"])
    both["a_" + name] = 1 - pd.DataFrame(probs, columns=labels)["entailment"].values
both.to_csv("results_v2/scores_models.csv", index=False)

core = both[both["kind"] != "direction_swap"].reset_index(drop=True)
swp = both[both["kind"] == "direction_swap"]


def metrics(df, col, t):
    fa = (df.loc[df["kind"] == "paraphrase", col] > t).mean()
    det = np.mean([(df.loc[df["kind"] == k, col] > t).mean() for k in DRIFT])
    return fa, det


def pick(train, col):
    best, best_t = -9.0, None
    for t in GRID:
        fa, det = metrics(train, col, t)
        if det - fa > best + 1e-12:
            best, best_t = det - fa, t
    return best_t


rng = np.random.default_rng(0)
res = {n: [] for n in MODELS}
for _ in range(N_SPLITS):
    tr = []
    for _, g in core.groupby("kind"):
        idx = rng.permutation(g.index.values)
        tr += list(idx[: len(idx) // 2])
    train, test = core.loc[tr], core.drop(tr)
    for n in MODELS:
        col = "a_" + n
        t = pick(train, col)
        fa, det = metrics(test, col, t)
        res[n].append((t, fa, det, (swp[col] > t).mean()))

rows = {}
for n, v in res.items():
    a = np.array(v)
    rows[n] = {
        "threshold chosen (mean)": a[:, 0].mean(),
        "threshold chosen (sd)": a[:, 0].std(),
        "held-out false alarm": a[:, 1].mean(),
        "held-out detection": a[:, 2].mean(),
        "swaps detected at that threshold": a[:, 3].mean(),
    }
table = pd.DataFrame(rows).round(3)
print("\n== Per-model threshold, chosen on one half and measured on the other ==")
print("Swap pairs are not used to choose the threshold.\n")
print(table.to_string())
table.to_csv("results_v2/experiment7_per_model_thresholds.csv")

fa = both[(both["kind"] == "paraphrase") & (both["a_DeBERTa"] > 0.75)]
print(f"\nDeBERTa false alarms on paraphrases (score > 0.75): {len(fa)}")
for _, r in fa.iterrows():
    print("\nINTENT:", r["intent"])
    print("STATE: ", r["state"])
    print("MiniLM=%.2f  DeBERTa=%.2f" % (r["a_MiniLM"], r["a_DeBERTa"]))
print("\nSaved: results_v2/experiment7_per_model_thresholds.csv and results_v2/scores_models.csv")

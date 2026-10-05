import sys

import numpy as np
import pandas as pd

import evaluate_nli as E
from evaluate_nli import PREFIX, auc

MODELS = sys.argv[1:] or [
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-v3-base",
]

base = pd.read_csv("testset_v2.csv")
swap = pd.read_csv("testset_swap.csv")
both = pd.concat([base, swap], ignore_index=True)

summary = {}
for m in MODELS:
    print(f"\n>>> running {m} on {len(both)} pairs ...", flush=True)
    E.NLI_MODEL = m
    probs, labels = E.run_nli(PREFIX + both["state"], PREFIX + both["intent"])
    p = pd.DataFrame(probs, columns=labels)
    both["score_a"] = 1 - p["entailment"].values

    harmless = both[both["kind"].isin(["identical", "paraphrase"])]
    para = both[both["kind"] == "paraphrase"]
    viol4 = both[both["kind"].isin(["negation", "number", "entity", "scope"])]
    swp = both[both["kind"] == "direction_swap"]

    summary[m.split("/")[-1]] = {
        "AUC violations vs harmless": auc(viol4["score_a"], harmless["score_a"]),
        "AUC swaps vs paraphrase": auc(swp["score_a"], para["score_a"]),
        "false alarm @0.5": (para["score_a"] > 0.5).mean(),
        "false alarm @0.75": (para["score_a"] > 0.75).mean(),
        "detect violations @0.5": (viol4["score_a"] > 0.5).mean(),
        "detect violations @0.75": (viol4["score_a"] > 0.75).mean(),
        "detect swaps @0.5": (swp["score_a"] > 0.5).mean(),
        "detect swaps @0.75": (swp["score_a"] > 0.75).mean(),
        "mean score of swaps": swp["score_a"].mean(),
    }

table = pd.DataFrame(summary).round(3)
print("\n== Same test sets, different NLI models (scorer A = 1 - P(entailment)) ==")
print("Thresholds are the same for every model, not tuned per model.\n")
print(table.to_string())
table.to_csv("results_v2/experiment6_models.csv")
print("\nSaved: results_v2/experiment6_models.csv")

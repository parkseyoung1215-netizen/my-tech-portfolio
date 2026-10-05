import re
import sys

import numpy as np
import pandas as pd

from evaluate_nli import PREFIX, run_nli

swap_path = sys.argv[1] if len(sys.argv) > 1 else "testset_swap.csv"
swap = pd.read_csv(swap_path)
base = pd.read_csv("testset_v2.csv")


def nli_scores(df):
    probs, labels = run_nli(PREFIX + df["state"], PREFIX + df["intent"])
    p = pd.DataFrame(probs, columns=labels)
    return 1 - p["entailment"].values, p["contradiction"].values


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def same_words_new_order(a, b):
    wa, wb = words(a), words(b)
    return sorted(wa) == sorted(wb) and wa != wb


swap["score_a"], swap["score_b"] = nli_scores(swap)
swap["rule_d"] = [same_words_new_order(i, s) for i, s in zip(swap["intent"], swap["state"])]
swap["pure_reorder"] = swap["rule_d"]

para = base[base["kind"] == "paraphrase"]
fa_d = np.mean([same_words_new_order(i, s) for i, s in zip(para["intent"], para["state"])])

print(f"\nswap rows: {len(swap)}   (pure reorder: {int(swap['pure_reorder'].sum())})")
print("Thresholds fixed before this run: A at 0.5 and 0.75, B at 0.5.")
print("Rule D = same words in a different order (no NLI).\n")
rows = {
    "A > 0.5": (swap["score_a"] > 0.5).mean(),
    "A > 0.75": (swap["score_a"] > 0.75).mean(),
    "B > 0.5": (swap["score_b"] > 0.5).mean(),
    "D (word order)": swap["rule_d"].mean(),
}
print("Detection of direction swaps:")
for k, v in rows.items():
    print(f"  {k:16s} {v:.0%}")
print(f"\nRule D false alarm on the 60 paraphrase rows of testset_v2: {fa_d:.0%}")

missed = swap[swap["score_a"] <= 0.75]
print(f"\nMissed by A > 0.75: {len(missed)}")
with pd.option_context("display.max_colwidth", 80, "display.width", 220):
    print(missed[["intent", "state", "score_a"]].round(2).to_string(index=False))

swap.to_csv("results_v2/experiment5_swap_scores.csv", index=False)
print("\nSaved: results_v2/experiment5_swap_scores.csv")

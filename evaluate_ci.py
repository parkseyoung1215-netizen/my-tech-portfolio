import numpy as np
import pandas as pd

d = pd.read_csv("results_v2/scores_models.csv")

# Thresholds fixed in advance: MiniLM 0.75 (mean chosen in Experiment 7), DeBERTa 0.5 (default).
CONFIG = {"MiniLM": ("a_MiniLM", 0.75), "DeBERTa": ("a_DeBERTa", 0.5)}
DRIFT = ["negation", "number", "entity", "scope", "different_task", "part_dropped", "unrelated"]
N_BOOT = 2000
rng = np.random.default_rng(0)


def stats(frame, col, t):
    flag = frame[col] > t
    kind = frame["kind"]
    return {
        "false alarm (paraphrase)": flag[kind == "paraphrase"].mean(),
        "detection (7 drift kinds, mean)": np.mean([flag[kind == k].mean() for k in DRIFT]),
        "swaps detected": flag[kind == "direction_swap"].mean(),
    }


groups = [g for _, g in d.groupby("kind")]


def resample():
    parts = [g.sample(len(g), replace=True, random_state=int(rng.integers(1_000_000_000))) for g in groups]
    return pd.concat(parts)


rows = {}
for name, (col, t) in CONFIG.items():
    point = stats(d, col, t)
    boots = pd.DataFrame([stats(resample(), col, t) for _ in range(N_BOOT)])
    lo, hi = boots.quantile(0.025), boots.quantile(0.975)
    rows[f"{name} (threshold {t})"] = {
        k: f"{point[k]:.1%}  [{lo[k]:.1%}, {hi[k]:.1%}]" for k in point
    }

table = pd.DataFrame(rows)
print(f"\n95% bootstrap intervals ({N_BOOT} resamples, rows resampled within each kind)\n")
print(table.to_string())
table.to_csv("results_v2/experiment8_confidence_intervals.csv")
print("\nSaved: results_v2/experiment8_confidence_intervals.csv")

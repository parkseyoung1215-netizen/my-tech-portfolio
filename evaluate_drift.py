"""evaluate_drift.py

Does the divergence score track how far an instruction has drifted?

Reads a CSV with columns intent, state, level (and optionally kind), scores
every row, and prints the mean score per drift level, the rank correlation
between level and score, and how often each threshold raises an alert.
Saves <out_dir>/scores.csv and <out_dir>/drift_by_level.png.

Drift levels:
  0 identical     1 reworded only    2 part dropped
  3 condition flipped    4 related but different task    5 unrelated

Usage:
  python3 evaluate_drift.py                                  # testset.csv -> results/
  python3 evaluate_drift.py testset_v2.csv results_v2        # bigger set -> results_v2/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "all-MiniLM-L6-v2"  # same model as simulate_drift.py
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ALERT_THRESHOLD = 0.3


def divergence_scores(intents, states):
    """1 - cosine similarity, clipped to [0, 1] (same score as simulate_drift.py)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    a = model.encode(list(intents), normalize_embeddings=True)
    b = model.encode(list(states), normalize_embeddings=True)
    cos = np.sum(a * b, axis=1)
    return np.clip(1.0 - cos, 0.0, 1.0)


def analyze(df):
    by_level = (
        df.groupby("level")["score"].agg(["count", "mean", "std", "min", "max"]).round(4)
    )
    rho = df["level"].rank().corr(df["score"].rank())  # Spearman correlation
    means = by_level["mean"].tolist()
    monotonic = all(x < y for x, y in zip(means, means[1:]))
    alerts = pd.DataFrame(
        {f">{t}": (df["score"] > t).groupby(df["level"]).mean() for t in THRESHOLDS}
    ).round(2)
    return by_level, rho, monotonic, alerts


def kind_table(df):
    """Mean score and alert rate per (level, kind). Only used when a kind column exists."""
    d = df.assign(flagged=df["score"] > ALERT_THRESHOLD)
    g = d.groupby(["level", "kind"])
    t = g.agg(n=("score", "size"), mean=("score", "mean"), flagged=("flagged", "mean"))
    t = t.rename(columns={"flagged": f"flagged>{ALERT_THRESHOLD}"})
    return t.round(3)


def make_plot(df, by_level, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        df["level"] + rng.uniform(-0.12, 0.12, len(df)),
        df["score"],
        alpha=0.5,
        s=22,
    )
    ax.plot(by_level.index, by_level["mean"], marker="o", color="black", label="mean")
    ax.set_xlabel("drift level (0 = identical, 5 = unrelated)")
    ax.set_ylabel("divergence score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Divergence score vs. drift level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("testset.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    df = pd.read_csv(path)
    missing = {"intent", "state", "level"} - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in {path}: {sorted(missing)}")

    df["score"] = divergence_scores(df["intent"], df["state"])
    by_level, rho, monotonic, alerts = analyze(df)

    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "scores.csv", index=False)
    make_plot(df, by_level, out_dir / "drift_by_level.png")

    print(f"\nRows: {len(df)}   Model: {MODEL_NAME}   Data: {path}")
    print("\n== Score per drift level ==")
    print(by_level.to_string())
    print(f"\nSpearman correlation (level vs score): {rho:.3f}")
    print(f"Mean score strictly increases with level: {monotonic}")
    print("\n== Share of rows that raise an alert, by threshold ==")
    print("(level 0-1 should stay low = few false alarms; 4-5 should be high)")
    print(alerts.to_string())
    if "kind" in df.columns:
        print(f"\n== By kind (alert threshold {ALERT_THRESHOLD}) ==")
        print(kind_table(df).to_string())
    print(f"\nSaved: {out_dir / 'scores.csv'} and {out_dir / 'drift_by_level.png'}")


if __name__ == "__main__":
    main()

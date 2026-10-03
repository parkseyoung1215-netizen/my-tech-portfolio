"""evaluate_nli.py

Second scorer for the drift test set: does the new state still *entail* the
original instruction? A natural-language-inference (NLI) model reads both
sentences and gives probabilities for entailment / neutral / contradiction.

    nli score = 1 - P(entailment)     (premise = state, hypothesis = intent)

The script compares this score with the embedding score already saved in
results/scores.csv (run evaluate_drift.py first). Headline metric: AUC for
telling each drift level apart from "preserved" rows (levels 0-1). 0.5 means
the scorer cannot tell them apart, 1.0 means it separates them perfectly.

Usage:  python3 evaluate_nli.py
Needs:  testset.csv and results/scores.csv in the current folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
DATA_PATH = Path("testset.csv")
EMB_PATH = Path("results/scores.csv")
OUT_DIR = Path("results")
PREFIX = "The assistant was asked to: "  # makes imperatives read like statements
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]


def to_probs(out):
    """Turn model output into probabilities (softmax unless already probabilities)."""
    out = np.asarray(out, dtype=float)
    if out.min() >= 0 and np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
        return out
    z = out - out.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_nli(premises, hypotheses):
    """Return (probs, label_names) for each premise/hypothesis pair."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(NLI_MODEL)
    labels = [DEFAULT_LABELS[i] for i in range(3)]
    try:
        id2label = {int(k): v.lower() for k, v in model.model.config.id2label.items()}
        if set(id2label.values()) == set(DEFAULT_LABELS):
            labels = [id2label[i] for i in range(3)]
    except Exception:
        pass
    pairs = list(zip(premises, hypotheses))
    return to_probs(model.predict(pairs)), labels


def auc(pos, neg):
    """P(random drifted row scores higher than random preserved row)."""
    pos = np.asarray(pos, dtype=float)[:, None]
    neg = np.asarray(neg, dtype=float)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def summarize(df, cols):
    means = df.groupby("level")[list(cols)].mean().round(3)
    rho = {c: df["level"].rank().corr(df[c].rank()) for c in cols}
    keep = df[df["level"] <= 1]
    rows = {}
    for lvl in range(2, 6):
        sub = df[df["level"] == lvl]
        rows[lvl] = {c: auc(sub[c], keep[c]) for c in cols}
    aucs = pd.DataFrame(rows).T.round(2)
    aucs.index.name = "level vs preserved (0-1)"
    return means, rho, aucs


def alert_table(df, col):
    return pd.DataFrame(
        {f">{t}": (df[col] > t).groupby(df["level"]).mean() for t in THRESHOLDS}
    ).round(2)


def make_plot(df, cols, titles, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.5), sharey=True)
    for ax, col, title in zip(np.atleast_1d(axes), cols, titles):
        ax.scatter(df["level"] + rng.uniform(-0.12, 0.12, len(df)), df[col], alpha=0.5, s=22)
        m = df.groupby("level")[col].mean()
        ax.plot(m.index, m.values, marker="o", color="black")
        ax.set_title(title)
        ax.set_xlabel("drift level (0 = identical, 5 = unrelated)")
        ax.set_ylim(-0.02, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("divergence score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    df = pd.read_csv(DATA_PATH)
    if not EMB_PATH.exists():
        raise SystemExit("results/scores.csv not found. Run evaluate_drift.py first.")
    emb = pd.read_csv(EMB_PATH)
    if len(emb) != len(df) or not (emb["state"].values == df["state"].values).all():
        raise SystemExit("results/scores.csv does not match testset.csv. Re-run evaluate_drift.py.")
    df["score_embedding"] = emb["score"].values

    probs, labels = run_nli(PREFIX + df["state"], PREFIX + df["intent"])
    p = pd.DataFrame(probs, columns=labels)
    df["p_entail"] = p["entailment"].values
    df["p_contradiction"] = p["contradiction"].values
    df["score_nli"] = 1.0 - df["p_entail"]

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "scores_nli.csv", index=False)
    cols = ["score_embedding", "score_nli"]
    make_plot(df, cols, ["Embedding (1 - cosine)", "NLI (1 - P(entailment))"],
              OUT_DIR / "compare_embedding_vs_nli.png")

    means, rho, aucs = summarize(df, cols)
    print(f"\nRows: {len(df)}   NLI model: {NLI_MODEL}")
    print("\n== Mean score per drift level ==")
    print(means.to_string())
    print("\nSpearman correlation (level vs score):")
    for c in cols:
        print(f"  {c}: {rho[c]:.3f}")
    print("\n== AUC: can the score tell each level apart from preserved rows (levels 0-1)? ==")
    print("(0.5 = no better than chance, 1.0 = perfect)")
    print(aucs.to_string())
    print("\n== Share of rows raising an alert, NLI score, by threshold ==")
    print(alert_table(df, "score_nli").to_string())
    print("\n== Level 3 (condition flipped), one row each ==")
    l3 = df[df["level"] == 3][["score_embedding", "score_nli", "p_contradiction", "state"]]
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(l3.round(3).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'scores_nli.csv'} and {OUT_DIR / 'compare_embedding_vs_nli.png'}")


if __name__ == "__main__":
    main()

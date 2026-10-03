"""evaluate_nli.py

Second scorer for the drift test set: does the new state still *entail* the
original instruction? A natural-language-inference (NLI) model reads both
sentences and gives probabilities for entailment / neutral / contradiction.

    nli score      = 1 - P(entailment)     (premise = state, hypothesis = intent)
    combined score = mean of embedding score and NLI score

The script compares the scorers using the embedding score already saved in
<out_dir>/scores.csv (run evaluate_drift.py first, with the same arguments).
Headline metric: AUC for telling each drift level apart from "preserved" rows
(levels 0-1). 0.5 means the scorer cannot tell them apart, 1.0 means it
separates them perfectly.

Usage:
  python3 evaluate_nli.py                                  # testset.csv, results/
  python3 evaluate_nli.py testset_v2.csv results_v2        # bigger set, results_v2/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
PREFIX = "The assistant was asked to: "  # makes imperatives read like statements
EMB_THR = 0.3  # alert threshold for the embedding score
NLI_THR = 0.5  # alert threshold for the NLI score
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]
COLS = ["score_embedding", "score_nli", "score_combined"]
TITLES = ["Embedding (1 - cosine)", "NLI (1 - P(entailment))", "Mean of both"]


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


def kind_table(df):
    """Per (level, kind): mean scores and how often each scorer raises an alert."""
    d = df.assign(
        flag_emb=df["score_embedding"] > EMB_THR,
        flag_nli=df["score_nli"] > NLI_THR,
    )
    d["flag_either"] = d["flag_emb"] | d["flag_nli"]
    g = d.groupby(["level", "kind"])
    t = g.agg(
        n=("level", "size"),
        emb_mean=("score_embedding", "mean"),
        nli_mean=("score_nli", "mean"),
        flagged_emb=("flag_emb", "mean"),
        flagged_nli=("flag_nli", "mean"),
        flagged_either=("flag_either", "mean"),
    )
    return t.round(2)


def make_plot(df, cols, titles, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 4.5), sharey=True)
    for ax, col, title in zip(np.atleast_1d(axes), cols, titles):
        ax.scatter(df["level"] + rng.uniform(-0.12, 0.12, len(df)), df[col], alpha=0.4, s=18)
        m = df.groupby("level")[col].mean()
        ax.plot(m.index, m.values, marker="o", color="black")
        ax.set_title(title)
        ax.set_xlabel("drift level (0 = identical, 5 = unrelated)")
        ax.set_ylim(-0.02, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("divergence score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("testset.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    emb_path = out_dir / "scores.csv"

    df = pd.read_csv(data_path)
    if not emb_path.exists():
        raise SystemExit(f"{emb_path} not found. Run evaluate_drift.py first (same arguments).")
    emb = pd.read_csv(emb_path)
    if len(emb) != len(df) or not (emb["state"].values == df["state"].values).all():
        raise SystemExit(f"{emb_path} does not match {data_path}. Re-run evaluate_drift.py.")
    df["score_embedding"] = emb["score"].values

    probs, labels = run_nli(PREFIX + df["state"], PREFIX + df["intent"])
    p = pd.DataFrame(probs, columns=labels)
    df["p_entail"] = p["entailment"].values
    df["p_contradiction"] = p["contradiction"].values
    df["score_nli"] = 1.0 - df["p_entail"]
    df["score_combined"] = (df["score_embedding"] + df["score_nli"]) / 2

    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "scores_nli.csv", index=False)
    make_plot(df, COLS, TITLES, out_dir / "compare_embedding_vs_nli.png")

    means, rho, aucs = summarize(df, COLS)
    print(f"\nRows: {len(df)}   NLI model: {NLI_MODEL}   Data: {data_path}")
    print("\n== Mean score per drift level ==")
    print(means.to_string())
    print("\nSpearman correlation (level vs score):")
    for c in COLS:
        print(f"  {c}: {rho[c]:.3f}")
    print("\n== AUC: can the score tell each level apart from preserved rows (levels 0-1)? ==")
    print("(0.5 = no better than chance, 1.0 = perfect)")
    print(aucs.to_string())
    print("\n== Share of rows raising an alert, NLI score, by threshold ==")
    print(alert_table(df, "score_nli").to_string())

    if "kind" in df.columns:
        kt = kind_table(df)
        kt.to_csv(out_dir / "by_kind.csv")
        print(f"\n== By kind: share flagged (embedding > {EMB_THR}, NLI > {NLI_THR}) ==")
        print("(paraphrase rows should stay low = false alarms; the rest should be high)")
        print(kt.to_string())
    else:
        print("\n== Level 3 (condition flipped), one row each ==")
        l3 = df[df["level"] == 3][["score_embedding", "score_nli", "p_contradiction", "state"]]
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(l3.round(3).to_string(index=False))

    print(f"\nSaved: {out_dir / 'scores_nli.csv'} and {out_dir / 'compare_embedding_vs_nli.png'}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import evaluate_nli as E

MODELS = {
    "MiniLM": ("cross-encoder/nli-MiniLM2-L6-H768", 0.75),
    "DeBERTa": ("cross-encoder/nli-deberta-v3-base", 0.50),
}
rng = np.random.default_rng(0)

# easy batch (labels assigned by reading outputs; "?" rows excluded)
b1 = pd.read_csv("results_v2/real_outputs.csv")
lab = pd.read_csv("real_labeled.csv")[["id", "label"]]
b1 = b1.merge(lab, on="id")
b1["label"] = b1["label"].astype(str)
b1 = b1[b1["label"].isin(["0", "1"])].copy()
b1["label"] = b1["label"].astype(int)
b1["batch"] = "easy"
b1["failed_rules"] = ""

# hard batch (labels computed by code)
b2 = pd.read_csv("results_v2/real_hard.csv")
b2["batch"] = "hard"

cols = ["id", "intent", "output", "summary", "label", "batch", "failed_rules"]
df = pd.concat([b1[cols], b2[cols]], ignore_index=True)
df["summary"] = df["summary"].fillna("")
df["failed_rules"] = df["failed_rules"].fillna("")


def auc(pos, neg):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def boot_rate(flags, n_boot=2000):
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return float("nan"), float("nan"), float("nan")
    b = [rng.choice(flags, len(flags)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * flags.mean(), 100 * lo, 100 * hi


def boot_auc(pos, neg, n_boot=2000):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc(pos, neg), lo, hi


def score_a(model_name):
    E.NLI_MODEL = model_name
    prem = [E.PREFIX + s for s in df["summary"]]
    hyp = [E.PREFIX + s for s in df["intent"]]
    probs, labels = E.run_nli(prem, hyp)
    probs = np.asarray(probs)
    labels = [str(l).lower() for l in labels]
    return 1 - probs[:, labels.index("entailment")]


rows = []
for name, (model_name, thr) in MODELS.items():
    df["a_" + name] = score_a(model_name)
    df["flag_" + name] = df["a_" + name] >= thr
    print(f"\n=== {name} (frozen threshold {thr}) ===")
    for subset, sub in [("easy", df[df["batch"] == "easy"]),
                        ("hard", df[df["batch"] == "hard"]),
                        ("pooled", df)]:
        ok = sub[sub["label"] == 1]
        bad = sub[sub["label"] == 0]
        fa = boot_rate(ok["flag_" + name])
        det = boot_rate(bad["flag_" + name])
        if len(bad) >= 2 and len(ok) >= 2:
            a = boot_auc(bad["a_" + name], ok["a_" + name])
        else:
            a = (float("nan"), float("nan"), float("nan"))
        print(f"[{subset}] followed n={len(ok)}, broke n={len(bad)}")
        print(f"  false alarm: {fa[0]:.1f}% [{fa[1]:.1f}, {fa[2]:.1f}]")
        print(f"  detection:   {det[0]:.1f}% [{det[1]:.1f}, {det[2]:.1f}]")
        print(f"  AUC:         {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]")
        rows.append({"model": name, "subset": subset, "n_followed": len(ok), "n_broke": len(bad),
                     "fa": fa[0], "fa_lo": fa[1], "fa_hi": fa[2],
                     "det": det[0], "det_lo": det[1], "det_hi": det[2],
                     "auc": a[0], "auc_lo": a[1], "auc_hi": a[2]})

    bad = df[df["label"] == 0]
    print("Missed violations (not flagged):")
    for _, r in bad[~bad["flag_" + name]].iterrows():
        print(f"  {r['batch']} id {r['id']} score {r['a_' + name]:.2f} [{r['failed_rules']}] | {r['intent'][:70]}")
    ok = df[df["label"] == 1]
    print("False alarms (flagged although rules followed):")
    for _, r in ok[ok["flag_" + name]].iterrows():
        print(f"  {r['batch']} id {r['id']} score {r['a_' + name]:.2f} | {r['intent'][:70]}")

df.to_csv("results_v2/real_scores.csv", index=False)
pd.DataFrame(rows).to_csv("results_v2/experiment10_real.csv", index=False)
print("\nsaved results_v2/real_scores.csv and results_v2/experiment10_real.csv")

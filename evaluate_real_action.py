import time
import numpy as np
import pandas as pd
from openai import OpenAI
import evaluate_nli as E

client = OpenAI()
GPT = "gpt-4o-mini"
MODELS = {
    "MiniLM": ("cross-encoder/nli-MiniLM2-L6-H768", 0.75),
    "DeBERTa": ("cross-encoder/nli-deberta-v3-base", 0.50),
}
rng = np.random.default_rng(0)

ACTION = ("Describe in one sentence, starting with 'The assistant', exactly what the assistant "
          "did in the following text, including its length and format.\n\nTEXT:\n")
OUTCOME = ("Rewrite the following instruction as one sentence describing what a fully correct "
           "response would have done, starting with 'The assistant'. Keep every requirement "
           "from the instruction.\n\nINSTRUCTION:\n")


def ask(prompt):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=GPT, messages=[{"role": "user", "content": prompt}], temperature=0.0)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print("retry after error:", e)
            time.sleep(3)
    return ""


b1 = pd.read_csv("results_v2/real_outputs.csv")
lab = pd.read_csv("real_labeled.csv")[["id", "label"]]
b1 = b1.merge(lab, on="id")
b1["label"] = b1["label"].astype(str)
b1 = b1[b1["label"].isin(["0", "1"])].copy()
b1["label"] = b1["label"].astype(int)
b1["batch"] = "easy"
b2 = pd.read_csv("results_v2/real_hard.csv")
b2["batch"] = "hard"
cols = ["id", "intent", "output", "label", "batch"]
df = pd.concat([b1[cols], b2[cols]], ignore_index=True)
df["output"] = df["output"].fillna("")

acts, exps = [], []
for i, r in df.iterrows():
    acts.append(ask(ACTION + r["output"]))
    exps.append(ask(OUTCOME + r["intent"]))
    print(f"done {i + 1}/{len(df)}")
df["action"] = acts
df["expected"] = exps
df.to_csv("results_v2/real_variant_texts.csv", index=False)


def auc(pos, neg):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def boot_rate(flags, n_boot=2000):
    flags = np.asarray(flags, dtype=float)
    b = [rng.choice(flags, len(flags)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * flags.mean(), 100 * lo, 100 * hi


def boot_auc(pos, neg, n_boot=2000):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc(pos, neg), lo, hi


rows = []
for name, (model_name, thr) in MODELS.items():
    E.NLI_MODEL = model_name
    probs, labels = E.run_nli(list(df["action"]), list(df["expected"]))
    probs = np.asarray(probs)
    labels = [str(l).lower() for l in labels]
    df["a_" + name] = 1 - probs[:, labels.index("entailment")]
    df["flag_" + name] = df["a_" + name] >= thr
    print(f"\n=== {name} (frozen threshold {thr}) ===")
    for subset, sub in [("easy", df[df["batch"] == "easy"]),
                        ("hard", df[df["batch"] == "hard"]),
                        ("pooled", df)]:
        ok = sub[sub["label"] == 1]
        bad = sub[sub["label"] == 0]
        fa = boot_rate(ok["flag_" + name])
        det = boot_rate(bad["flag_" + name])
        a = boot_auc(bad["a_" + name], ok["a_" + name])
        print(f"{subset:6s} n={len(ok)}/{len(bad)} | FA {fa[0]:.1f}% [{fa[1]:.1f},{fa[2]:.1f}]"
              f" | Det {det[0]:.1f}% [{det[1]:.1f},{det[2]:.1f}]"
              f" | AUC {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]")
        rows.append({"model": name, "subset": subset, "n_followed": len(ok), "n_broke": len(bad),
                     "fa": fa[0], "fa_lo": fa[1], "fa_hi": fa[2],
                     "det": det[0], "det_lo": det[1], "det_hi": det[2],
                     "auc": a[0], "auc_lo": a[1], "auc_hi": a[2]})

df.to_csv("results_v2/real_variant_scores.csv", index=False)
pd.DataFrame(rows).to_csv("results_v2/experiment11_real_action.csv", index=False)
print("\nsaved results_v2/real_variant_scores.csv and results_v2/experiment11_real_action.csv")

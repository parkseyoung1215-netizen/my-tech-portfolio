import numpy as np
import pandas as pd
import evaluate_nli as E

MODELS = {
    "MiniLM": ("cross-encoder/nli-MiniLM2-L6-H768", 0.75),
    "DeBERTa": ("cross-encoder/nli-deberta-v3-base", 0.50),
}
rng = np.random.default_rng(0)

old = pd.read_csv("testset_v2.csv")
old = old[old["kind"] == "paraphrase"].copy()
old["set"] = "old"
fresh = pd.read_csv("testset_fresh.csv")
fresh["set"] = "fresh"
df = pd.concat([old, fresh], ignore_index=True)

def ci_rate(flags, n_boot=2000):
    flags = np.asarray(flags, dtype=float)
    boots = [rng.choice(flags, len(flags)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * flags.mean(), 100 * lo, 100 * hi

def score_a(model_name):
    E.NLI_MODEL = model_name
    prem = [E.PREFIX + s for s in df["state"]]
    hyp = [E.PREFIX + s for s in df["intent"]]
    probs, labels = E.run_nli(prem, hyp)
    probs = np.asarray(probs)
    labels = [str(l).lower() for l in labels]
    return 1 - probs[:, labels.index("entailment")]

rows = []
for name, (model_name, thr) in MODELS.items():
    df["a_" + name] = score_a(model_name)
    flag = df["a_" + name] >= thr
    df["flag_" + name] = flag

    para_fresh = df[(df["set"] == "fresh") & (df["kind"] == "paraphrase")]
    para_all = df[df["kind"] == "paraphrase"]
    viol = df[(df["set"] == "fresh") & (df["kind"] != "paraphrase")]

    fa_f = ci_rate(para_fresh["flag_" + name])
    fa_p = ci_rate(para_all["flag_" + name])
    det = ci_rate(viol["flag_" + name])

    print(f"\n=== {name} (frozen threshold {thr}) ===")
    print(f"False alarm, fresh paraphrases (n=60): {fa_f[0]:.1f}% [{fa_f[1]:.1f}, {fa_f[2]:.1f}]")
    print(f"False alarm, old+fresh pooled (n=120): {fa_p[0]:.1f}% [{fa_p[1]:.1f}, {fa_p[2]:.1f}]")
    print(f"Detection, fresh violations (n=40):    {det[0]:.1f}% [{det[1]:.1f}, {det[2]:.1f}]")
    rows.append({"model": name, "threshold": thr,
                 "fa_fresh": fa_f[0], "fa_fresh_lo": fa_f[1], "fa_fresh_hi": fa_f[2],
                 "fa_pooled": fa_p[0], "fa_pooled_lo": fa_p[1], "fa_pooled_hi": fa_p[2],
                 "det": det[0], "det_lo": det[1], "det_hi": det[2]})

    print("Per kind (flagged / total):")
    for kind in ["negation", "number", "entity", "scope"]:
        sub = viol[viol["kind"] == kind]
        print(f"  {kind}: {int(sub['flag_' + name].sum())}/{len(sub)}")
    sc = viol[viol["kind"] == "scope"]
    print(f"  scope broadening (ids 91-95): {int(sc[sc['id'] <= 95]['flag_' + name].sum())}/5")
    print(f"  scope narrowing  (ids 96-100): {int(sc[sc['id'] > 95]['flag_' + name].sum())}/5")

    print("False alarms (fresh paraphrases):")
    for _, r in para_fresh[para_fresh["flag_" + name]].iterrows():
        print(f"  id {r['id']} score {r['a_' + name]:.2f} | {r['intent']} || {r['state']}")
    print("Misses (fresh violations):")
    for _, r in viol[~viol["flag_" + name]].iterrows():
        print(f"  id {r['id']} {r['kind']} score {r['a_' + name]:.2f} | {r['intent']} || {r['state']}")

df.to_csv("results_v2/scores_fresh.csv", index=False)
pd.DataFrame(rows).to_csv("results_v2/experiment9_fresh.csv", index=False)
print("\nsaved results_v2/scores_fresh.csv and results_v2/experiment9_fresh.csv")

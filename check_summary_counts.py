import re
import numpy as np
import pandas as pd

d = pd.read_csv("results_v2/real_variant_texts.csv")
d["output"] = d["output"].fillna("")
d["action"] = d["action"].fillna("")

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def words(t):
    return re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*", str(t))


def sentences(t):
    parts = re.split(r"(?<=[.!?])\s+", str(t).strip())
    return [p for p in parts if p.strip()]


def to_int(tok):
    tok = tok.lower()
    return int(tok) if tok.isdigit() else NUM.get(tok)


def claimed(text, unit):
    pat = r"\b(\d+|" + "|".join(NUM) + r")[-\s]" + unit + r"s?\b"
    m = re.search(pat, text, re.I)
    return to_int(m.group(1)) if m else None


def report(unit, counter):
    d["actual"] = d["output"].apply(lambda t: len(counter(t)))
    d["claim"] = d["action"].apply(lambda t: claimed(t, unit))
    have = d[d["claim"].notna()].copy()
    ok = have[have["claim"] == have["actual"]]
    err = (have["claim"] - have["actual"]).abs()
    print(f"\n[{unit}] summaries that state a {unit} count: {len(have)}/{len(d)}")
    if len(have):
        print(f"  claimed count equals actual: {len(ok)}/{len(have)} ({100 * len(ok) / len(have):.0f}%)")
        print(f"  mean absolute error: {err.mean():.1f}, max error: {err.max():.0f}")
        off = have[have["claim"] != have["actual"]].head(5)
        for _, r in off.iterrows():
            print(f"  e.g. id {r['id']} ({r['batch']}): claimed {int(r['claim'])}, actual {int(r['actual'])}")


report("word", words)
report("sentence", sentences)
print(f"\nrule-breaking outputs: {int((d['label'] == 0).sum())}, rule-following: {int((d['label'] == 1).sum())}")

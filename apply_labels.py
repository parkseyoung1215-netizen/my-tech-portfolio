import pandas as pd

d = pd.read_csv("real_to_label.csv")
labels = [1] * 50
labels[2] = 0     # id 3: 20 words, "under 20"
labels[11] = 0    # id 12: 39 words, "exactly 40"
labels[14] = "?"  # id 15: ambiguous
labels[47] = "?"  # id 48: ambiguous
d["label"] = labels
d.to_csv("real_labeled.csv", index=False)
print(d["label"].value_counts())

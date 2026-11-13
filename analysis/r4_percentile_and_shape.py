"""Pilot R4: two council challenges, tested on the experts' own annotations.

1. PERCENTILE BASELINE (First Principles): ISGEO (Foster 2002) sets vCDR cut-offs at a population
   percentile. A percentile cut-off needs no labels and cancels any monotone distortion of style.
   If it makes referral style-independent, the "calibrate on diagnosed cases" fix is redundant
   except for choosing capacity. What remains after it is the SCRAMBLE: real ranking differences.

2. IS STYLE A CONSTANT? (Contrarian): if each expert = consensus + constant, everything downstream
   is arithmetic. Test whether the offset depends on cup size, disc size or diagnosis, and whether
   any expert's ranking breaks down in a subgroup.

Usage: python -I r4_percentile_and_shape.py
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
a = pd.read_parquet("r1_annotations.parquet")
lab = pd.read_csv("r1_labels.csv", dtype={"folder": str})
a["vcdr"] = a["cup_v_t"] / a["disc_v_t"]
a["key"] = a["split"] + "/" + a["folder"]
lab["key"] = lab["split"] + "/" + lab["folder"]
w = a.pivot(index="key", columns="rater", values="vcdr")
disc = a.pivot(index="key", columns="rater", values="disc_v_t").mean(axis=1)
y = lab.set_index("key")["label"].reindex(w.index)
w, disc, y = w[y.notna()], disc[y.notna()], y[y.notna()].astype(int)
split = pd.Series(w.index.str.split("/").str[0], index=w.index)
raters = list(w.columns)
yv = y.to_numpy()

# ---------------------------------------------------------------- 1. percentile baseline
print("1. LABEL-FREE PERCENTILE CUT-OFF: refer the top q% of eyes by each expert's own vCDR")
print(f"{'rule':<26}{'sens range':>16}{'spec range':>16}{'pairwise disagreement':>24}")


def summarise(decisions, label):
    sens = [d[yv == 1].mean() for d in decisions.T]
    spec = [(~d[yv == 0]).mean() for d in decisions.T]
    flips = np.mean([(decisions[:, i] != decisions[:, j]).mean()
                     for i, j in itertools.combinations(range(decisions.shape[1]), 2)])
    print(f"{label:<26}{min(sens):>7.2f}-{max(sens):<8.2f}{min(spec):>7.2f}-{max(spec):<8.2f}{100 * flips:>20.1f}%")


fixed = (w[raters] >= 0.6).to_numpy()
summarise(fixed, "fixed vCDR >= 0.6")
for q in (0.10, 0.15, 0.20):
    pct = np.column_stack([(w[r] >= w[r].quantile(1 - q)).to_numpy() for r in raters])
    summarise(pct, f"top {int(q * 100)}% by percentile")
# percentile computed per camera (a deployment would only see its own camera's distribution)
for q in (0.15,):
    cols = []
    for r in raters:
        thr = w[r].groupby(split.str.startswith("Training")).transform(lambda s: s.quantile(1 - q))
        cols.append((w[r] >= thr).to_numpy())
    summarise(np.column_stack(cols), f"top {int(q * 100)}% per camera")
print("   With a percentile rule all experts refer the same NUMBER of eyes; remaining disagreement is")
print("   about WHICH eyes, i.e. genuine ranking differences (the scramble).")

# ---------------------------------------------------------------- 2. shape of style
print("\n2. IS EACH EXPERT'S STYLE A CONSTANT OFFSET?")
cons = w[raters].mean(axis=1)
terc = pd.qcut(cons, 3, labels=["small cups", "medium cups", "large cups"])
dterc = pd.qcut(disc, 3, labels=["small discs", "medium discs", "large discs"])
dev = w[raters].sub(cons, axis=0)
print("   Offset by consensus cup size (tertiles):")
print("   " + dev.groupby(terc).mean().T.round(3).to_string().replace("\n", "\n   "))
print("\n   Offset by disc size (tertiles):")
print("   " + dev.groupby(dterc).mean().T.round(3).to_string().replace("\n", "\n   "))
print("\n   Offset by clinical diagnosis:")
print("   " + dev.groupby(y.map({0: "normal", 1: "glaucoma"})).mean().T.round(3).to_string().replace("\n", "\n   "))

print("\n   Ranking quality (AUC for clinical glaucoma) by disc size tertile:")
rows = []
for g in dterc.cat.categories:
    m = (dterc == g).to_numpy()
    if yv[m].min() == yv[m].max():
        continue
    rows.append({"subgroup": g, "n glaucoma": int(yv[m].sum()),
                 **{r: roc_auc_score(yv[m], w[r].to_numpy()[m]) for r in raters},
                 "mean of 7": roc_auc_score(yv[m], cons.to_numpy()[m])})
print("   " + pd.DataFrame(rows).set_index("subgroup").round(3).to_string().replace("\n", "\n   "))

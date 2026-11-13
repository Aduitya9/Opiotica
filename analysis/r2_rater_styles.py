"""Pilot R2: do seven glaucoma experts differ systematically, and what does that do to referral?

Reads R1's parsed ellipses (convention T: angle 90 = upright, `height` is vertical, verified against
the rasterised masks) and the clinical glaucoma labels. No model, no simulation: every number is
computed from the experts' own annotations of the same 1,195 labelled images.

Usage: python -I r2_rater_styles.py
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
RNG = np.random.default_rng(0)
B = 2000

a = pd.read_parquet("r1_annotations.parquet")
lab = pd.read_csv("r1_labels.csv", dtype={"folder": str})
a["vcdr"] = a["cup_v_t"] / a["disc_v_t"]
a["key"] = a["split"] + "/" + a["folder"]
lab["key"] = lab["split"] + "/" + lab["folder"]
wide = a.pivot(index="key", columns="rater", values="vcdr")
disc = a.pivot(index="key", columns="rater", values="disc_v_t")
cup = a.pivot(index="key", columns="rater", values="cup_v_t")
y = lab.set_index("key")["label"].reindex(wide.index)
keep = y.notna()
wide, disc, cup, y = wide[keep], disc[keep], cup[keep], y[keep].astype(int)
split = wide.index.str.split("/").str[0]
raters = list(wide.columns)
n = len(wide)
print(f"{n} labelled images x {len(raters)} experts; clinical glaucoma {y.sum()} ({100 * y.mean():.1f}%)\n")

# ---------------------------------------------------------------- 1. systematic style
print("1. STYLE: each expert's mean vCDR minus the image's 7-expert mean (paired, same images)")
dev = wide.sub(wide.mean(axis=1), axis=0)
boot = np.array([dev.iloc[RNG.integers(0, n, n)].mean().to_numpy() for _ in range(B)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
cupdev = (cup.div(disc.mean(axis=1), axis=0)).sub(cup.div(disc.mean(axis=1), axis=0).mean(axis=1), axis=0)
discdev = disc.div(disc.mean(axis=1), axis=0) - 1
print(f"{'expert':>8}{'mean vCDR':>11}{'offset':>9}{'95% CI':>18}{'cup size vs mean':>18}{'disc size vs mean':>19}")
for j, r in enumerate(raters):
    print(f"{r:>8}{wide[r].mean():>11.3f}{dev[r].mean():>+9.3f}   [{lo[j]:+.3f}, {hi[j]:+.3f}]"
          f"{100 * cupdev[r].mean() / cup.div(disc.mean(axis=1), axis=0).mean().mean():>+16.1f}%"
          f"{100 * discdev[r].mean():>+17.1f}%")
fr = stats.friedmanchisquare(*[wide[r] for r in raters])
print(f"Friedman test across the 7 experts (paired): chi2 {fr.statistic:.0f}, p {fr.pvalue:.1e}")

# stability of each expert's offset across the three independent splits
print("\n   Is each expert's offset stable across the three image sets (Zeiss train; Canon val, test)?")
tab = dev.groupby(split).mean().T
print("   " + tab.round(3).to_string().replace("\n", "\n   "))
print(f"   correlation of expert offsets between sets: train-val {np.corrcoef(tab.iloc[:, 1], tab.iloc[:, 2])[0, 1]:.2f}, "
      f"train-test {np.corrcoef(tab.iloc[:, 1], tab.iloc[:, 0])[0, 1]:.2f}, val-test {np.corrcoef(tab.iloc[:, 2], tab.iloc[:, 0])[0, 1]:.2f}")


def icc(mat, kind):
    x = mat.to_numpy()
    nn, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (nn - 1)
    msc = nn * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((nn - 1) * (k - 1))
    if kind == "agreement":
        return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / nn)
    return (msr - mse) / (msr + (k - 1) * mse)


sd_within = wide.std(axis=1, ddof=1)
print(f"\n   within-image SD across experts: median {sd_within.median():.3f}, 90th pct {sd_within.quantile(0.9):.3f}")
print(f"   ICC(2,1) absolute agreement {icc(wide, 'agreement'):.3f}; ICC(3,1) consistency {icc(wide, 'consistency'):.3f}"
      "  (the gap is the systematic part)")

# ---------------------------------------------------------------- 2. referral
print("\n2. REFERRAL: refer if vCDR >= threshold")
for t in (0.5, 0.6, 0.7):
    ref = wide >= t
    rates = ref.mean()
    unanimous = (ref.all(axis=1) | (~ref).all(axis=1)).mean()
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    pair_flip = np.mean([(ref.iloc[:, i] != ref.iloc[:, j]).mean() for i, j in pairs])
    print(f"   threshold {t:.1f}: referral rate by expert {', '.join(f'{100 * v:.1f}%' for v in rates)}")
    print(f"      range {100 * rates.min():.1f}% to {100 * rates.max():.1f}% ({rates.max() / max(rates.min(), 1e-9):.1f}x); "
          f"experts not unanimous on {100 * (1 - unanimous):.1f}% of eyes; two random experts disagree on {100 * pair_flip:.1f}%")

# ---------------------------------------------------------------- 3. against the clinical diagnosis
print("\n3. CLINICAL TRUTH: how well does each expert's vCDR separate diagnosed glaucoma (independent label)?")
print(f"{'expert':>10}{'AUC':>7}{'95% CI':>17}" + "".join(f"{f'sens@{t}':>10}{f'spec@{t}':>10}" for t in (0.6, 0.7))
      + f"{'NB@pt.10':>10}")
yv = y.to_numpy()
idx_boot = [RNG.integers(0, n, n) for _ in range(B)]
cols = raters + ["mean of 7"]
wide["mean of 7"] = wide[raters].mean(axis=1)
for r in cols:
    s = wide[r].to_numpy()
    auc = roc_auc_score(yv, s)
    bs = [roc_auc_score(yv[i], s[i]) for i in idx_boot if yv[i].min() != yv[i].max()]
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    row = f"{r:>10}{auc:>7.3f}   [{blo:.3f}, {bhi:.3f}]"
    for t in (0.6, 0.7):
        pred = s >= t
        row += f"{pred[yv == 1].mean():>10.2f}{(~pred[yv == 0]).mean():>10.2f}"
    pt = 0.10
    pred = s >= 0.6
    nb = (pred & (yv == 1)).mean() - (pred & (yv == 0)).mean() * pt / (1 - pt)
    row += f"{nb:>10.4f}"
    print(row)
treat_all = yv.mean() - (1 - yv.mean()) * 0.10 / 0.90
print(f"   net benefit of 'refer everyone' at pt 0.10: {treat_all:.4f}; of 'refer no one': 0")

# paired AUC difference: best vs worst expert
aucs = {r: roc_auc_score(yv, wide[r]) for r in raters}
best, worst = max(aucs, key=aucs.get), min(aucs, key=aucs.get)
d = [roc_auc_score(yv[i], wide[best].to_numpy()[i]) - roc_auc_score(yv[i], wide[worst].to_numpy()[i])
     for i in idx_boot if yv[i].min() != yv[i].max()]
print(f"   best expert {best} vs worst {worst}: AUC difference {aucs[best] - aucs[worst]:+.3f}, "
      f"95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

# ---------------------------------------------------------------- 4. the operating point that does not travel
print("\n4. ONE THRESHOLD, SEVEN STYLES: at the threshold that gives 85% sensitivity with the 7-expert mean,")
s_mean = wide["mean of 7"].to_numpy()
thr = np.quantile(s_mean[yv == 1], 0.15)
print(f"   threshold vCDR {thr:.3f}; sensitivity and specificity if a model reproduced each expert instead:")
for r in raters:
    pred = wide[r].to_numpy() >= thr
    print(f"   {r:>8}: sensitivity {pred[yv == 1].mean():.2f}, specificity {(~pred[yv == 0]).mean():.2f}, "
          f"referral rate {100 * pred.mean():.1f}%")

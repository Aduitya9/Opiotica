"""Pilot R3: does the proposed fix work in principle, on the experts' own measurements?

R2 showed the seven experts rank eyes alike but put the line in different places. If that is the
whole story, two cheap corrections should work:

  A. LEVEL CORRECTION. Learn each expert's linear relation to the leave-one-out consensus on one image
     set (Zeiss training images) and remove it on the others (Canon). Referral disagreement between
     experts should collapse if styles are mostly offset/scale.
  B. LOCAL CALIBRATION TO CLINICAL TRUTH. Whatever the style, choose the referral cut-off from a small
     sample of locally diagnosed patients to hit a target sensitivity. How many diagnosed glaucoma
     cases does a clinic need before the achieved sensitivity is reliable?

Both are previews of RQ4 using human measurements in place of model outputs.

Usage: python -I r3_fix_preview.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RNG = np.random.default_rng(3)
a = pd.read_parquet("r1_annotations.parquet")
lab = pd.read_csv("r1_labels.csv", dtype={"folder": str})
a["vcdr"] = a["cup_v_t"] / a["disc_v_t"]
a["key"] = a["split"] + "/" + a["folder"]
lab["key"] = lab["split"] + "/" + lab["folder"]
w = a.pivot(index="key", columns="rater", values="vcdr")
y = lab.set_index("key")["label"].reindex(w.index)
w, y = w[y.notna()], y[y.notna()].astype(int)
split = pd.Series(w.index.str.split("/").str[0], index=w.index)
raters = list(w.columns)


def disagreement(mat, t):
    ref = mat >= t
    rates = ref.mean()
    pairs = [(i, j) for i in range(mat.shape[1]) for j in range(i + 1, mat.shape[1])]
    flip = np.mean([(ref.iloc[:, i] != ref.iloc[:, j]).mean() for i, j in pairs])
    return rates.min(), rates.max(), flip


# ---------------------------------------------------------------- A. level correction
train = split == "Training-400"
test = ~train
corrected = pd.DataFrame(index=w.index[test])
print("A. LEVEL CORRECTION (fit on Zeiss training eyes, applied to Canon eyes)")
print(f"{'expert':>8}{'intercept':>11}{'slope':>8}{'resid SD':>10}")
for r in raters:
    others = w.drop(columns=r).mean(axis=1)
    b, c = np.polyfit(others[train], w.loc[train, r], 1)          # expert = c + b * consensus
    corrected[r] = (w.loc[test, r] - c) / b                          # back onto the consensus scale
    resid = w.loc[train, r] - (c + b * others[train])
    print(f"{r:>8}{c:>+11.3f}{b:>8.3f}{resid.std():>10.3f}")
for t in (0.5, 0.6, 0.7):
    lo0, hi0, f0 = disagreement(w[test], t)
    lo1, hi1, f1 = disagreement(corrected, t)
    print(f"   threshold {t:.1f}: referral range {100 * lo0:5.1f}-{100 * hi0:5.1f}% -> {100 * lo1:5.1f}-{100 * hi1:5.1f}% ; "
          f"two random experts disagree {100 * f0:4.1f}% -> {100 * f1:4.1f}%")

# ---------------------------------------------------------------- B. local calibration to clinical truth
print("\nB. LOCAL CALIBRATION: pick the cut-off from a small diagnosed sample to reach 85% sensitivity,")
print("   then score it on all remaining eyes. 1,000 random draws per sample size.")
target = 0.85
pos_idx = np.where(y.to_numpy() == 1)[0]
neg_idx = np.where(y.to_numpy() == 0)[0]
print(f"{'style':>10}{'fixed 0.6: sens / spec':>26}" + "".join(f"{f'n_glaucoma={k}':>22}" for k in (5, 10, 20, 40)))
rows = {}
for r in raters + ["mean of 7"]:
    s = (w[raters].mean(axis=1) if r == "mean of 7" else w[r]).to_numpy()
    yy = y.to_numpy()
    fixed = f"{(s[yy == 1] >= 0.6).mean():.2f} / {(s[yy == 0] < 0.6).mean():.2f}"
    cells = ""
    for k in (5, 10, 20, 40):
        sens = []
        for _ in range(1000):
            cal_pos = RNG.choice(pos_idx, k, replace=False)
            cal_neg = RNG.choice(neg_idx, 9 * k, replace=False)         # 10% prevalence, as in screening referrals
            thr = np.quantile(s[cal_pos], 1 - target)                   # cut-off at 85% sensitivity in the sample
            rest = np.setdiff1d(np.arange(len(s)), np.r_[cal_pos, cal_neg])
            sens.append((s[rest][yy[rest] == 1] >= thr).mean())
        sens = np.array(sens)
        cells += f"   {sens.mean():.2f} [{np.percentile(sens, 5):.2f},{np.percentile(sens, 95):.2f}]"
        rows[(r, k)] = sens
    print(f"{r:>10}{fixed:>26}{cells}")
print("\n   Cells: achieved sensitivity on held-out eyes, mean [5th, 95th percentile] across draws.")
print("   Target 0.85. A style-independent fix should land near 0.85 for EVERY expert, given enough cases.")

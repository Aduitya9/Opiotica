"""Pilot R6: where does REFUGE's OFFICIAL merged mask sit on the seven-expert convention scale?

R5 reported `inf` for the official merge: the BMP reader treated palette indices as grey levels.
Correct encoding, verified: `_cup.bmp` holds cup = index 0, background = 255;
`_disc.bmp` holds disc = index 128, background = 255.

The official mask is what every REFUGE-trained model in the literature actually learned from, so
its position on the convention scale is the anchor for the audit tool.

Usage: python -I r6_official_merge.py <REFUGE-Multirater folder>
"""
import glob
import os
import struct
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]


def read_bmp8(path):
    b = open(path, "rb").read()
    off = struct.unpack("<I", b[10:14])[0]
    w, h = struct.unpack("<ii", b[18:26])
    bpp = struct.unpack("<H", b[28:30])[0]
    assert bpp == 8, bpp
    row = (w + 3) // 4 * 4
    a = np.frombuffer(b, dtype=np.uint8, count=row * abs(h), offset=off).reshape(abs(h), row)[:, :w]
    return a[::-1] if h > 0 else a


def vertical_extent(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    return int(rows.max() - rows.min() + 1) if len(rows) else 0


t0 = time.time()
rows = []
for split in ("Training-400", "Validation-400", "Test-400"):
    for d in sorted(glob.glob(os.path.join(ROOT, split, "*"))):
        folder = os.path.basename(d)
        if not folder.isdigit():
            continue
        cup = read_bmp8(os.path.join(d, f"{folder}_cup.bmp")) == 0        # cup  = index 0
        disc = read_bmp8(os.path.join(d, f"{folder}_disc.bmp")) == 128    # disc = index 128
        dv = vertical_extent(disc)
        rows.append({"split": split, "folder": folder,
                     "official_vcdr": vertical_extent(cup) / dv if dv else np.nan})
off = pd.DataFrame(rows)
print(f"{len(off)} images decoded in {time.time() - t0:.0f}s; "
      f"{off['official_vcdr'].isna().sum()} failed\n")

here = os.path.dirname(os.path.abspath(__file__))
a = pd.read_parquet(os.path.join(here, "r1_annotations.parquet"))
a["vcdr"] = a["cup_v_t"] / a["disc_v_t"]
w = a.pivot_table(index=["split", "folder"], columns="rater", values="vcdr").reset_index()
raters = [c for c in w.columns if str(c).startswith("user")]
m = w.merge(off, on=["split", "folder"], how="inner")
lab = pd.read_csv(os.path.join(here, "r1_labels.csv"), dtype={"folder": str})
m = m.merge(lab, on=["split", "folder"], how="left")
m["consensus"] = m[raters].mean(axis=1)

print("WHERE THE OFFICIAL REFUGE MERGE SITS (the label everyone actually trains on)")
print(f"{'source':<26}{'mean vCDR':>10}{'offset vs 7-expert mean':>26}{'refer@0.6':>11}{'refer@0.7':>11}")
for r in raters + ["consensus", "official_vcdr"]:
    s = m[r]
    name = "7-expert mean" if r == "consensus" else ("OFFICIAL MERGE (8th expert)" if r == "official_vcdr" else r)
    off_v = (s - m["consensus"]).mean()
    print(f"{name:<26}{s.mean():>10.3f}{off_v:>+26.3f}{100 * (s >= 0.6).mean():>10.1f}%{100 * (s >= 0.7).mean():>10.1f}%")

ok = m["label"].notna()
yy = m.loc[ok, "label"].astype(int).to_numpy()
print("\nAgainst the clinical diagnosis (sensitivity / specificity at vCDR >= 0.6):")
for r in raters + ["consensus", "official_vcdr"]:
    s = m.loc[ok, r].to_numpy()
    name = "7-expert mean" if r == "consensus" else ("OFFICIAL MERGE" if r == "official_vcdr" else r)
    print(f"   {name:<26}{(s[yy == 1] >= 0.6).mean():>6.2f} / {(s[yy == 0] < 0.6).mean():.2f}")

print(f"\nCorrelation of the official merge with each expert (how much it follows each one):")
for r in raters:
    print(f"   {r}: r = {m[r].corr(m['official_vcdr']):.3f},  mean difference {(m['official_vcdr'] - m[r]).mean():+.3f}")
m.to_csv(os.path.join(here, "r6_official_merge.csv"), index=False)

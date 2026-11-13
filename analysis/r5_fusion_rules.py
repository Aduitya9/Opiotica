"""Pilot R5: how far does the label-FUSION RULE alone move the cup, on the same images and outlines?

Public fundus datasets fuse their experts differently: Drishti-GS uses agreement of >= 3 of 4 (and
DoFE re-fused it as all 4), RIM-ONE-r3 averages 5, REFUGE had an 8th senior specialist merge 7,
Chakshu uses STAPLE of 5. Models trained on one convention and tested on another register a "domain
gap" even if the images were identical. Here every rule is applied to REFUGE's same seven experts'
outlines, so any difference is convention alone.

Ellipses are rasterised on a local grid in annotation coordinates (convention T: rotation = angle-90).
The official REFUGE merged masks (BMP, full resolution) are read directly for comparison.

Usage: python -I r5_fusion_rules.py <REFUGE-Multirater folder>
"""
import glob
import json
import os
import re
import struct
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]
RATERS = [f"user{k}" for k in range(1, 8)]
STEP = 0.5          # grid step in annotation pixels


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    strings = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf-8"))
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    cells = {}
    for ref, attrs, val in re.findall(r'<c r="([A-Z]+[0-9]+)"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>', sheet):
        cells[ref] = strings[int(val)] if 't="s"' in attrs else val
    return [(json.loads(cells[f"C{r}"]), json.loads(cells[f"D{r}"])) for r in range(3, 10) if cells.get(f"A{r}") in RATERS]


def inside(e, X, Y):
    t = np.deg2rad(e["angle"] - 90.0)
    dx, dy = X - e["centerX"], Y - e["centerY"]
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / (e["width"] / 2)) ** 2 + (v / (e["height"] / 2)) ** 2 <= 1.0


def vertical_extent(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    return (rows.max() - rows.min() + 1) if len(rows) else 0


def read_bmp(path):
    b = open(path, "rb").read()
    off = struct.unpack("<I", b[10:14])[0]
    w, h = struct.unpack("<ii", b[18:26])
    bpp = struct.unpack("<H", b[28:30])[0]
    assert bpp == 8
    row = (w + 3) // 4 * 4
    a = np.frombuffer(b, dtype=np.uint8, count=row * abs(h), offset=off).reshape(abs(h), row)[:, :w]
    return a[::-1] if h > 0 else a


RULES = {"unanimous (7/7)": 7, ">= 5 of 7": 5, "majority (>= 4 of 7)": 4, ">= 3 of 7 (Drishti-like)": 3,
         "union (>= 1 of 7)": 1}


def main():
    t0 = time.time()
    out = []
    for x in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*.xlsx"))):
        folder_dir = os.path.dirname(x)
        folder = os.path.basename(folder_dir)
        split = os.path.basename(os.path.dirname(folder_dir))
        ann = read_xlsx(x)
        discs = [d for d, _ in ann]
        cups = [c for _, c in ann]
        cx = np.mean([d["centerX"] for d in discs])
        cy = np.mean([d["centerY"] for d in discs])
        half = 0.75 * max(max(d["height"], d["width"]) for d in discs) + 5
        g = np.arange(-half, half, STEP)
        X, Y = np.meshgrid(cx + g, cy + g)
        D = np.stack([inside(d, X, Y) for d in discs])
        C = np.stack([inside(c, X, Y) for c in cups])
        rec = {"split": split, "folder": folder}
        nd, nc = D.sum(axis=0), C.sum(axis=0)
        for name, k in RULES.items():
            dv = vertical_extent(nd >= k)
            rec[name] = vertical_extent(nc >= k) / dv if dv else np.nan
        # soft average of 7, thresholded at 0.5 (RIM-ONE-like average)
        dv = vertical_extent(D.mean(axis=0) >= 0.5)
        rec["average >= 0.5 (RIM-ONE-like)"] = vertical_extent(C.mean(axis=0) >= 0.5) / dv if dv else np.nan
        # official REFUGE merge by the senior specialist
        try:
            cupb = read_bmp(os.path.join(folder_dir, f"{folder}_cup.bmp"))
            discb = read_bmp(os.path.join(folder_dir, f"{folder}_disc.bmp"))
            fg = lambda m: (m < 128) if (m < 128).mean() < 0.5 else (m >= 128)
            rec["official merge (8th expert)"] = vertical_extent(fg(cupb)) / vertical_extent(fg(discb))
        except Exception:  # noqa: BLE001
            rec["official merge (8th expert)"] = np.nan
        out.append(rec)
    df = pd.DataFrame(out)
    print(f"{len(df)} images in {time.time() - t0:.0f}s\n")
    lab = pd.read_csv("r1_labels.csv", dtype={"folder": str})
    df = df.merge(lab, on=["split", "folder"], how="left")
    rules = [c for c in df.columns if c not in ("split", "folder", "label")]
    print(f"{'fusion rule (same 7 experts, same images)':<36}{'mean vCDR':>10}{'refer@0.6':>11}{'refer@0.7':>11}"
          f"{'sens@0.6':>10}{'spec@0.6':>10}")
    lab_ok = df["label"].notna()
    yy = df.loc[lab_ok, "label"].astype(int).to_numpy()
    for r in rules:
        s = df[r]
        ss = df.loc[lab_ok, r].to_numpy()
        print(f"{r:<36}{s.mean():>10.3f}{100 * (s >= 0.6).mean():>10.1f}%{100 * (s >= 0.7).mean():>10.1f}%"
              f"{(ss[yy == 1] >= 0.6).mean():>10.2f}{(ss[yy == 0] < 0.6).mean():>10.2f}")
    base = df["official merge (8th expert)"]
    print("\nmean |vCDR difference| versus the official REFUGE merge, per rule:")
    for r in rules[:-1]:
        print(f"   {r:<36}{np.nanmean(np.abs(df[r] - base)):.3f}")
    df.to_csv("r5_fusion_vcdr.csv", index=False)


if __name__ == "__main__":
    main()

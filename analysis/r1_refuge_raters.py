"""Pilot R1: seven glaucoma experts, the same 1,200 fundus images, and the referral decision.

REFUGE-MultiRater keeps each expert's raw optic disc and cup annotation (an ellipse) per image.
The vertical cup-to-disc ratio (vCDR) is the measurement glaucoma screening refers on (typically
vCDR >= 0.6 or 0.7). The glaucoma label in REFUGE comes from clinical records, not from these
annotations, so it is an independent reference.

Questions answered on real annotations, no model and no simulation:
  1. Do experts differ SYSTEMATICALLY (same images, paired), not just randomly?
  2. How often does the referral decision change with the expert?
  3. What referral rate would a screening programme get by adopting each expert's style?
  4. Whose vCDR best separates clinically diagnosed glaucoma from normal?

Usage: python -I r1_refuge_raters.py <REFUGE-Multirater folder>
"""
import glob
import json
import os
import re
import struct
import sys
import zipfile
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]
SPLITS = {"Training-400": "REFUGE1Train.csv", "Validation-400": "REFUGE1Val.csv", "Test-400": "REFUGE1Test.csv"}
RATERS = [f"user{k}" for k in range(1, 8)]


# ------------------------------------------------------------------ raw annotations
def read_xlsx(path):
    z = zipfile.ZipFile(path)
    strings = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf-8"))
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    cells = {}
    for ref, attrs, val in re.findall(r'<c r="([A-Z]+[0-9]+)"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>', sheet):
        cells[ref] = strings[int(val)] if 't="s"' in attrs else val
    out = {}
    for row in range(3, 10):
        user = cells.get(f"A{row}")
        if user in RATERS:
            out[user] = (json.loads(cells[f"C{row}"]), json.loads(cells[f"D{row}"]))
    return out


def extents(e):
    """Vertical and horizontal full extents of a rotated ellipse, for two angle conventions."""
    a, b = e["width"] / 2, e["height"] / 2
    t = np.deg2rad(e["angle"])
    # convention R: the ellipse's width axis is rotated by `angle` from horizontal
    vert_r = 2 * np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    horz_r = 2 * np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    # convention T: angle 90 is the tool's upright default, so rotation is (angle - 90)
    t2 = np.deg2rad(e["angle"] - 90)
    vert_t = 2 * np.sqrt((a * np.sin(t2)) ** 2 + (b * np.cos(t2)) ** 2)
    horz_t = 2 * np.sqrt((a * np.cos(t2)) ** 2 + (b * np.sin(t2)) ** 2)
    return vert_r, horz_r, vert_t, horz_t, e["centerX"], e["centerY"]


# ------------------------------------------------------------------ minimal PNG decoder (8-bit grey), for the axis check only
def read_png_grey(path):
    data = open(path, "rb").read()
    pos, idat = 8, b""
    w = h = 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and ctype == 0, "only 8-bit greyscale handled"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    img = np.zeros((h, w), dtype=np.uint8)
    prev = np.zeros(w, dtype=np.int32)
    stride = w + 1
    for y in range(h):
        f = raw[y * stride]
        line = np.frombuffer(raw, dtype=np.uint8, count=w, offset=y * stride + 1).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:
            cur = np.cumsum(line) % 256
        elif f == 2:
            cur = (line + prev) % 256
        else:
            cur = np.zeros(w, dtype=np.int32)
            for x in range(w):
                left = cur[x - 1] if x else 0
                up = prev[x]
                if f == 3:
                    cur[x] = (line[x] + (left + up) // 2) % 256
                else:
                    ul = prev[x - 1] if x else 0
                    p = left + up - ul
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                    pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else ul)
                    cur[x] = (line[x] + pred) % 256
        img[y] = cur
        prev = cur
    return img


def mask_extent(path):
    m = read_png_grey(path)
    fg = m > 127 if (m > 127).mean() < 0.5 else m < 128
    ys, xs = np.nonzero(fg)
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1


# ------------------------------------------------------------------ labels
def load_labels():
    rows = []
    for split, csvname in SPLITS.items():
        df = pd.read_csv(os.path.join(ROOT, csvname))
        for _, r in df.iterrows():
            mm = str(r["multimaskName"]).strip()
            m = re.search(r"/(\d{4})/", mm)
            if m:
                rows.append({"split": split, "folder": m.group(1), "orig": str(r["imgName"]).split("/")[-1],
                             "label": int(r["label"])})
    lab = pd.DataFrame(rows)
    dup = lab.groupby(["split", "folder"]).size()
    conflict = lab.groupby(["split", "folder"])["label"].nunique()
    print(f"label rows {len(lab)}; folders referenced {lab.groupby('split')['folder'].nunique().to_dict()}; "
          f"folders referenced twice or more: {(dup > 1).sum()}; with conflicting labels: {(conflict > 1).sum()}")
    clean = lab.groupby(["split", "folder"]).filter(lambda g: g["label"].nunique() == 1)
    return clean.groupby(["split", "folder"])["label"].first().reset_index()


# ------------------------------------------------------------------ main
def main():
    recs = []
    for split in SPLITS:
        for x in sorted(glob.glob(os.path.join(ROOT, split, "*", "*.xlsx"))):
            folder = os.path.basename(os.path.dirname(x))
            ann = read_xlsx(x)
            for user, (disc, cup) in ann.items():
                dv_r, dh_r, dv_t, dh_t, dcx, dcy = extents(disc)
                cv_r, ch_r, cv_t, ch_t, ccx, ccy = extents(cup)
                recs.append(dict(split=split, folder=folder, rater=user, disc_v_r=dv_r, disc_h_r=dh_r, disc_v_t=dv_t,
                                 disc_h_t=dh_t, cup_v_r=cv_r, cup_v_t=cv_t, disc_angle=disc["angle"], cup_angle=cup["angle"],
                                 disc_w=disc["width"], disc_h=disc["height"]))
    a = pd.DataFrame(recs)
    print(f"{a['folder'].nunique()} images x {a['rater'].nunique()} raters = {len(a)} annotations "
          f"(complete images: {(a.groupby(['split', 'folder']).size() == 7).sum()})")
    print(f"angles: disc {a['disc_angle'].round(0).value_counts().head(4).to_dict()}, "
          f"cup {a['cup_angle'].round(0).value_counts().head(4).to_dict()}")

    # ---- which axis is vertical? compare with one rasterised mask per split
    print("\nAxis check against the rasterised masks (ratio of vertical to horizontal extent):")
    for split in SPLITS:
        folder = sorted(d for d in os.listdir(os.path.join(ROOT, split)) if d.isdigit())[0]
        row = a[(a["split"] == split) & (a["folder"] == folder) & (a["rater"] == "user1")].iloc[0]
        mv, mh = mask_extent(os.path.join(ROOT, split, folder, f"{folder}_seg_disc_1.png"))
        print(f"  {split} {folder} user1 disc: mask v/h {mv / mh:.3f} | convention R {row.disc_v_r / row.disc_h_r:.3f}"
              f" | convention T {row.disc_v_t / row.disc_h_t:.3f}")
    return a


if __name__ == "__main__":
    ann = main()
    ann.to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_annotations.parquet"))
    labels = load_labels()
    labels.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_labels.csv"), index=False)
    print(labels.groupby("split")["label"].agg(["count", "sum"]).rename(columns={"sum": "glaucoma"}))

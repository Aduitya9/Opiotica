"""Generate the Week 0 Colab notebooks from source, so they are diffable and reproducible.

Edit THIS file, never the .ipynb - the notebooks are build artefacts.

Run:  python whose-eyes/tools/build_notebooks.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks")


def nb(cells, gpu: bool, gpu_type: str = "L4"):
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                **({"source": body.splitlines(keepends=True)} if kind == "markdown" else
                   {"execution_count": None, "outputs": [],
                    "source": body.splitlines(keepends=True)}),
            }
            for kind, body in cells
        ],
        "metadata": {
            "accelerator": "GPU" if gpu else "None",
            "colab": {"provenance": [], "toc_visible": True,
                      **({"gpuType": gpu_type} if gpu else {})},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# --------------------------------------------------------------------------- shared setup cell
SETUP = '''# --- environment -----------------------------------------------------------
# WHERE THINGS LIVE, and why:
#   PROJECT  this project's code (src/, analysis/)   -> Drive, or a GitHub clone
#   RAW      the 1.5 GB dataset                      -> LOCAL disk: Drive I/O is slow,
#                                                       and re-fetching costs ~3 minutes
#   WORK     crops, weights, results                 -> Drive, so a disconnect costs nothing
REPO_URL = "https://github.com/Aduitya9/Opiotica.git"   # cloned fresh each session
DRIVE_PROJECT = "/content/drive/MyDrive/whose-eyes"     # offline fallback if REPO_URL is ""
WORK = "/content/drive/MyDrive/whose-eyes-work"
RAW = "/content/raw"

import json, os, subprocess, sys
from google.colab import drive

drive.mount("/content/drive")

if REPO_URL:
    if not os.path.isdir("/content/project"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/project"], check=True)
    PROJECT = "/content/project"
else:
    PROJECT = DRIVE_PROJECT
assert os.path.isdir(os.path.join(PROJECT, "src")), (
    f"No src/ under {PROJECT}. Copy the `whose-eyes` folder into your Drive at "
    f"{DRIVE_PROJECT}, or set REPO_URL above.")
sys.path.insert(0, PROJECT)
for sub in ("crops", "runs", "results"):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
os.makedirs(RAW, exist_ok=True)

# --- GPU ---------------------------------------------------------------------
# Batch size is set from the card, not guessed: U-Net + EfficientNet-B4 at 512 px with AMP
# needs roughly 1.5 GB per sample. Everything else in the schedule stays fixed.
import torch

if torch.cuda.is_available():
    GPU = torch.cuda.get_device_name(0)
    VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH = 24 if VRAM > 60 else 16 if VRAM > 35 else 8 if VRAM > 20 else 4
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    GPU, VRAM, BATCH = "cpu", 0.0, 2

print(f"project : {PROJECT}")
print(f"work    : {WORK}   (Drive - survives a disconnect)")
print(f"raw     : {RAW}   (local - re-fetched each session)")
print(f"gpu     : {GPU}  {VRAM:.0f} GB  ->  batch size {BATCH}")
'''

HF_SETUP = '''# --- Hugging Face: the archival copy ------------------------------------------
# The token comes from Colab Secrets (key icon in the left sidebar) under the name HF_TOKEN.
# It is never written into a cell, a file, or this repository.
!pip -q install -U huggingface_hub

from huggingface_hub import HfApi

HF_REPO = "Aduitya9/opiotica-whose-eyes"          # PRIVATE: it holds REFUGE-derived pixels
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception as error:
    HF_TOKEN = None
    print("No HF_TOKEN in Colab Secrets:", error)

PUSH_TO_HF = HF_TOKEN is not None
hf = HfApi(token=HF_TOKEN) if PUSH_TO_HF else None
print("upload enabled" if PUSH_TO_HF else
      "upload DISABLED - add HF_TOKEN under Secrets to archive this run; "
      "everything still works locally and in Drive")
'''

GPU_GUARD = '''# --- every compared run must sit on the SAME card ---------------------------
# TF32 and cuDNN kernels differ between L4 and A100, so a noise floor measured on one card
# and extremes measured on the other would put a hardware difference inside Delta_model.
# The gate refuses to report across mixed hardware.
def assert_same_gpu(frame, column="gpu"):
    cards = sorted(set(frame[column].dropna()))
    if len(cards) > 1:
        raise RuntimeError(
            "Runs span more than one GPU: " + ", ".join(cards) + ".\\n"
            "Delete whose-eyes-work/runs and whose-eyes-work/results/gate_runs.csv and redo "
            "all 11 runs on one card, or the comparison is contaminated.")
    return cards[0] if cards else "unknown"
'''

# ---------------------------------------------------------------- notebook 00
NB00 = [
    ("markdown", """# 00 — Inventory, integrity, and the two decisions that must be made before any training

**Runtime → Change runtime type → CPU.** This notebook trains nothing, and running it on a
GPU just burns compute units.

It exists to settle three things on the record:

1. **What is actually in the release** — counts, resolutions, and whether all 7 experts'
   masks are present for every eye.
2. **The camera/split confound**, re-derived here rather than taken on trust. It is the fact
   that made a camera-versus-convention 2×2 impossible, so it belongs in your own output.
3. **Two frozen decisions**: the vCDR measured from the *official per-expert PNG masks*
   (not the ellipse sheets), and the **crop margin**, chosen once, from the data, before any
   model exists — so it can never be tuned against the result.

About 20 minutes, most of it the one-pass mask scan, which checkpoints as it goes.
"""),
    ("code", SETUP),
    ("code", '''# --- get REFUGE-MultiRater ---------------------------------------------------
# The repo holds ONE 1.5 GB zip, not an unpacked tree, so it is fetched and extracted.
!pip -q install huggingface_hub

from huggingface_hub import hf_hub_download
import glob, zipfile

RELEASE_GLOB = os.path.join(RAW, "**", "Training-400")
if not glob.glob(RELEASE_GLOB, recursive=True):
    zip_path = hf_hub_download(
        repo_id="realslimman/REFUGE-MultiRater",
        filename="REFUGE-Multirater.zip",
        repo_type="dataset",
        local_dir="/content/hf",
    )
    print(f"downloaded {os.path.getsize(zip_path) / 1e9:.2f} GB; extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    print("extracted")
else:
    print("already extracted this session")

RELEASE = sorted({os.path.dirname(p) for p in glob.glob(RELEASE_GLOB, recursive=True)})[0]
print("release root:", RELEASE)
print(sorted(os.listdir(RELEASE)))
'''),
    ("code", '''# --- inventory: resolution and file completeness, one row per eye ------------
import pandas as pd
from PIL import Image
from src import refuge

INVENTORY = os.path.join(WORK, "results", "inventory.csv")
rows = []
for split, image_id, folder in ([] if os.path.exists(INVENTORY)
                                else refuge.iter_images(RELEASE)):
    files = set(os.listdir(folder))
    jpg = next((f for f in files if f.lower().endswith((".jpg", ".jpeg"))), None)
    width, height = Image.open(os.path.join(folder, jpg)).size if jpg else (0, 0)
    rows.append({
        "split": split, "camera": refuge.CAMERA[split], "image_id": image_id,
        "width": width, "height": height,
        "has_official_cup": f"{image_id}_cup.bmp" in files,
        "has_official_disc": f"{image_id}_disc.bmp" in files,
        "n_expert_cup": sum(f"{image_id}_seg_cup_{k}.png" in files for k in range(1, 8)),
        "n_expert_disc": sum(f"{image_id}_seg_disc_{k}.png" in files for k in range(1, 8)),
    })
if os.path.exists(INVENTORY):          # re-runs are cheap: this takes a few minutes otherwise
    inventory = pd.read_csv(INVENTORY, dtype={"image_id": str})
    print("loaded the saved inventory; delete results/inventory.csv to rebuild it\\n")
else:
    inventory = pd.DataFrame(rows)
    # the clinical diagnosis, already resolved in analysis/r1_labels.csv: the release's own
    # REFUGE1*.csv files repeat folder ids across rows, so 5 eyes have conflicting label rows
    # and were dropped. 1,195 eyes keep a clean label; 119 of them are glaucoma.
    clinical = pd.read_csv(os.path.join(PROJECT, "analysis", "r1_labels.csv"),
                           dtype={"folder": str}).rename(columns={"folder": "image_id"})
    inventory = inventory.merge(clinical, on=["split", "image_id"], how="left")
    inventory.to_csv(INVENTORY, index=False)

print(f"{len(inventory)} eyes")
print(f"with a clean clinical label: {int(inventory.label.notna().sum())}  "
      f"(glaucoma {int(inventory.label.fillna(0).sum())})")
incomplete = inventory[(inventory.n_expert_cup < 7) | (inventory.n_expert_disc < 7)
                       | ~inventory.has_official_cup | ~inventory.has_official_disc]
print(f"eyes missing any mask: {len(incomplete)}")
if len(incomplete):
    display(incomplete.head(20))
'''),
    ("code", '''# --- the confound, in your own output ----------------------------------------
print("RESOLUTION x SPLIT - if each split has exactly one resolution, camera IS split:\\n")
print(pd.crosstab(inventory.split,
                  inventory.width.astype(str) + " x " + inventory.height.astype(str)).to_string())
print("""
Read it carefully. Training-400 is Zeiss at 2124x2056; Validation-400 and Test-400 are
Canon at 1634x1634. No eye appears on both cameras, so camera, split and patient set are the
same variable. A clean camera-versus-convention 2x2 cannot be built from REFUGE, and the
pre-registration treats the cross-dataset result as a LOWER BOUND instead.""")
'''),
    ("code", '''# --- one pass over every mask: per-expert vCDR AND disc boxes ----------------
# Fix A2: the training targets are the official per-expert PNGs, so the convention offsets
# must be re-derived from THOSE, not from the ellipse sheets the pilots used.
# Checkpoints every 200 eyes, so a Colab disconnect costs at most a couple of minutes.
import time
import numpy as np
from src import crops as crops_mod

MEASURED = os.path.join(WORK, "results", "png_measurements.csv")
done = pd.read_csv(MEASURED, dtype={"image_id": str}) if os.path.exists(MEASURED) \\
    else pd.DataFrame(columns=["image_id"])
seen = set(done.image_id)
records, t0 = [], time.time()

for n, (split, image_id, folder) in enumerate(refuge.iter_images(RELEASE), 1):
    if image_id in seen:
        continue
    rec = {"split": split, "camera": refuge.CAMERA[split], "image_id": image_id}
    for k in range(1, 8):
        cup, disc = refuge.expert_masks(folder, image_id, k)
        ct, cl, cb, cr = crops_mod.mask_box(cup)
        dt, dl, db, dr = crops_mod.mask_box(disc)
        rec[f"cup_v_user{k}"], rec[f"disc_v_user{k}"] = cb - ct, db - dt
        rec[f"box_user{k}"] = f"{dt},{dl},{db},{dr}"
    ocup, odisc = refuge.official_masks(folder, image_id)
    oct_, ocl, ocb, ocr = crops_mod.mask_box(ocup)
    odt, odl, odb, odr = crops_mod.mask_box(odisc)
    rec["cup_v_official"], rec["disc_v_official"] = ocb - oct_, odb - odt
    rec["box_official"] = f"{odt},{odl},{odb},{odr}"
    records.append(rec)
    if len(records) % 200 == 0:
        pd.concat([done, pd.DataFrame(records)]).to_csv(MEASURED, index=False)
        print(f"  {n:4d}/1200   {time.time() - t0:5.0f}s   (checkpointed)")

png = pd.concat([done, pd.DataFrame(records)], ignore_index=True)
for name in [f"user{k}" for k in range(1, 8)] + ["official"]:
    png[f"vcdr_{name}"] = png[f"cup_v_{name}"] / png[f"disc_v_{name}"]
png.to_csv(MEASURED, index=False)
print(f"\\n{len(png)} eyes measured in {time.time() - t0:.0f}s")
'''),
    ("code", '''# --- do the PNG masks give the same convention offsets as the ellipses? ------
from src import vcdr

RATERS = [f"user{k}" for k in range(1, 8)]
wide = png[[f"vcdr_{r}" for r in RATERS]].rename(columns=lambda c: c.replace("vcdr_", ""))
ellipse = pd.read_csv(os.path.join(PROJECT, "analysis", "r6_official_merge.csv"),
                      dtype={"folder": str})

compare = pd.DataFrame({
    "from PNG masks": vcdr.offsets(wide, RATERS),
    "from ellipse sheets": vcdr.offsets(ellipse, RATERS),
})
compare["difference"] = compare["from PNG masks"] - compare["from ellipse sheets"]
compare["refer@0.6 (PNG)"] = [vcdr.referral_rate(wide[r], 0.6) for r in RATERS]
print(compare.round(4).to_string())

delta_png = float((wide["user1"] - wide["user7"]).mean())
print(f"\\nDelta_ann from PNG masks : {delta_png:.4f}")
print(f"Delta_ann in the prereg  : {vcdr.DELTA_ANN:.4f}")
if abs(delta_png - vcdr.DELTA_ANN) > 0.01:
    print("\\n*** AMEND THE PRE-REGISTRATION BEFORE RUNNING NOTEBOOK 02. The gate is defined "
          "against Delta_ann, so a shift of more than 0.01 goes into §10 with today's date "
          "and a reason - never silently.")
else:
    print("\\nWithin tolerance: the gate keeps Delta_ann = 0.269.")
'''),
    ("code", '''# --- decision 2: the crop margin, chosen once, from the data -----------------
# The frozen box must hold the UNION of all seven experts' discs on every eye, or a liberal
# expert's outline gets clipped and the comparison is rigged before training starts.
def column(frame, name):
    return np.array([[int(v) for v in s.split(",")] for s in frame[name]], dtype=float)

m = png.merge(inventory[["image_id", "height", "width"]], on="image_id", how="left")
official = column(m, "box_official")
expert = np.stack([column(m, f"box_user{k}") for k in range(1, 8)])     # (7, N, 4)
union = np.stack([expert[:, :, 0].min(0), expert[:, :, 1].min(0),
                  expert[:, :, 2].max(0), expert[:, :, 3].max(0)], axis=1)
cy = (official[:, 0] + official[:, 2]) / 2
cx = (official[:, 1] + official[:, 3]) / 2
diameter = np.maximum(official[:, 2] - official[:, 0], official[:, 3] - official[:, 1])
H, W = m.height.to_numpy(float), m.width.to_numpy(float)

# The box may extend past the image border; notebook 01 pads those parts. It is NOT clamped
# to the image: clamping collapses the box for a disc near an edge, which capped containment
# at 1190/1200 however large the margin grew. See PREREGISTRATION.md Amendment 2.
results = []
for margin in np.arange(1.2, 3.01, 0.1):
    half = margin * diameter / 2
    held = ((cy - half <= union[:, 0]) & (cx - half <= union[:, 1])
            & (cy + half >= union[:, 2]) & (cx + half >= union[:, 3]))
    outside = np.maximum(0, -(cy - half)) + np.maximum(0, (cy + half) - H) \\
        + np.maximum(0, -(cx - half)) + np.maximum(0, (cx + half) - W)
    results.append({"margin": round(float(margin), 2), "contained": int(held.sum()),
                    "fraction": float(held.mean()),
                    "eyes_needing_padding": int((outside > 0).sum())})
margins = pd.DataFrame(results)
print(margins.to_string(index=False))

safe = margins[margins.fraction >= 1.0]
if not len(safe):
    raise SystemExit(
        "\\nSTOP. No margin in 1.2-3.0 contains all seven experts' discs on every eye.\\n"
        "Do NOT pick the largest and carry on - that silently trades away disc resolution\\n"
        "on every crop. Find out which eyes fail and why first; the earlier version of this\\n"
        "cell fell back to 3.0 here and it took a code audit to notice.")
MARGIN = float(safe.margin.min())
print(f"\\nCHOSEN MARGIN = {MARGIN}  (smallest containing every expert's disc on every eye)")
print(f"eyes needing any padding at this margin: "
      f"{int(margins.loc[margins.margin == MARGIN, 'eyes_needing_padding'].iloc[0])}")

json.dump({"margin": MARGIN, "anchor": "official-merge disc",
           "rule": "smallest margin containing the 7-expert disc union on 100% of eyes",
           "table": results},
          open(os.path.join(WORK, "results", "crop_margin.json"), "w"), indent=2)
print("frozen to results/crop_margin.json - never recomputed later")
'''),
    ("markdown", """## What to record before moving on

| | expected |
|---|---|
| Eyes with all 7 experts + the official merge | **1200** |
| Eyes with a clean clinical label | **1195** (119 glaucoma) |
| Resolutions per split | Zeiss 2124×2056 / Canon 1634×1634 |
| `Delta_ann` from the PNG masks | compare against **0.269**; amend §10 if it moved > 0.01 |
| Chosen crop margin | written to `results/crop_margin.json` |

Any eye missing a mask is **excluded and counted in the paper**, never silently dropped.

Next: `01_frozen_localiser.ipynb` — switch the runtime to **L4** for that one.
"""),
]

# ---------------------------------------------------------------- notebook 01
NB01 = [
    ("markdown", """# 01 — The frozen disc localiser, and the crops every label source will share

**Runtime → Change runtime type → GPU → L4.** An A100 buys nothing here: this is one short
training run and then a lot of disk writing.

Audit fix A1. If the crop moves when the convention changes, a model trained on a liberal
expert is also looking at a different field of view, and nothing downstream can separate the
two. So one localiser is trained **once**, on the official merge only, then frozen; it emits
one box per eye; and all nine label sources are cropped with that same box.

Containment of all seven experts' discs is then **proven on every eye**, not assumed.

About 30–45 minutes. It resumes if Colab drops you.
"""),
    ("code", SETUP),
    ("code", '''!pip -q install "segmentation-models-pytorch==0.4.0" "timm>=1.0.9" || pip -q install segmentation-models-pytorch

import glob, hashlib, time
import numpy as np, pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp

from src import refuge, crops as crops_mod, labels as labels_mod, train as train_mod

RELEASE = sorted({os.path.dirname(p) for p in
                  glob.glob(os.path.join(RAW, "**", "Training-400"), recursive=True)})[0]
inventory = pd.read_csv(os.path.join(WORK, "results", "inventory.csv"), dtype={"image_id": str})
png = pd.read_csv(os.path.join(WORK, "results", "png_measurements.csv"), dtype={"image_id": str})
MARGIN = json.load(open(os.path.join(WORK, "results", "crop_margin.json")))["margin"]
folder_of = {i: f for _, i, f in refuge.iter_images(RELEASE)}
print(f"{len(inventory)} eyes | frozen margin {MARGIN} | batch {BATCH} on {GPU}")
'''),
    ("code", '''# --- the fixed split, drawn once and saved -----------------------------------
# 320 Zeiss eyes to train on, 80 held back for early stopping, 800 Canon eyes to evaluate on.
# Byte-identical for every label source, for the whole study.
SPLIT_FILE = os.path.join(WORK, "results", "zeiss_train80_val20.json")
if os.path.exists(SPLIT_FILE):
    split_ids = json.load(open(SPLIT_FILE))
    print("reusing the existing split - it is never redrawn")
else:
    zeiss = inventory[inventory.camera == "Zeiss"].image_id.tolist()
    shuffled = list(np.random.default_rng(20261113).permutation(zeiss))
    split_ids = {"train": sorted(shuffled[:320]), "val": sorted(shuffled[320:]),
                 "eval": sorted(inventory[inventory.camera == "Canon"].image_id)}
    json.dump(split_ids, open(SPLIT_FILE, "w"), indent=2)
    print("split drawn and frozen")
print({k: len(v) for k, v in split_ids.items()})
'''),
    ("code", '''# --- localiser dataset: whole image -> official disc only --------------------
LOC_SIZE = 512

class LocaliserSet(torch.utils.data.Dataset):
    def __init__(self, ids, augment=False):
        self.ids, self.augment = list(ids), augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        image_id = self.ids[k]
        folder = folder_of[image_id]
        jpg = next(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))
        image = Image.open(os.path.join(folder, jpg)).convert("RGB").resize((LOC_SIZE, LOC_SIZE))
        _, disc = refuge.official_masks(folder, image_id)
        disc = np.array(Image.fromarray(disc.astype(np.uint8) * 255)
                        .resize((LOC_SIZE, LOC_SIZE), Image.NEAREST)) > 127
        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        y = disc[None].astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            x, y = x[:, :, ::-1].copy(), y[:, :, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)

print(len(LocaliserSet(split_ids["train"])), "training eyes for the localiser")
'''),
    ("code", '''# --- train the localiser ONCE, then freeze it --------------------------------
LOC_DIR = os.path.join(WORK, "runs", "localiser")
os.makedirs(LOC_DIR, exist_ok=True)
WEIGHTS = os.path.join(LOC_DIR, "best.pt")

localiser = smp.Unet("efficientnet-b4",
                     encoder_weights=None if os.path.exists(WEIGHTS) else "imagenet",
                     in_channels=3, classes=1)

if os.path.exists(WEIGHTS):
    localiser.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    localiser = localiser.cuda().eval()
    print("localiser already trained - loading the frozen weights, not retraining")
else:
    train_mod.set_seed(20261113)
    localiser = localiser.cuda()
    opt = torch.optim.AdamW(localiser.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loaders = {
        "train": torch.utils.data.DataLoader(LocaliserSet(split_ids["train"], augment=True),
                                             batch_size=BATCH, shuffle=True, num_workers=2,
                                             drop_last=True, pin_memory=True),
        "val": torch.utils.data.DataLoader(LocaliserSet(split_ids["val"]), batch_size=BATCH,
                                           num_workers=2, pin_memory=True),
    }
    best = float("inf")
    for epoch in range(30):
        line = {}
        for phase in ("train", "val"):
            localiser.train(phase == "train")
            total = count = 0
            with torch.set_grad_enabled(phase == "train"):
                for x, y in loaders[phase]:
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        loss = train_mod.dice_bce(localiser(x), y)
                    if phase == "train":
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    total += loss.item() * x.size(0); count += x.size(0)
            line[phase] = total / count
        marker = ""
        if line["val"] < best:
            best, marker = line["val"], "  *"
            torch.save(localiser.state_dict(), WEIGHTS)
        print(f"epoch {epoch:2d}  train {line['train']:.4f}  val {line['val']:.4f}{marker}")
    localiser.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    localiser.eval()

for p in localiser.parameters():
    p.requires_grad = False
print("\\nFROZEN. This model is never trained again in this project.")
'''),
    ("code", '''# --- one box per eye, from the frozen localiser ------------------------------
all_ids = inventory.image_id.tolist()
rows, failures = [], []

with torch.no_grad():
    for n, image_id in enumerate(all_ids, 1):
        folder = folder_of[image_id]
        jpg = next(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))
        full = Image.open(os.path.join(folder, jpg)).convert("RGB")
        W_full, H_full = full.size
        x = torch.from_numpy(np.asarray(full.resize((LOC_SIZE, LOC_SIZE)), dtype=np.float32)
                             .transpose(2, 0, 1)[None] / 255.0).cuda()
        with torch.amp.autocast("cuda"):
            pred = torch.sigmoid(localiser(x))[0, 0].float().cpu().numpy() > 0.5
        if not pred.any():
            failures.append({"image_id": image_id, "reason": "localiser found no disc"})
            continue
        top, left, bottom, right = crops_mod.mask_box(pred)
        sy, sx = H_full / LOC_SIZE, W_full / LOC_SIZE
        disc_full = np.zeros((H_full, W_full), dtype=bool)
        disc_full[int(top * sy):int(bottom * sy), int(left * sx):int(right * sx)] = True
        box = crops_mod.square_box(disc_full, (H_full, W_full), margin=MARGIN)
        rows.append({"image_id": image_id, "height": H_full, "width": W_full,
                     "top": box[0], "left": box[1], "bottom": box[2], "right": box[3]})
        if n % 200 == 0:
            print(f"  {n}/{len(all_ids)}")

boxes = pd.DataFrame(rows)
BOX_FILE = os.path.join(WORK, "crops", "boxes.csv")
boxes.to_csv(BOX_FILE, index=False)
digest = hashlib.sha256(open(BOX_FILE, "rb").read()).hexdigest()
json.dump({"sha256": digest, "margin": MARGIN, "n": len(boxes), "gpu": GPU,
           "failures": failures},
          open(os.path.join(WORK, "crops", "boxes.sha256.json"), "w"), indent=2)
print(f"\\n{len(boxes)} boxes frozen.  sha256 {digest[:16]}...")
print("failures:", failures if failures else "none")
'''),
    ("code", '''# --- PROVE the box holds every expert's disc ---------------------------------
def column(frame, name):
    return np.array([[int(v) for v in s.split(",")] for s in frame[name]], dtype=int)

check = boxes.merge(png, on="image_id", how="inner")
expert = np.stack([column(check, f"box_user{k}") for k in range(1, 8)])
union = np.stack([expert[:, :, 0].min(0), expert[:, :, 1].min(0),
                  expert[:, :, 2].max(0), expert[:, :, 3].max(0)], axis=1)
report = pd.DataFrame({
    "image_id": check.image_id,
    "slack_top": union[:, 0] - check.top.to_numpy(),
    "slack_left": union[:, 1] - check.left.to_numpy(),
    "slack_bottom": check.bottom.to_numpy() - union[:, 2],
    "slack_right": check.right.to_numpy() - union[:, 3],
})
report["contains_union"] = (report[["slack_top", "slack_left",
                                    "slack_bottom", "slack_right"]] >= 0).all(axis=1)
report["padding_fraction"] = [
    crops_mod.padding_fraction((r.top, r.left, r.bottom, r.right), (r.height, r.width))
    for r in check.itertuples()]
report.to_csv(os.path.join(WORK, "results", "containment.csv"), index=False)

padded = report[report.padding_fraction > 0]
print(f"eyes whose box extends past the image border: {len(padded)} / {len(report)}"
      f"  (max padding {100 * report.padding_fraction.max():.1f}% of the crop)")
print("   that padding is black, which is what a fundus surround already is\\n")
bad = report[~report.contains_union]
print(f"eyes where the frozen box clips an expert's disc: {len(bad)} / {len(report)}")
print("\\nsmallest slack in pixels (how close the box came to clipping):")
print(report[["slack_top", "slack_left", "slack_bottom", "slack_right"]].min().to_string())
if len(bad):
    display(bad.head(20))
    print("\\nThese eyes are EXCLUDED and the count goes in the paper. They are not dropped "
          "quietly, and the margin is NOT raised now - that would be tuning after the fact.")
'''),
    ("code", '''# --- export the crops: one .npz per eye, 19 arrays, identical pixels ---------
CROP = train_mod.CONFIG.crop_size
keep = set(report.loc[report.contains_union, "image_id"])
train_set, val_set = set(split_ids["train"]), set(split_ids["val"])
manifest, t0 = [], time.time()

for n, (_, row) in enumerate(boxes.iterrows(), 1):
    image_id = row.image_id
    rel = os.path.join("crops", f"{image_id}.npz")
    if image_id not in keep:
        continue
    if os.path.exists(os.path.join(WORK, rel)):           # resume after a disconnect
        manifest.append({"image_id": image_id, "npz": rel,
                         "camera": inventory.loc[inventory.image_id == image_id,
                                                 "camera"].iloc[0],
                         "part": "train" if image_id in train_set
                                 else "val" if image_id in val_set else "eval"})
        continue
    folder = folder_of[image_id]
    jpg = next(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))
    box = (int(row.top), int(row.left), int(row.bottom), int(row.right))

    def cut(arr, nearest=False):
        # crops_mod.crop pads where the box leaves the image. Plain slicing must NOT be used:
        # a negative index wraps around and pastes the opposite side of the retina into the crop.
        piece = crops_mod.crop(arr.astype(np.uint8) * 255 if nearest else arr, box)
        return np.array(Image.fromarray(piece).resize(
            (CROP, CROP), Image.NEAREST if nearest else Image.BILINEAR))

    arrays = {"image": cut(np.asarray(Image.open(os.path.join(folder, jpg)).convert("RGB")))
                       .astype(np.uint8)}
    cups, discs = [], []
    for k in range(1, 8):
        cup, disc = refuge.expert_masks(folder, image_id, k)
        cup, disc = cut(cup, nearest=True) > 127, cut(disc, nearest=True) > 127
        arrays[f"cup_user{k}"], arrays[f"disc_user{k}"] = cup, disc
        cups.append(cup); discs.append(disc)
    ocup, odisc = refuge.official_masks(folder, image_id)
    arrays["cup_official"] = cut(ocup, nearest=True) > 127
    arrays["disc_official"] = cut(odisc, nearest=True) > 127
    arrays["cup_majority"] = labels_mod.fuse(cups, "majority")
    arrays["disc_majority"] = labels_mod.fuse(discs, "majority")

    np.savez_compressed(os.path.join(WORK, rel), **arrays)
    manifest.append({"image_id": image_id, "npz": rel,
                     "camera": inventory.loc[inventory.image_id == image_id, "camera"].iloc[0],
                     "part": "train" if image_id in train_set
                             else "val" if image_id in val_set else "eval"})
    if n % 100 == 0:
        print(f"  {n}/{len(boxes)}   {time.time() - t0:5.0f}s")

manifest = pd.DataFrame(manifest)
manifest.to_csv(os.path.join(WORK, "crops", "manifest.csv"), index=False)
print(f"\\n{len(manifest)} crops ready in {time.time() - t0:.0f}s")
print(manifest.groupby(["part", "camera"]).size().to_string())
'''),
    ("code", '''# --- the check that must pass before notebook 02 -----------------------------
# Same eye, two conventions. The IMAGE must be identical and the CUPS must not be.
sample = np.load(os.path.join(WORK, manifest.npz.iloc[0]))
liberal, conservative = sample["cup_user1"].sum(), sample["cup_user7"].sum()
print(f"image array      : {sample['image'].shape} {sample['image'].dtype}")
print(f"cup_user1 pixels : {liberal}")
print(f"cup_user7 pixels : {conservative}")
print(f"ratio            : {liberal / max(conservative, 1):.2f}x")
assert liberal != conservative, (
    "user1 and user7 produced the same cup - the export is wrong and nothing after this "
    "point means anything.")
print("\\nOK: one image, different outlines. That difference is the whole experiment.")
'''),
    ("code", HF_SETUP),
    ("code", '''# --- archive the frozen crops and the decisions that produced them -----------
# Why upload at all: the crops are the one expensive artefact (~45 min to rebuild) and they
# are what makes every later run comparable. Why PRIVATE: they contain REFUGE pixels.
if PUSH_TO_HF:
    for name in ("boxes.csv", "boxes.sha256.json", "manifest.csv"):
        hf.upload_file(path_or_fileobj=os.path.join(WORK, "crops", name),
                       path_in_repo=f"crops/{name}", repo_id=HF_REPO, repo_type="dataset",
                       commit_message=f"Frozen crop metadata: {name}")
    for name in ("inventory.csv", "png_measurements.csv", "crop_margin.json",
                 "containment.csv", "zeiss_train80_val20.json"):
        path = os.path.join(WORK, "results", name)
        if os.path.exists(path):
            hf.upload_file(path_or_fileobj=path, path_in_repo=f"results/{name}",
                           repo_id=HF_REPO, repo_type="dataset",
                           commit_message=f"Notebook 00/01 output: {name}")
    print("metadata uploaded\\n")

    # the crops themselves: ~1,200 files, a few hundred MB. Slow; skip if you are short of time
    # and simply re-run this notebook later - nothing downstream needs them to be on the Hub.
    UPLOAD_CROPS = True
    if UPLOAD_CROPS:
        upload = getattr(hf, "upload_large_folder", None)
        if upload:
            upload(folder_path=os.path.join(WORK, "crops"), repo_id=HF_REPO,
                   repo_type="dataset", allow_patterns=["*.npz"])
        else:
            hf.upload_folder(folder_path=os.path.join(WORK, "crops"), path_in_repo="crops",
                             repo_id=HF_REPO, repo_type="dataset", allow_patterns=["*.npz"],
                             commit_message="Frozen crops: image + 18 masks per eye")
        print("crops uploaded")
    print(f"\\nhttps://huggingface.co/datasets/{HF_REPO}")
'''),
    ("markdown", """## Done

`whose-eyes-work/crops/` now holds one `.npz` per eye: the image once, and eighteen masks —
seven experts × (cup, disc), plus majority and the official merge.

Next: `02_noise_floor_gate.ipynb`. Stay on the **same GPU type** you used here if you can;
notebook 02 refuses to report a gate computed across mixed hardware.
"""),
]

# ---------------------------------------------------------------- notebook 02
NB02 = [
    ("markdown", """# 02 — The seed-noise floor, and the pre-registered gate

**Runtime → GPU → L4** (an A100 is roughly 2× faster and several times the compute units;
either is fine, but **all 11 runs must be on the same card**, and this notebook enforces that).

This is the day that decides whether the project has a paper in it.

**The rule, fixed in `PREREGISTRATION.md` §5 before any model existed:**

```
Delta_model = mean over eyes [ vCDR(model trained on user1) - vCDR(model trained on user7) ]
Delta_ann   = 0.269                      (the annotation-level gap, already measured)
SD_seed     = SD over 5 same-label runs of the per-model mean predicted vCDR

PASS  if  Delta_model > 0.135  (at least half the annotation gap survives training)
     AND  Delta_model > 3 x SD_seed      (it clears the noise floor)
```

**The noise floor runs first, on purpose.** Without it, "models inherit conventions" cannot be
falsified, and 0.135 on its own is an assertion rather than a test.

**If it fails, nothing is re-tuned.** The paper becomes the negative result, reported with the
same prominence. Read that again before you run the cells.

11 runs: roughly 2.5–3.5 h on an L4, 1.5–2 h on an A100. Every run is checkpointed, so a
disconnect costs one run at most.
"""),
    ("code", SETUP),
    ("code", '''!pip -q install "segmentation-models-pytorch==0.4.0" "timm>=1.0.9" || pip -q install segmentation-models-pytorch

import time
import numpy as np, pandas as pd
from dataclasses import replace
from src import train as train_mod, vcdr, labels as labels_mod

CFG = replace(train_mod.CONFIG, batch_size=BATCH)

manifest = pd.read_csv(os.path.join(WORK, "crops", "manifest.csv"), dtype={"image_id": str})
train_rows = manifest[manifest.part == "train"]
val_rows = manifest[manifest.part == "val"]
eval_rows = manifest[manifest.part == "eval"]
print(f"train {len(train_rows)} (Zeiss) | val {len(val_rows)} (Zeiss) | "
      f"eval {len(eval_rows)} (Canon)")
print(f"gpu {GPU} | batch {CFG.batch_size} | epochs {CFG.epochs} | patience {CFG.patience}")
print(f"\\nTHE GATE:  Delta_model > 0.135  AND  Delta_model > 3 x SD_seed"
      f"   (Delta_ann = {vcdr.DELTA_ANN})")
'''),
    ("code", GPU_GUARD),
    ("code", '''# --- one reusable run ---------------------------------------------------------
RESULTS = os.path.join(WORK, "results", "gate_runs.csv")

def load_done():
    return pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame(
        columns=["label_source", "seed", "gpu", "mean_vcdr"])

def run(label_source: str, seed: int):
    done = load_done()
    tag = f"{label_source}_seed{seed}"
    if len(done) and ((done.label_source == label_source) & (done.seed == seed)).any():
        print(f"{tag}: already done, skipping")
        return
    print(f"\\n=== {tag} ===")
    t0 = time.time()
    model, info = train_mod.train_one(
        train_mod.CropDataset(train_rows, label_source, WORK, augment=True),
        train_mod.CropDataset(val_rows, label_source, WORK),
        seed=seed, out_dir=os.path.join(WORK, "runs", tag), config=CFG, log_every=5)
    predicted = train_mod.predict_vcdr(
        model, train_mod.CropDataset(eval_rows, label_source, WORK),
        batch_size=CFG.batch_size)
    np.save(os.path.join(WORK, "runs", tag, "eval_vcdr.npy"), predicted)
    row = {"label_source": label_source, "seed": seed, "gpu": GPU,
           "mean_vcdr": float(np.nanmean(predicted)),
           "refer_at_0.6": vcdr.referral_rate(predicted, 0.6),
           "refer_at_0.7": vcdr.referral_rate(predicted, 0.7),
           "best_val_loss": info["best_val_loss"], "minutes": (time.time() - t0) / 60}
    pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
    print(f"  mean vCDR {row['mean_vcdr']:.4f} | refer@0.6 {100 * row['refer_at_0.6']:.1f}%"
          f" | {row['minutes']:.1f} min")
'''),
    ("code", '''# --- STEP 1: the noise floor. 5 seeds, ONE label. Before anything else. ------
for seed in (101, 102, 103, 104, 105):
    run(labels_mod.NOISE_FLOOR_LABEL, seed)

floor = load_done()
floor = floor[floor.label_source == labels_mod.NOISE_FLOOR_LABEL]
assert_same_gpu(floor)
SD_SEED = float(floor.mean_vcdr.std(ddof=1))
print(f"\\nSEED-NOISE FLOOR on '{labels_mod.NOISE_FLOOR_LABEL}' ({len(floor)} seeds, {GPU})")
print(floor[["seed", "mean_vcdr", "refer_at_0.6", "minutes"]].to_string(index=False))
print(f"\\n  SD_seed      = {SD_SEED:.4f}")
print(f"  3 x SD_seed  = {3 * SD_SEED:.4f}   <- any effect smaller than this is noise")
'''),
    ("code", '''# --- STEP 2: the two extremes, 3 seeds each ----------------------------------
for label_source in labels_mod.EXTREMES:
    for seed in (42, 43, 44):
        run(label_source, seed)

runs = load_done()
card = assert_same_gpu(runs)
summary = runs.groupby("label_source").agg(
    mean_vcdr=("mean_vcdr", "mean"), sd=("mean_vcdr", "std"),
    refer_06=("refer_at_0.6", "mean"), n=("seed", "count"))
print(f"all {len(runs)} runs on {card}\\n")
print(summary.round(4).to_string())
'''),
    ("code", '''# --- STEP 3: the gate ---------------------------------------------------------
# Delta_model is computed PER EYE and per seed pair, so it is a like-for-like contrast
# rather than the difference of two separately averaged numbers.
def load(label_source, seed):
    return np.load(os.path.join(WORK, "runs", f"{label_source}_seed{seed}", "eval_vcdr.npy"))

per_seed = [float(np.nanmean(load("user1", s) - load("user7", s))) for s in (42, 43, 44)]
DELTA_MODEL = float(np.mean(per_seed))

verdict = vcdr.gate(DELTA_MODEL, SD_SEED)
verdict["delta_per_seed"] = per_seed
verdict["gpu"] = card
verdict["refer_at_0.6"] = {s: float(summary.loc[s, "refer_06"]) for s in summary.index}
json.dump(verdict, open(os.path.join(WORK, "results", "gate.json"), "w"), indent=2)

print("=" * 74)
print(f"  Delta_ann   (annotations)  {verdict['delta_ann']:.4f}")
print(f"  Delta_model (trained)      {verdict['delta_model']:.4f}    per seed: "
      + ", ".join(f"{d:.4f}" for d in per_seed))
print(f"  retention                  {100 * verdict['retention']:.1f}%     (must be >= 50%)")
print(f"  3 x SD_seed                {verdict['noise_floor_3sd']:.4f}"
      f"    (Delta_model must exceed it)")
print(f"  hardware                   {card}")
print("=" * 74)
print(f"  {verdict['verdict']}")
print("=" * 74)
'''),
    ("code", HF_SETUP),
    ("code", '''# --- archive the gate --------------------------------------------------------
# The verdict and the per-run numbers are small and belong with the paper. The checkpoints are
# ~70 MB each; they are uploaded so a reviewer can reproduce a prediction without retraining.
if PUSH_TO_HF:
    for name in ("gate.json", "gate_runs.csv"):
        hf.upload_file(path_or_fileobj=os.path.join(WORK, "results", name),
                       path_in_repo=f"results/{name}", repo_id=HF_REPO, repo_type="dataset",
                       commit_message=f"Gate output: {name}")
    print("verdict uploaded")

    UPLOAD_WEIGHTS = True          # ~11 x 70 MB for the gate's runs
    if UPLOAD_WEIGHTS:
        hf.upload_folder(folder_path=os.path.join(WORK, "runs"), path_in_repo="runs",
                         repo_id=HF_REPO, repo_type="dataset",
                         allow_patterns=["*/best.pt", "*/history.json", "*/eval_vcdr.npy"],
                         commit_message=f"Gate runs on {card}: weights, history, eval vCDR")
        print("checkpoints uploaded")
    print(f"\\nhttps://huggingface.co/datasets/{HF_REPO}")
'''),
    ("markdown", """## Whichever way it went

**PASS** — go on to the full grid: the other seven label sources, 3 seeds each (27 more U-Net
runs), then RETFound and DINOv3 on an **A100**. The referral-rate table is the paper's headline.

**FAIL** — stop and write the negative result. Do **not** raise the margin, change the
architecture, swap the evaluation set, or lower 0.135. Each of those is why published findings
fail to replicate, and the pre-registration exists to put them out of reach. A clean negative
is still a contribution: the domain-generalisation literature assumes conventions propagate
through training and has never measured it.

Either way, append the outcome and today's date to `PREREGISTRATION.md` §10.
"""),
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, cells, gpu, card in [
        ("00_inventory.ipynb", NB00, False, ""),
        ("01_frozen_localiser.ipynb", NB01, True, "L4"),
        ("02_noise_floor_gate.ipynb", NB02, True, "L4"),
    ]:
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb(cells, gpu, card), fh, indent=1)
        print(f"wrote {os.path.basename(path)}  ({len(cells)} cells, GPU={gpu or '-'})")

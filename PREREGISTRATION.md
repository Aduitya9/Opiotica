# Pre-registration — Whose Eyes?

**Expert labelling conventions, not only cameras, set the referral rate of AI glaucoma screening.**

First committed 13 November 2026, **before any segmentation model was trained**. Everything
below was decided while only *annotation-level* numbers were known (pilots R1–R6, Chákṣu).
No trained-model number existed when this file was written. Amendments are dated at the end
and never replace the original text.

---

## 1. What is already measured (annotations only, no models)

These are the facts this study builds on. They come from the 7 experts' own outlines on the
1,200 REFUGE images and are reproduced by `analysis/r1`–`analysis/r6`.

| Quantity | Value |
|---|---|
| Expert vCDR offset, most liberal (user1) | **+0.148** (bootstrap CI +0.145 … +0.152) |
| Expert vCDR offset, most conservative (user7) | **−0.121** |
| Annotation-level spread, user1 − user7 | **Δ_ann = 0.269** |
| Other five experts | within ±0.022 of the 7-expert mean |
| Friedman χ² across the 7 experts | 4043 (p ≪ 0.001) |
| ICC(2,1) absolute agreement / ICC(3,1) consistency | 0.436 / 0.689 |
| Referral rate at vCDR ≥ 0.6, by expert | **3.3 % … 56.8 %** (17×) |
| Referral rate at vCDR ≥ 0.7, by expert | 1.5 % … 8.1 % |
| AUC for clinical glaucoma, by expert | 0.904 … 0.944; mean-of-7 = 0.953 |
| **Official REFUGE merge** (8th senior specialist) | offset **+0.009**, refer@0.6 **9.8 %**, sens/spec **0.70 / 0.97** |
| Chákṣu, 5 Indian experts, 1,345 images | "suspect" rate **8.6 % … 32.6 %**; median pairwise κ 0.38 |

**What this does *not* say.** The official REFUGE label is *well centred* and is better against
the clinical diagnosis than any individual expert. This study therefore does **not** claim that
REFUGE-trained models learned a skewed label. See §6.

## 2. The question

An annotation-level difference is only interesting if a **trained model inherits it**. So:

> When the only thing that changes is whose outlines the model learned from — identical images,
> identical crops, identical architecture, identical seeds — how much of Δ_ann survives into the
> model's predicted vCDR, and into the referral rate it would produce in deployment?

## 3. Design — the one arm that is free of confounds

Verified from the raw JPEG headers before this file was written:

```
Training-400    400 images   all 2124 x 2056   Zeiss
Validation-400  400 images   all 1634 x 1634   Canon
Test-400        400 images   all 1634 x 1634   Canon
```

Camera, split and patient set are the **same variable** in REFUGE. No eye appears on both
cameras. A camera-versus-convention 2×2 is therefore **impossible** and is not attempted.

**Primary arm (exact).** All seven experts annotated all 1,200 images, so:

* **Train** on Training-400 (Zeiss): 320 images, with a **fixed** internal validation set of 80
  for early stopping. The split is drawn once, saved to `splits/zeiss_train80_val20.json`, and
  is byte-identical for every label source.
* **Evaluate** on Validation-400 + Test-400 (Canon, 800 eyes), never touched during training.
* **Only the label changes.** Images, crops, split, architecture, schedule and seeds are held
  identical. Any difference in the output is convention and nothing else.

Evaluating on the other camera is deliberate: it is the realistic deployment case, and it makes
the finding stronger if convention survives a camera change.

**Robustness check (pre-registered, secondary).** 5-fold cross-validation *within* Training-400,
so train and evaluation share one camera. Reported alongside; it does not replace the primary.

**Arm 2 (confounded, reported as such).** REFUGE → Drishti-GS / RIM-ONE-r3 / Chákṣu gives the
total cross-dataset gap: camera + population + convention, inseparable. Used only to place
Arm 1's effect against the gap the domain-generalisation literature reports.

**Arm 3 (partial relief).** Chákṣu holds the **same five experts across three cameras**, so
convention is fixed while camera varies. Patients still differ between cameras; this is stated
every time the arm is used.

## 4. The label sources — 9 trained, 3 analysed only

**Trained (9):** `user1` … `user7`, `majority` (≥ 4 of 7), `official` (the released merge).

**Analysed but not trained (3):** `unanimous` (7/7), `union` (≥ 1 of 7), `STAPLE`.
Reason, measured in R5: majority (≥ 4/7) and average ≥ 0.5 returned **identical** results
(0.456 / 7.8 % / 0.64), so training both carries no information. Training runs saved: 6.

Training targets are the **official per-expert PNG masks** (`{id}_seg_cup_{1..7}.png`,
`{id}_seg_disc_{1..7}.png`). The ellipse parameters are used for *analysis only*; they agreed
with the rasterised masks to three decimals on the axis check, but the PNGs are what the dataset
authors published as labels.

## 5. The gate — decided before it is run

Let `vCDR(M_L, i)` be model `M_L`'s predicted vertical cup-to-disc ratio on evaluation eye `i`.

```
Δ_model = mean_i [ vCDR(M_user1, i) − vCDR(M_user7, i) ]
R       = Δ_model / Δ_ann              (Δ_ann = 0.269, fixed above)
SD_seed = SD over 5 same-label seeds of the per-model mean predicted vCDR
```

**The seed-noise floor is run first**, on the `official` label, 5 seeds. Without it, "models
inherit conventions" is unfalsifiable.

```
PASS   if  Δ_model > 0.135  (R ≥ 0.50)   AND   Δ_model > 3 × SD_seed
FAIL   otherwise
```

**If it FAILS the study is not abandoned and the design is not changed.** The paper becomes
*"Expert conventions differ sharply in annotations but are washed out by training"* — a negative
result that is itself worth reporting, because the domain-generalisation literature assumes the
opposite without testing it. It will be reported with the same prominence. No new threshold will
be chosen after seeing Δ_model.

**Run budget to reach the gate:** 5 (noise floor) + 3 (user1) + 3 (user7) = **11 runs**.

## 6. Outcomes

**Primary.** Referral rate at vCDR ≥ 0.6 on the 800 Canon eyes, per label source, with a
patient-clustered bootstrap CI (10,000 resamples).

**Secondary.**
1. Referral rate at vCDR ≥ 0.7.
2. Sensitivity and specificity against REFUGE's clinical diagnosis at both cut-offs.
3. AUC for clinical glaucoma, per label source.
4. **Inter-convention spread** per eye: `max_L vCDR(M_L, i) − min_L vCDR(M_L, i)`, and the
   fraction of eyes whose interval straddles the cut-off ("convention-sensitive").
5. Dice and Hausdorff of cup and disc against each label source.

**Explicitly not an outcome.** Accuracy against a "true vCDR". No true vCDR exists — that is the
thesis of this paper. Conformal prediction is used **only** against the clinical diagnosis, where
a ground truth genuinely exists, and never against vCDR.

## 7. The fixes, and what is already known about them

Pre-registered as evaluated, not as working. Annotation-level previews (R3, R4) are listed so a
reader can see what was expected before models existed.

| Fix | Applies to | Annotation-level preview |
|---|---|---|
| **Label-free percentile cut-off** (top 15 %) | SHIFT | sens 0.71–0.82, spec 0.91–0.92; pairwise disagreement 18.7 % → **10.8 %** |
| **Level correction** fitted on one camera, applied to another | SHIFT | referral 2.8–62.4 % → **5.2–18.6 %**; disagreement 20.4 % → **8.6 %** |
| **Local calibration** to 85 % sensitivity | SHIFT | n=5 → 0.74; n=20 → 0.81; **n=40 → 0.83–0.84** |
| **Rater fusion** | SCRAMBLE | AUC 0.904–0.944 → **0.953** |

Offsets are **not constant** (R4): user1 is +0.196 on small cups but +0.099 on large; user4 flips
sign with cup size; user2's AUC falls from 0.971 to 0.836 between disc-size tertiles. So the
correction is a **fitted curve in cup and disc size**, never a single number.

## 8. Statistics

* 3 seeds for every reported model; the seed SD is printed next to every number.
* Bootstrap CIs clustered by patient (mandatory for PAPILA, which has two eyes per patient).
* Friedman for paired comparisons across label sources; no post-hoc test is chosen after
  seeing the data — Nemenyi is fixed now.
* ICC(2,1) and ICC(3,1) reported as a pair; the gap is the systematic component.
* **Known power limit, stated now:** only 119 clinically diagnosed glaucoma cases exist in the
  1,200. Disc-size tertiles hold 33 / 38 / 48 cases. Subgroup AUCs are exploratory and will be
  labelled as such.

## 9. Limitations fixed in advance

1. **The camera/patient confound** (§3). Arm 2 yields a lower bound: *"convention alone accounts
   for at least X of the Y total cross-dataset gap"*, never a clean decomposition.
2. REFUGE's clinical label is, in the organisers' words, not based on the fundus image only —
   so disc appearance is part of the diagnosis. The label is independent of **these seven
   annotations**, but not of disc assessment in general. Stated in every sensitivity table.
3. AIROGS' own reference is itself grader-assigned. Comparisons against it are
   grader-versus-grader, not grader-versus-truth.
4. RIGA is **CC BY-NC**: used in the paper, excluded from any released artefact.
5. Chákṣu Arm 3 varies camera and patients together.

## 10. Amendments

Anything changed after 13 November 2026 is appended here with a date and a reason. The text
above is never edited.

### Amendment 1 — 13 November 2026 — AIROGS access and licence

**No model had been trained when this was written. The gate in §5 is untouched.**

§9 limitation 3 and `docs/datasets.md` stated that AIROGS requires a manually approved access
request. **That was wrong.** Checked against the Zenodo record and its API on 13 November 2026:
record 5793241 is **open access** — 7 files, 54.9 GB, direct download, no request, no login.

The real constraint is the licence, which was recorded only as "see record":

> **CC BY-NC-ND 4.0** — Attribution, **Non-Commercial, No Derivatives**.
> AIROGS-Lite (record 7178671) carries the same licence.

Consequences, adopted now rather than discovered at submission:

1. AIROGS is used to **compute statistics only** — referral rates and how they move when the
   convention changes. Measurements are reported; nothing derived from its images is
   distributed.
2. No AIROGS crop, derived image, or AIROGS-trained checkpoint enters any released artefact,
   the demo app included. ND makes this stricter than the NC-only case of RIGA.
3. The paper's data-availability statement names AIROGS as open under CC BY-NC-ND and states
   that this study redistributes nothing derived from it.

This removes the only claimed scheduling dependency in Week 0; nothing else in the plan moves.

### Amendment 2 — 13 November 2026 — the crop box pads instead of clamping

**No model had been trained when this was written. The gate in §5 is untouched.**

Notebook 00 ran and its margin table plateaued: 1,190 of 1,200 eyes contained the seven
experts' disc union at margin 1.8, and **no larger margin ever did better** — 1.9, 2.0 … 3.0
all returned 1,190. §3's rule, *"the smallest margin containing the union on 100 % of eyes"*,
was therefore unsatisfiable, and the cell's fallback silently selected the **largest** margin,
3.0.

**Cause** — a defect in `src/crops.py:square_box`, not a property of the data. The box was
clamped to stay square *and* inside the image, so for a disc near a border `half` collapsed to
the edge distance and the opposite side of the union got clipped. Confirmed on all 1,200 eyes:
the ten failures sit a median **167 px** from the nearest border against **419 px** for the
release as a whole, i.e. about one disc radius from the edge.

**Fix** — the box may now extend past the border and `crops.crop` pads those parts with black,
which is what a fundus surround already is. Plain slicing is explicitly forbidden there: a
negative index wraps around and pastes the opposite side of the retina into the crop. With
padding, containment reaches **1200/1200 at margin 1.8**.

**Consequences.**

1. The pre-registered rule in §3 is now satisfiable and is applied unchanged. The margin
   becomes **1.8**, not 3.0.
2. This is a correction, not a tuning: no model exists, no gate number exists, and the margin
   applies identically to every label source. Choosing 3.0 would have shrunk the optic disc to
   about a third of each 512 px crop instead of 56 %, throwing away resolution on the one
   structure the study measures, for no gain.
3. The fallback is gone. The cell now stops with an explanation rather than choosing for you.
4. `tests/test_vcdr.py` gains two regression tests: an edge-adjacent disc must be contained by
   padding and not by clipping, and `crop` must pad rather than wrap.
5. Notebook 00 must be re-run to re-derive the margin. It now reloads the saved inventory and
   measurements, so the re-run costs minutes, not the full pass.

# Whose Eyes?

**Expert labelling conventions, not only cameras, silently set the referral rate of AI glaucoma
screening.**

A model inherits the labelling convention of whoever drew its training masks. On the same 1,200
REFUGE images, the seven experts who annotated them would refer between **3.3 % and 56.8 %** of
eyes at the standard vCDR ≥ 0.6 cut-off — a seventeen-fold difference, on identical pixels. This
project measures how much of that survives into a trained model and into a deployed referral
rate, and then fixes it.

```
+========================================================================+
|  SAME 1,200 EYES. SAME PIXELS. SEVEN EXPERTS.                          |
+========================================================================+
|                                                                        |
|  conservative <------------|------------------------> liberal         |
|   user7   user3 user4  OFFICIAL  user2              user1             |
|   -0.121  -0.022 -0.015  +0.009  +0.007            +0.148             |
|                                                                        |
|   refer@0.6   3.3%    ...    9.8%    ...           56.8%              |
|                                                                        |
|  ICC(2,1) 0.436  vs  ICC(3,1) 0.689                                    |
|  -> the gap between them is the part a level correction can remove     |
|     with no labels at all.                                             |
+========================================================================+
```

## What is already measured, and what is not

**Measured** (annotations only, no models — `analysis/`, reproduced by `tests/`):
expert offsets, referral-rate spread, ICC pair, Friedman χ² = 4043, the fusion-rule ladder,
and the Chákṣu replication where five Indian experts call 8.6 %–32.6 % of the same 1,345 images
glaucoma-suspect.

**Not yet measured:** whether a trained model inherits any of it. That is the day-4 gate in
`PREREGISTRATION.md` §5, and it is written down before the first model is trained.

**Explicitly not claimed:** that REFUGE's released label is skewed. It is not — it sits
+0.009 from the seven-expert mean with sensitivity/specificity 0.70/0.97, better than every
individual expert. The real exposure is *cross-dataset* conventions and single-expert datasets.

## Layout

```
PREREGISTRATION.md     the gate, the outcomes, the limitations - written first
docs/week0.md          the day-by-day plan
docs/datasets.md       every dataset: licence, access state, and where it is stored
docs/models.md         every model: the question it answers, and what it costs
docs/storage.md        why pixels are private and measurements are public
docs/colab.md          the Colab links, runtimes, and the same-GPU rule
src/refuge.py          verified readers: palette BMPs, per-expert PNGs, ellipse sheets
src/vcdr.py            vCDR, offsets, referral rates, ICC, cluster bootstrap, the gate
src/crops.py           the frozen box and its containment proof
src/labels.py          the 9 trained label sources and the 3 analysed-only ones
src/train.py           one training loop, identical for every label source
notebooks/00..02       inventory -> frozen localiser -> the gate
analysis/r1..r6        the pilots, and the CSVs the pre-registration quotes
tests/                 12 tests that re-derive every quoted number
tools/                 notebook builder, link checker
```

## Start here

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Then `docs/week0.md`, day 1.

## Licence, and what it does and does not cover

The `LICENSE` file is **CC0 1.0**, and it applies to **the code and documentation in this
repository only** — `src/`, `tools/`, `tests/`, `notebooks/`, `docs/`, `PREREGISTRATION.md`.

It does **not** apply to the files in `analysis/`. Those are derived from third-party datasets
whose terms travel with them and cannot be waived by a downstream author:

* `chaksu_decisions.csv` derives from **Chákṣu, CC BY 4.0** — reuse requires attribution.
  The citation is in [analysis/README.md](analysis/README.md).
* `r1_labels.csv`, `r5_fusion_vcdr.csv`, `r6_official_merge.csv` derive from **REFUGE /
  REFUGE-MultiRater** and are used under the REFUGE challenge's own terms.

No image pixels are published in this repository at all. Crops and model weights live in a
private Hugging Face repo; [docs/storage.md](docs/storage.md) explains the split.

## Honesty notes

* Camera, split and patient set are the **same variable** in REFUGE — no eye appears on both
  cameras. A camera-versus-convention 2×2 is impossible and is not attempted. The cross-dataset
  result is reported as a **lower bound**.
* There is no "true vCDR", so nothing is calibrated against one. Uncertainty is reported as
  **inter-convention spread**, which needs no ground truth. Conformal prediction is used only
  against the clinical diagnosis.
* Only 119 clinically diagnosed glaucoma cases exist in the 1,200. Subgroup analyses are
  labelled exploratory.

# Whose Eyes?

**Expert labelling conventions, not only cameras, silently set the referral rate of AI glaucoma
screening.**

A model inherits the labelling convention of whoever drew its training masks. On the same 1,200
REFUGE images, the seven experts who annotated them would refer between **3.4 % and 66.7 %** of
eyes at the standard vCDR ≥ 0.6 cut-off — a **19.5-fold** difference, on identical pixels. This
project measures how much of that survives into a trained model and into a deployed referral
rate, and then fixes it.

Those figures are measured from the **official per-expert PNG masks**, which are what a model
actually trains on. The middle five experts sit in a narrow band, **8.3 %–13.3 %**; the spread
is driven by the two extremes, and user1's figure is the least stable number in the study
because their distribution is centred on the cut-off (`PREREGISTRATION.md` §10, Amendment 3).

```
+========================================================================+
|  SAME 1,200 EYES. SAME PIXELS. SEVEN EXPERTS.                          |
+========================================================================+
|                                                                        |
|  vCDR offset from the seven-expert consensus, per expert:             |
|                                                                        |
|  conservative <-----------|------------------------> liberal          |
|    user7  user4 user3 user6 user5 user2              user1            |
|   -0.113  -.018 -.010 -.009 -.004 +.001             +0.152            |
|                             ^                                          |
|                       OFFICIAL MERGE +0.000                            |
|                                                                        |
|  refer@0.6   3.4%   13.2% 10.8%  9.1%  8.2% 10.0%    66.7%            |
|                                                                        |
|  ICC(2,1) 0.438  vs  ICC(3,1) 0.693                                    |
|  -> the gap between them is the part a level correction can remove     |
|     with no labels at all.                                             |
+========================================================================+
```

Two things in that picture matter more than the headline:

* **The released REFUGE label is not skewed.** It sits at **+0.000** from the consensus. So
  this project does *not* claim that REFUGE-trained models learned a biased label — the
  exposure is *cross-dataset* conventions and single-expert datasets being a lottery.
* **Offset does not determine referral rate.** user4 is the second most conservative by offset
  yet refers 13.2 %, more than user2 at +0.001. Where an expert's distribution sits relative to
  the cut-off matters as much as where its centre is.

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
analysis/r1..r7        the pilots, and the CSVs the pre-registration quotes
tests/                 17 tests that re-derive every quoted number
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

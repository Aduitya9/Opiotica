---
license: other
license_name: mixed-upstream-terms
pretty_name: "Whose Eyes? — derived measurements, frozen crops and run outputs"
tags:
  - glaucoma
  - fundus
  - inter-observer-variability
  - annotation-bias
  - medical-imaging
---

# Whose Eyes? — working data

**This repository is PRIVATE on purpose.** It holds artefacts derived from the REFUGE fundus
images, and REFUGE's terms do not grant the right to redistribute its imagery. Keeping it
private means the files are stored and versioned, not published.

Code, pre-registration and the public derived measurements:
**<https://github.com/Aduitya9/Opiotica>**

## The study in one paragraph

A segmentation model inherits the labelling convention of whoever drew its training masks. On
the same 1,200 REFUGE eyes, the seven experts who annotated them would refer between **3.3%
and 56.8%** at the standard vCDR ≥ 0.6 cut-off — a seventeen-fold spread on identical pixels.
This project measures how much of that survives into a trained model and into a deployed
referral rate, under a gate fixed before the first model was trained.

## Contents

| Path | What it is | Contains pixels? |
|---|---|---|
| `analysis/*.csv` | per-expert vCDR, fusion rules, clinical labels, Chákṣu decisions | no |
| `crops/` | one `.npz` per eye: the frozen crop + 18 masks | **yes** |
| `runs/` | model checkpoints and per-run evaluation vCDR | no, but derived |
| `results/` | inventory, containment report, the gate's verdict | no |

`crops/` and `runs/` are written by notebooks 01 and 02 and uploaded afterwards; the analysis
tables are here from the start so the notebooks can run without the GitHub checkout.

## Provenance and attribution

**Chákṣu** — CC BY 4.0, attribution required wherever reused:

> Kumar, J. R. H., Seelamantula, C. S., Gagan, J. H., Kamath, Y. S., Kuzhuppilly, N. I. R.,
> Vivekanand, U., Gupta, P., & Patil, S. *Chákṣu: A glaucoma specific fundus image database.*
> Scientific Data (2023). <https://doi.org/10.6084/m9.figshare.20123135>

**REFUGE / REFUGE-MultiRater** — used under the REFUGE challenge's terms
(<https://refuge.grand-challenge.org/>). The public mirror's card states apache-2.0, but that
is the mirror uploader's label and does not transfer a right to re-publish the imagery.

**RIGA** is CC BY-NC and is deliberately **absent** from this repository, so that nothing
released from this project inherits the non-commercial clause.

## Reproducing it instead of downloading it

Everything here is regenerable from public sources in about an hour:

1. `notebooks/00_inventory.ipynb` — inventory, and the two frozen decisions
2. `notebooks/01_frozen_localiser.ipynb` — the frozen box and the crops
3. `notebooks/02_noise_floor_gate.ipynb` — the seed-noise floor and the gate

The crops are cached here only to save repeating step 2.

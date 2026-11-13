# Week 0 — in chronological order

Starting 13 November 2026. Budget: 5–6 hours a day. Week 0 has exactly one purpose:

> **Find out whether the project has a paper in it, before spending a month on one.**

Everything here leads to the day-4 gate. Nothing else gets built until that gate is decided.

```
+========================================================================+
|  WEEK 0 AT A GLANCE                                                    |
+========================================================================+
|  DAY 1  access requests + environment      you + me    ~3 h            |
|  DAY 2  notebook 00: inventory             Colab CPU   ~2 h            |
|  DAY 3  notebook 01: frozen localiser      Colab L4    ~3 h            |
|  DAY 4  notebook 02: THE GATE              Colab L4    ~4 h            |
|  DAY 5  branch on the gate                 depends                     |
|  DAY 6  freeze the pre-registration (OSF)  you         ~2 h            |
|  DAY 7  write up Week 0                    both        ~3 h            |
+========================================================================+
```

The three notebooks are already in your Drive and open straight in Colab — links, runtime
settings and the same-GPU rule are in [colab.md](colab.md).

---

## DAY 1 — the things with lead times, first

One of these needs you to accept a licence in your own account. **No account is created and no
licence accepted on your behalf** — those are yours to do.

### 1.1 — AIROGS: nothing to request (correction)

An earlier version of this plan said AIROGS needed a manually approved Zenodo request and told
you to send it first. **That was wrong**, and it was checked on 13 November 2026 against the
record and the Zenodo API: <https://zenodo.org/records/5793241> is **open access**, 7 files,
54.9 GB, direct download, no request and no login. There is no queue and nothing to wait for.
Download it in Week 4 when you actually need it.

What does matter is the licence: **CC BY-NC-ND 4.0** — Attribution, Non-Commercial,
**No Derivatives**. So AIROGS is used to **compute statistics only**. You may publish the
numbers; you may not ship a crop, a derived image, or an AIROGS-trained checkpoint in any
released artefact. This is stricter than RIGA's NC-only terms. It is written into
`PREREGISTRATION.md` §10, Amendment 1.

### 1.2 — RETFound (gated model, one click, but only you can click it)

1. Sign in to Hugging Face.
2. Open <https://huggingface.co/YukunZhou/RETFound_mae_natureCFP>
3. Accept the licence terms on the model card.
4. Create a **read** token and put it in **Colab → Secrets** as `HF_TOKEN`. Do not paste it
   into a chat, a notebook cell, or a commit.

RETFound is not needed until Week 2. Accept the terms now so it is not a surprise then.

### 1.3 — RIGA (replication panel, Week 4)

1. Open <https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z> **in a browser** —
   the page returns 403 to scripts.
2. Accept **CC BY-NC**.
3. `MESSIDOR.zip` needs **Globus**, not a plain download. Set up a Globus account now so this
   is not discovered in Week 4.

RIGA is **non-commercial**. It appears in the paper; nothing RIGA-derived goes into the
released tool, or the tool inherits the NC clause.

### 1.4 — Environment

Colab Pro, L4 for everything in Week 0. A100 is not needed until the RETFound runs in Week 2;
spending compute units on it now is waste.

```bash
pip install -r whose-eyes/requirements.txt
python -m pytest whose-eyes/tests -q
```

All 12 tests must pass. They re-derive the pilot numbers the pre-registration is built on, so a
failure means a number in `PREREGISTRATION.md` has gone stale.

### 1.5 — Put the project where Colab can reach it

**Already done.** `src/`, `analysis/` and the three notebooks are in **My Drive → whose-eyes**,
and the notebooks find them there by default. If you later push to GitHub, set `REPO_URL` at
the top of each notebook instead and it will clone a fresh copy each session.

### 1.6 — Read the pre-registration, properly

Read `PREREGISTRATION.md` end to end, especially §5. It is short on purpose. Once notebook 02
has run you are not allowed to change the threshold, and the point of reading it today is that
you will have agreed to that while you still had nothing invested.

**End of day 1:** RETFound licence accepted, `HF_TOKEN` in Colab Secrets, tests green,
pre-registration read.

---

## DAY 2 — notebook 00: inventory

Run `notebooks/00_inventory.ipynb` top to bottom. CPU runtime is enough; the mask scan is the
slow part at roughly 6–10 minutes.

It settles four things:

| Output | What it is for |
|---|---|
| `inventory.csv` | counts, resolutions, which eyes are missing a mask |
| the resolution × split table | the camera/patient confound, in **your** output, not my claim |
| `png_measurements.csv` | convention offsets re-derived from the **PNG training targets** |
| `crop_margin.json` | the crop margin, chosen once, before any model exists |

**The one thing that can go wrong and must not be glossed over.** The notebook prints
`Delta_ann` computed from the PNG masks and compares it with the pre-registered **0.269**,
which came from the ellipse sheets. If they differ by more than 0.01, stop and add a dated
amendment to `PREREGISTRATION.md` §10 **before** running notebook 02. The gate is defined
against `Delta_ann`; silently letting it move is how a pre-registration becomes decoration.

---

## DAY 3 — notebook 01: the frozen localiser

Run `notebooks/01_frozen_localiser.ipynb` on an **L4**.

```
+------------------------------------------------------------------+
|  train ONE localiser on the official merge  ->  FREEZE it        |
|            |                                                      |
|            v                                                      |
|  one box per eye  ->  crops/boxes.csv  + sha256                  |
|            |                                                      |
|            v                                                      |
|  PROVE the box holds all 7 experts' discs  ->  containment.csv   |
|            |                                                      |
|            v                                                      |
|  export 1 .npz per eye: image + 18 masks, identical pixels       |
+------------------------------------------------------------------+
```

**The check that matters.** The last markdown cell tells you to open one `.npz` and confirm
`cup_user1` is visibly larger than `cup_user7`. user1's cups run about 32% larger and user7's
about 25% smaller, so the difference should be obvious. If they look the same, the export is
broken and every number after this point is meaningless.

Eyes where the box clips an expert's disc are **excluded and counted in the paper**. Do not
raise the margin to rescue them — the margin was chosen yesterday, from the data, on purpose.

---

## DAY 4 — notebook 02: the gate

This is the day. Run `notebooks/02_noise_floor_gate.ipynb` on an **L4**. Eleven runs,
roughly 2–3 hours.

```
+========================================================================+
|  STEP 1   seed-noise floor:  'official' label x 5 seeds   ->  SD_seed  |
|  STEP 2   the extremes:      user1 x 3,  user7 x 3        ->  D_model  |
|  STEP 3   the pre-registered decision                                  |
|                                                                        |
|     PASS  if  D_model > 0.135        (>= half the annotation gap)      |
|           AND D_model > 3 x SD_seed  (clears the noise floor)          |
+========================================================================+
```

The noise floor runs **first**, deliberately. Without it, "models inherit conventions" is
unfalsifiable and 0.135 is an assertion rather than a test.

**If it PASSES**, Week 1 is the remaining 27 U-Net runs across all nine label sources.

**If it FAILS**, the paper becomes *"expert conventions differ sharply in annotations but are
washed out by training"*. That is a real finding — the domain-generalisation literature assumes
the opposite and has never measured it. Do not raise the margin, change the architecture, swap
the evaluation set, or lower 0.135. Each of those is the reason published findings fail to
replicate, and the pre-registration exists specifically to put them out of reach.

Append the outcome and the date to `PREREGISTRATION.md` §10 either way.

---

## DAY 5 — branch

| Gate | What day 5 is |
|---|---|
| **PASS** | Start the 27-run grid. Draft the referral-rate table with the two extremes already in it. |
| **FAIL** | Write the negative result's abstract. Re-aim the system at the Chákṣu replication, where the annotation-level effect is independent and already measured (8.6% – 32.6% suspect rates, median κ 0.38). |

---

## DAY 6 — freeze the pre-registration publicly

Post `PREREGISTRATION.md` to **OSF** (<https://osf.io/>) with the gate's outcome already in it.
A pre-registration posted *after* the result is known is worth nothing unless it says so — so
say so: state the date the gate ran and that the registration was written before it.

You will need an OSF account. **Create it yourself**; I will not.

---

## DAY 7 — write up Week 0

Four short sections, written while it is fresh:

1. **What the annotations say** — the R1–R6 numbers, now re-derived from the PNG masks.
2. **What the confound forces** — why Stage D is a lower bound and not a decomposition.
3. **The noise floor** — `SD_seed`, and why it is reported next to every later number.
4. **The gate** — the rule, the outcome, and what follows from it.

That is the skeleton of the paper's Methods, written before any results exist to rationalise.

---

## Standing rules for the whole project

* **No account is created for you by an agent.** Hugging Face, Zenodo, OSF, Globus — yours.
* **Tokens live in Colab Secrets**, never in a notebook cell, a commit, or a chat message.
* **Raw CPCB-style restricted files and RIGA-derived artefacts never reach a public repo.**
* `torch.load(..., weights_only=True)` everywhere. It is already in `src/train.py`.
* Every reported number carries its **seed SD** beside it.
* Anything changed after 13 November 2026 is a **dated amendment**, never an edit.

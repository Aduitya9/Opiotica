# Datasets — what we use, where it is stored, and how

Every row was checked against the source's own API on **13 November 2026**, including the
parquet column schemas of the Hugging Face mirrors. Four of those checks changed the plan; they
are marked ⚠️ and explained below. Re-check before submission with `python tools/check_links.py`.

---

## 1. The short version

```
+========================================================================+
|  READY TO USE                                                          |
|    REFUGE-MultiRater   7 experts, per-expert masks      THE CORE ARM   |
|    PAPILA (HF)         2 experts, per-expert masks      ready          |
|    Chakshu (HF)        5 expert DECISIONS + 2 fusions   ready          |
|    AIROGS (Zenodo)     open, statistics only            ready          |
+------------------------------------------------------------------------+
|  NEEDS SOURCING FROM THE ORIGINAL SITE                                 |
|    Drishti-GS          the HF mirror has only 30 rows   blocked        |
|    RIM-ONE-r3          HF has the DL release: no masks  blocked        |
|    RIGA                Deep Blue + Globus, CC BY-NC     manual         |
|    Chakshu per-expert  figshare full release, ~200 GB   optional       |
+========================================================================+
```

---

## 2. Full table

| # | Dataset | What it gives this study | Size | Licence | Where the raw data lives | Where our derivatives live |
|---|---|---|---|---|---|---|
| 1 | **REFUGE-MultiRater** ★ | **7 experts' own cup+disc masks on all 1,200 eyes** — the only arm with no confound | 1.497 GB, one zip | REFUGE challenge terms | `/content/raw` (local, ephemeral) | crops → **HF private** |
| 2 | **PAPILA** | 2 experts' own masks; **2 eyes per patient** → forces clustered bootstrap | 2.7 GB | check card | HF mirror, streamed | measurements → GitHub |
| 3 | **Chákṣu** | 5 experts' **decisions** + **majority and median** fused masks on the same eyes | 1.6 GB | **CC BY 4.0** (attribution) | HF mirror, streamed | measurements → GitHub |
| 4 | **AIROGS** | screening scale: 101,442 images, 54,274 subjects, ~500 sites | 54.9 GB, 7 files | **CC BY-NC-ND** | Zenodo, downloaded when needed | **numbers only** → GitHub |
| 5 | **Drishti-GS** ⚠️ | a different fusion rule (≥3 of 4 experts) | 0.12 GB | research use | **IIIT-H original** | measurements → GitHub |
| 6 | **RIM-ONE-r3** ⚠️ | another fusion rule (average of 5) | small | research use | **original site** | measurements → GitHub |
| 7 | **RIGA** ⚠️ | **6 ophthalmologists × 750 images** — independent replication of the offset finding | 12.9 GB | **CC BY-NC** | Deep Blue + Globus | **numbers only** → GitHub |

---

## 3. The four checks that changed the plan

**⚠️ Drishti-GS — the Hugging Face mirror is unusable.**
`dennisheraldi/Drishti_GS` has **30 rows** and two columns, `pixel_values` and a single
`label` image. Drishti-GS proper is 101 images with **four** experts' separate markings, and
the ≥3-of-4 fusion rule is the entire reason it is in this study. The mirror carries neither.
Source it from <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php>.

**⚠️ RIM-ONE — what is on the Hub is the wrong release.**
`ai4ophth/rim_one_dl_dataset` is **RIM-ONE DL**, a 447-row *classification* set:
`fundus image`, `OCT images`, `metadata`, and **no masks at all**. RIM-ONE-**r3** is the one
with five experts' segmentations averaged. It is not on the Hub and must come from the original
Universidad de La Laguna release.

**✅ PAPILA — better than expected.**
`ai4ophth/PAPILA-dataset` carries **`cup_exp1`, `cup_exp2`, `disc_exp1`, `disc_exp2`** as
separate image columns over 488 rows. Both experts' own contours are there, so PAPILA is a
genuine second convention arm, not just a clinical-label set.

**✅ Chákṣu — partly, and the useful part is there.**
`MedOtter/Chaksu` has 1,345 rows with `decision_expert_1..5` (five experts' glaucoma calls),
`disc_mask`/`cup_mask`, and **both `*_majority` and `*_median` fusions** of the same five
experts, plus precomputed `vcdr`, `acdr`, `hcdr`. What it does **not** have is each expert's
individual mask. So:

* decision-level disagreement (8.6 %–32.6 % suspect rates, median κ 0.38) — **usable now**
* majority-vs-median on identical eyes — **usable now**, a free second fusion-rule comparison
* per-expert masks — only in the figshare full release (~200 GB), and **optional**

---

## 4. Where each thing is stored, and why

```
+------------------------------------------------------------------------+
|  RAW IMAGES          /content/raw   (Colab local disk)                 |
|    ephemeral on purpose: Drive I/O dominates the runtime, and          |
|    re-fetching REFUGE from the Hub costs about 3 minutes               |
+------------------------------------------------------------------------+
|  WORKING OUTPUT      Drive: MyDrive/whose-eyes-work/                   |
|    crops, checkpoints, results - survives a Colab disconnect           |
+------------------------------------------------------------------------+
|  ARCHIVE (pixels)    HF PRIVATE: Aduitya9/opiotica-whose-eyes          |
|    frozen crops (.npz), checkpoints, run outputs                       |
|    PRIVATE because a crop is still a REFUGE photograph                 |
+------------------------------------------------------------------------+
|  ARCHIVE (numbers)   GitHub PUBLIC: Aduitya9/Opiotica                  |
|    code, pre-registration, derived measurement CSVs                    |
|    public because the paper's reproducibility claim depends on it      |
+------------------------------------------------------------------------+
```

The split is decided by one question: **does the artefact contain image pixels?** A vertical
cup-to-disc ratio is a measurement and is published. A crop is a photograph of a patient's eye
and is not. [storage.md](storage.md) gives the reasoning in full.

### Per dataset

| Dataset | How we get it | Where raw sits | What we keep, and where |
|---|---|---|---|
| REFUGE-MultiRater | `hf_hub_download` of the single zip, then extract | `/content/raw` | 1,200 `.npz` crops → **HF private**; vCDR tables → **GitHub** |
| PAPILA | stream the parquet from the Hub | not stored | per-expert vCDR + clustered CIs → **GitHub** |
| Chákṣu | stream the parquet from the Hub | not stored | decisions + fusion comparison → **GitHub** |
| AIROGS | direct Zenodo download, Week 4 | `/content/raw`, deleted after | **referral-rate numbers only** → GitHub. ND forbids shipping anything derived from its images |
| Drishti-GS | IIIT-H original | `/content/raw` | fusion-rule vCDR → **GitHub** |
| RIM-ONE-r3 | original site | `/content/raw` | fusion-rule vCDR → **GitHub** |
| RIGA | browser + Globus | local, Week 4 | **replication offsets only** → GitHub. NC keeps it out of any artefact |

---

## 5. Licence constraints that bind the deliverable

| Dataset | Clause | What it forbids |
|---|---|---|
| **AIROGS** | CC BY-**NC-ND** | **No derivatives.** No crop, no derived image, no AIROGS-trained checkpoint in any released artefact. Statistics only. Strictest of the set. |
| **RIGA** | CC BY-**NC** | Nothing RIGA-derived in the released tool, or the tool inherits non-commercial. |
| **Chákṣu** | CC BY 4.0 | Attribution required — citation in [../analysis/README.md](../analysis/README.md). |
| **REFUGE** | challenge terms | Imagery not redistributable. The HF mirror's "apache-2.0" is the mirror uploader's label, not the organisers'. |

**Do not use** `Nj-1111/Glaucoma_Dataset`. It is a 512 px re-release: the optic disc is about
75 px across while the expert offsets are about 10 px, so resizing destroys the effect being
measured.

---

## 6. Links

| Resource | Link |
|---|---|
| REFUGE-MultiRater | <https://huggingface.co/datasets/realslimman/REFUGE-MultiRater> |
| REFUGE challenge (terms) | <https://refuge.grand-challenge.org/> |
| Chákṣu (HF mirror) | <https://huggingface.co/datasets/MedOtter/Chaksu> |
| Chákṣu (full, per-expert masks) | <https://doi.org/10.6084/m9.figshare.20123135> |
| PAPILA (HF mirror) | <https://huggingface.co/datasets/ai4ophth/PAPILA-dataset> |
| PAPILA (original) | <https://figshare.com/articles/dataset/PAPILA/14798004> |
| Drishti-GS (original — use this) | <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php> |
| AIROGS (open) | <https://zenodo.org/records/5793241> |
| AIROGS-Lite | <https://zenodo.org/record/7178671> |
| AIROGS challenge | <https://airogs.grand-challenge.org/data-and-challenge/> |
| RIGA | <https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z> |

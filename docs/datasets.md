# Datasets and models — links, licences, access state

Every link below returned HTTP 200 when last checked on **13 November 2026**. Re-check before
citing in the paper; `tools/check_links.py` does it in one command.

## Core data

| # | Dataset | Link | Size | Experts | Licence | Access |
|---|---|---|---|---|---|---|
| 1 | **REFUGE-MultiRater** — the core | <https://huggingface.co/datasets/realslimman/REFUGE-MultiRater> | 1.5 GB | **7** | card says apache-2.0; **original REFUGE terms govern** | open ✅ |
| 2 | REFUGE challenge (terms of use) | <https://refuge.grand-challenge.org/> | — | — | — | open ✅ |
| 3 | **Chákṣu** — India, 3 cameras | <https://huggingface.co/datasets/MedOtter/Chaksu> | 1.6 GB | 5 | CC BY 4.0 | open ✅ |
| 4 | Chákṣu full release, per-expert masks | <https://doi.org/10.6084/m9.figshare.20123135> | ~200 GB | 5 | CC BY 4.0 | open ✅ |
| 5 | **PAPILA** | <https://huggingface.co/datasets/ai4ophth/PAPILA-dataset> | 2.7 GB | 2 | check card | open ✅ |
| 6 | PAPILA original | <https://figshare.com/articles/dataset/PAPILA/14798004> | — | 2 | CC BY | open ✅ |
| 7 | **Drishti-GS** | <https://huggingface.co/datasets/dennisheraldi/Drishti_GS> | 0.12 GB | 4 | check card | open ✅ |
| 8 | Drishti-GS original (IIIT-H) | <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php> | — | 4 | research use | open ✅ |
| 9 | **AIROGS** — screening scale | <https://zenodo.org/records/5793241> | **54.9 GB**, 7 files | graders | **CC BY-NC-ND 4.0** | **open ✅ — no request needed** |
| 10 | AIROGS challenge page | <https://airogs.grand-challenge.org/data-and-challenge/> | — | — | — | open ✅ |
| 11 | AIROGS-Lite — small subset | <https://zenodo.org/record/7178671> | 4 files | — | **CC BY-NC-ND 4.0** | open ✅ |
| 12 | **RIGA** — replication panel | <https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z> | 12.9 GB | **6** | **CC BY-NC** | ⚠️ browser + Globus |

**AIROGS**: 101,442 images, 54,274 subjects, ~500 sites. **Open access — download it whenever
you need it.** An earlier draft of this file claimed the Zenodo record required a manually
approved request; that was checked on 13 November 2026 against the record and its API and is
false. There is no queue and no lead time.

The real constraint is **CC BY-NC-ND 4.0**: Attribution, Non-Commercial, **No Derivatives**.
AIROGS is therefore used to **compute statistics only**. Measurements are published; no crop,
no derived image and no AIROGS-trained checkpoint goes into any released artefact. ND is
stricter than RIGA's NC-only terms, so AIROGS is the tighter of the two.

**RIGA**: DOI 10.7302/Z23R0R29. The Deep Blue page returns 403 to scripts; open it in a browser.
`MESSIDOR.zip` needs **Globus**, not a plain download. CC BY-NC means RIGA may appear in the
paper but **no RIGA-derived weights or files go into the released tool**, or the tool inherits NC.

**Do not use** `Nj-1111/Glaucoma_Dataset`. It is a 512 px re-release. At that size the optic disc
is about 75 px across while the expert offsets are about 10 px, so resizing destroys the very
effect being measured.

## Models

| Component | Link | Note |
|---|---|---|
| **RETFound** (MAE ViT-L, Nature 2023) | <https://huggingface.co/YukunZhou/RETFound_mae_natureCFP> | ⚠️ **gated** — only the account owner can accept the terms |
| **DINOv3 ViT-L** — general-vision control | <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> | open ✅ |
| **SAM 2.1 Hiera-L** | <https://huggingface.co/facebook/sam2.1-hiera-large> | open ✅ |
| **MedSAM2** | <https://huggingface.co/wanglab/MedSAM2> | open ✅ |
| **segmentation_models.pytorch** — U-Net + EfficientNet-B4 | <https://github.com/qubvel-org/segmentation_models.pytorch> | the workhorse |
| **nnU-Net v2** | <https://github.com/MIC-DKFZ/nnUNet> | strong baseline |

## Why each dataset is in the study

| Dataset | Role it plays, and nothing else |
|---|---|
| REFUGE-MultiRater | **The only exact arm.** 7 experts on the same 1,200 eyes → convention is isolated. |
| Chákṣu | Same 5 experts across 3 cameras → camera varies with convention held fixed (Arm 3). Also the India-specific replication. |
| Drishti-GS | A different **fusion rule** (≥ 3 of 4) on a different population → Arm 2. |
| RIM-ONE-r3 | Average-of-5 fusion → Arm 2. |
| PAPILA | Two experts, **two eyes per patient** → forces patient-clustered bootstrap. |
| AIROGS | Screening scale: how many extra people are referred per 100,000 when the convention moves. |
| RIGA | 6 ophthalmologists × 750 images → independent replication of the offset finding. |

## What only the account owner can do

These need a human to accept terms or be approved. **Never delegate them to an agent**, and
no account or API key is created on the user's behalf:

1. **AIROGS** — open the Zenodo record, click *Request access*, state the research purpose.
2. **RETFound** — sign in to Hugging Face, open the model page, accept the licence.
3. **RIGA** — open Deep Blue in a browser, accept CC BY-NC, set up Globus for `MESSIDOR.zip`.

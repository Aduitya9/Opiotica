# Models — what we train, why each one is in, and what it costs

Every model here answers a question the others cannot. Anything that does not earn its place is
not run: compute units are finite and a model with no question attached is just noise in the
results table.

---

## 1. The short version

```
+========================================================================+
|  MODEL                      ROLE                        GPU    RUNS    |
+========================================================================+
|  U-Net + EfficientNet-B4    the workhorse               L4       32    |
|    - frozen disc localiser    (1, trained once)                        |
|    - seed-noise floor         (5 seeds, one label)                     |
|    - the 9 label sources      (9 x 3 seeds = 27)                       |
+------------------------------------------------------------------------+
|  RETFound  (MAE ViT-L)      does retinal pretraining    A100     12    |
|                             inherit MORE faithfully?                   |
+------------------------------------------------------------------------+
|  DINOv3 ViT-L               the control that separates  A100     12    |
|                             "retinal" from "just big"                  |
+------------------------------------------------------------------------+
|  SAM 2.1 / MedSAM2          zero-shot: where does a     any       0    |
|                             model with NO convention                   |
|                             training actually land?                    |
+------------------------------------------------------------------------+
|  nnU-Net v2                 the baseline a reviewer     L4      opt.   |
|                             will ask about                             |
+========================================================================+
                                                   total trained:  57
```

---

## 2. Each model, and the question it answers

### U-Net + EfficientNet-B4 — the workhorse

<https://github.com/qubvel-org/segmentation_models.pytorch>

Three separate jobs, all with the identical schedule in [`src/train.py`](../src/train.py):

1. **The frozen disc localiser** (notebook 01). Trained **once**, on the official merge only,
   then frozen forever. It emits one box per eye so that every label source is cropped from
   byte-identical pixels. If this model were retrained per convention, the experiment would be
   contaminated at step one.
2. **The seed-noise floor** (notebook 02). The same label trained five times with five seeds.
   The spread is the floor every convention effect must clear. Without it, "models inherit
   conventions" is unfalsifiable.
3. **The nine label sources** × 3 seeds. This is the paper's main table.

Why EfficientNet-B4 and not something larger: 320 training crops. A bigger encoder would
overfit and the study is about the *label*, not about squeezing out Dice.

### RETFound — the "faithful inheritance" test

<https://huggingface.co/YukunZhou/RETFound_mae_natureCFP> — **gated, accept the licence yourself**

A masked-autoencoder ViT-L pretrained on 1.6 M retinal images (Nature, 2023). The interesting
hypothesis, and the one a reviewer will care about: **a better model may inherit its
annotator's convention more faithfully, not less.** If RETFound reproduces the user1−user7 gap
more exactly than the U-Net does, then improving the backbone makes the problem *worse*, which
is the opposite of what the domain-generalisation literature assumes.

Run on 4 label sources (the two extremes, majority, official) × 3 seeds. A100, because ViT-L at
512 px will not fit comfortably on an L4 at a useful batch size.

### DINOv3 ViT-L — the control that makes RETFound interpretable

<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>

Without this, a RETFound result is uninterpretable: you cannot tell whether faithful
inheritance comes from *retinal* pretraining or from simply being a large pretrained ViT.
DINOv3 is the same capacity, same input size, general-vision pretraining. Same 4 × 3 grid.

**This is the single most important control in the study**, and it is the one most likely to
get cut for time. Do not cut it.

### SAM 2.1 and MedSAM2 — zero-shot, no training at all

<https://huggingface.co/facebook/sam2.1-hiera-large> · <https://huggingface.co/wanglab/MedSAM2>

Prompted with the frozen box, never trained on any convention. They answer a question none of
the trained models can: **where does a model that has learned nobody's convention actually
sit?** If zero-shot lands near the consensus, that is evidence the conventions are a training
artefact rather than something inherent to the images. Costs no training time.

### nnU-Net v2 — the baseline a reviewer will ask for

<https://github.com/MIC-DKFZ/nnUNet>

Optional, run once if time allows, purely so the paper can say the effect is not an artefact of
a weak segmentation baseline. It is not part of the convention grid.

---

## 3. What actually ships

Not seven models. One, plus arithmetic:

```
   1 consensus U-Net  (ONNX, ~25 MB)
          |
          v
   vCDR_consensus = 0.58
          |
          +--> fitted per-convention offset curve  (NOT a constant:
          |     user1 is +0.196 on small cups but +0.099 on large)
          v
   user7-style 0.47  |  official 0.59  |  user1-style 0.73
   REFER?     NO     |      NO         |      YES
          -> FLAG: CONVENTION-SENSITIVE
```

Seven models would not run on a free CPU Space, and shipping one model plus the measured offset
curves *demonstrates the paper's own finding* rather than merely asserting it.

Nothing RIGA- or AIROGS-derived goes into the shipped artefact: RIGA is CC BY-NC and AIROGS is
CC BY-NC-**ND**. See [datasets.md](datasets.md) §5.

---

## 4. Compute budget

| Stage | Model | Runs | GPU | Rough time |
|---|---|---|---|---|
| Localiser | U-Net B4 | 1 | L4 | 20 min |
| Seed-noise floor | U-Net B4 | 5 | L4 | 1.0 h |
| **The gate** (user1, user7) | U-Net B4 | 6 | L4 | 1.5 h |
| Remaining 7 label sources | U-Net B4 | 21 | L4 | 4.5 h |
| RETFound grid | ViT-L | 12 | A100 | 5 h |
| DINOv3 grid | ViT-L | 12 | A100 | 5 h |
| Zero-shot | SAM 2.1 / MedSAM2 | 0 | any | 30 min inference |
| | | **57** | | **~18 h** |

Batch size is set from the detected card, never guessed: 4 on a T4, 8 on an L4, 16 on a 40 GB
A100, 24 on an 80 GB A100. Nothing else in the schedule varies, because the whole study rests
on only the label changing.

**The hardware rule.** Every run that gets compared must sit on the same card — TF32 and cuDNN
kernels differ between L4 and A100, and that difference would land inside `Delta_model` where
nobody would see it. `assert_same_gpu` in notebook 02 refuses to report a gate computed across
mixed hardware. The U-Net grid belongs on L4, the ViT-L grids on A100, and the two are never
compared to each other directly.

---

## 5. Order of work

Nothing past the gate is built until the gate is decided.

```
  notebook 01   localiser                    ->  frozen crops
  notebook 02   noise floor + 2 extremes     ->  THE GATE
                        |
         +--------------+--------------+
         |                             |
       PASS                          FAIL
         |                             |
  21 more U-Net runs            write the negative result:
  RETFound + DINOv3             conventions differ sharply in
  zero-shot                     annotations but are washed out
  the shipped app               by training. Change nothing else.
```

If the gate fails, RETFound and DINOv3 are **not** run. There would be nothing for them to
inherit, and spending 10 A100-hours to confirm an absence is how compute budgets die.

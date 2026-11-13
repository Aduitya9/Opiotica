# Running on Colab

The notebooks are live in your Drive at **My Drive → whose-eyes**, and each link below opens
straight in Colab. Verified on 13 November 2026: all three load and render.

| # | Notebook | Runtime | Time | Link |
|---|---|---|---|---|
| 00 | Inventory + the two frozen decisions | **CPU** | ~20 min | <https://colab.research.google.com/drive/14QVR0Ee72w4zlojgyCAof4_WAZhqHuAa> |
| 01 | Frozen localiser + crop export | **L4** | ~30–45 min | <https://colab.research.google.com/drive/1GfGN_NKksEhmRM5ns_YzVA2COOkLgXjV> |
| 02 | Seed-noise floor + **the gate** | **L4** or A100 | 2.5–3.5 h (L4) | <https://colab.research.google.com/drive/1gA0m9DFqfdU2H2lc6nBzDdwngP_VOl_m> |

Drive folder: <https://drive.google.com/drive/folders/1CZrWyVXuQ997OrrT3G-vCzM97PEeEoQr>

Run them **in order**. Each one reads what the previous one wrote.

## Set the runtime by hand

Colab does not reliably apply the accelerator saved in a notebook's metadata, so set it
yourself each time: **Runtime → Change runtime type**.

* **Notebook 00 → CPU.** It trains nothing. Running it on a GPU just burns compute units.
* **Notebook 01 → L4.** One short training run; an A100 buys nothing because most of the time
  goes on writing crops to Drive.
* **Notebook 02 → L4 or A100.** An A100 is roughly 2× faster and several times the units.

## The one hardware rule

**All 11 runs in notebook 02 must be on the same card.** TF32 and cuDNN kernels differ between
L4 and A100, so a noise floor measured on one and extremes measured on the other would put a
hardware difference inside `Delta_model`. The notebook records the GPU with every run and
`assert_same_gpu` refuses to report a gate computed across mixed hardware. If you hit that
error, delete `whose-eyes-work/runs/` and `whose-eyes-work/results/gate_runs.csv` and redo all
eleven on one card.

Batch size is set from the detected VRAM, not guessed: 4 on a T4, 8 on an L4, 16 on a 40 GB
A100, 24 on an 80 GB A100. Nothing else in the schedule varies.

## Where things live, and why

```
/content/drive/MyDrive/whose-eyes/           the code      (what you open)
/content/drive/MyDrive/whose-eyes-work/      crops, weights, results   (created on first run)
/content/raw/                                the 1.5 GB dataset        (LOCAL, ephemeral)
```

The dataset goes to **local disk**, not Drive: Drive I/O is slow enough to dominate the
runtime, and re-fetching from Hugging Face costs about three minutes. Everything you would
hate to lose goes to Drive, so a Colab disconnect costs at most one training run.

## If Colab drops you

Every long step resumes:

* notebook 00 checkpoints the mask scan every 200 eyes
* notebook 01 skips the localiser if `runs/localiser/best.pt` exists, and skips crops already
  exported
* notebook 02 skips any run already in `results/gate_runs.csv`

Re-run the notebook from the top. It picks up where it stopped.

## Keeping Drive and the repo in step

The Drive copy is what Colab executes; the repo is the source of truth. After changing
anything in `src/`, re-upload that file to **My Drive → whose-eyes → src**, or push to GitHub
and set `REPO_URL` at the top of each notebook, which clones a fresh copy every session and
removes the whole problem.

Never edit the `.ipynb` files directly — they are generated. Edit `tools/build_notebooks.py`,
run it, and re-upload.

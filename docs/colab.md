# Running on Colab

The notebooks open straight from the public GitHub repo, so the link always serves the current
version and there is no stale copy to keep in step.

| # | Notebook | Runtime | Time | Open in Colab |
|---|---|---|---|---|
| 00 | Inventory + the two frozen decisions | **CPU** | ~20 min | <https://colab.research.google.com/github/Aduitya9/Opiotica/blob/main/notebooks/00_inventory.ipynb> |
| 01 | Frozen localiser + crop export | **L4** | ~45 min | <https://colab.research.google.com/github/Aduitya9/Opiotica/blob/main/notebooks/01_frozen_localiser.ipynb> |
| 02 | Seed-noise floor + **the gate** | **L4** or A100 | 2.5–3.5 h | <https://colab.research.google.com/github/Aduitya9/Opiotica/blob/main/notebooks/02_noise_floor_gate.ipynb> |

Run them **in order** — each reads what the previous one wrote.

## Every notebook checks itself before doing slow work

Each notebook runs in **its own virtual machine**. `/content` is wiped between them; only
Drive and the Hub survive. That caught us once: notebook 01 assumed notebook 00's
`/content/raw` was still there and died with `IndexError: list index out of range` ten lines
in, after pip had already spent two minutes building wheels.

So notebooks 01 and 02 now open with a **preflight cell** that, before anything slow:

* stops if the runtime has no GPU, naming the menu path, rather than failing at the first
  `.cuda()` call twenty minutes later
* checks every file it needs and, for each missing one, says **which notebook produces it**
* fetches and extracts the release itself if this VM does not have it (about 4 minutes)

If a preflight raises, nothing has been wasted and the message says what to do.
`tests/test_notebooks.py` enforces all of this statically, so a notebook that reintroduces the
problem fails the test suite before it reaches Colab.

A GitHub-opened notebook is read-only. Colab will offer **File → Save a copy in Drive** the
moment you edit or run it; take the offer, and your outputs are saved in your own copy while
the link above keeps serving the canonical version.

## Set the runtime by hand

Colab does not reliably apply the accelerator saved in notebook metadata, so set it each time:
**Runtime → Change runtime type**.

* **00 → CPU.** It trains nothing; a GPU just burns compute units.
* **01 → L4.** One short training run; most of the time goes on writing crops, so an A100
  buys nothing.
* **02 → L4 or A100.** An A100 is roughly 2× faster and several times the units.

## The one hardware rule

**All 11 runs in notebook 02 must be on the same card.** TF32 and cuDNN kernels differ between
L4 and A100, so a noise floor measured on one and extremes on the other would put a hardware
difference inside `Delta_model`. Every run records its GPU and `assert_same_gpu` refuses to
report a gate computed across mixed hardware. If you hit that error, delete
`whose-eyes-work/runs/` and `whose-eyes-work/results/gate_runs.csv` and redo all eleven on one
card.

Batch size is read from the detected VRAM, not guessed: 4 on a T4, 8 on an L4, 16 on a 40 GB
A100, 24 on an 80 GB A100. Nothing else in the schedule varies.

## One secret to add, once

Add your Hugging Face write token to **Colab → Secrets** (the key icon in the left sidebar):

```
name   HF_TOKEN
value  <your hf_... token>
```

Then switch on **Notebook access** for each notebook. With it set, notebooks 01 and 02 archive
their outputs to the private dataset repo. Without it they still run end to end — they just
print `upload DISABLED` and keep everything in Drive.

Never paste the token into a cell. `userdata.get("HF_TOKEN")` is the only way it is read.

## Where things live, and why

```
github.com/Aduitya9/Opiotica                 the code      (cloned to /content/project)
/content/raw/                                the 1.5 GB dataset   (LOCAL, ephemeral)
/content/drive/MyDrive/whose-eyes-work/      crops, weights, results  (Drive, persistent)
huggingface.co/datasets/Aduitya9/            the archival copy    (PRIVATE)
  opiotica-whose-eyes
```

The dataset goes to **local disk**, not Drive: Drive I/O is slow enough to dominate the
runtime, and re-fetching from Hugging Face costs about three minutes. Anything you would hate
to lose goes to Drive, so a disconnect costs at most one training run.
[storage.md](storage.md) explains why the crops are private and the measurements are public.

## If Colab drops you

Every long step resumes — re-run from the top and it picks up where it stopped:

* **00** checkpoints the mask scan every 200 eyes
* **01** skips the localiser if `runs/localiser/best.pt` exists, and skips crops already written
* **02** skips any run already in `results/gate_runs.csv`

## Changing a notebook

Never edit the `.ipynb` directly — they are generated. Edit `tools/build_notebooks.py`, run
`python tools/build_notebooks.py`, and push. The Colab links then serve the new version
immediately.

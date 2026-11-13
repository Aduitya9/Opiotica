# Where each artefact lives, and why

Two homes, split on one line: **does it contain image pixels?**

```
+========================================================================+
|  GITHUB  (public)          github.com/Aduitya9/Opiotica                |
|    code, docs, pre-registration, DERIVED NUMBERS                       |
|    - no imagery, no weights, no large files                            |
|    - public, because the paper's reproducibility claim depends on it   |
+------------------------------------------------------------------------+
|  HUGGING FACE  (PRIVATE)   Aduitya9/opiotica-whose-eyes                |
|    frozen crops (.npz), model weights, run outputs                     |
|    - PRIVATE, because the crops contain REFUGE image pixels            |
|    - HF handles multi-GB files through LFS; GitHub does not            |
+========================================================================+
```

## Why the crops cannot go in a public repo

A crop is still a REFUGE photograph — smaller and resized, but the same pixels of the same
patient's eye. Publishing 1,200 of them is redistributing the dataset, which the REFUGE
challenge's terms do not grant. The Hugging Face mirror's card says apache-2.0, but that is the
mirror uploader's label, not the organisers', and it does not transfer the right to re-publish.

Derived *numbers* are different. A vertical cup-to-disc ratio is a measurement, not a copy of
the image, and publishing measurements is both standard practice and necessary for anyone to
check the paper's arithmetic. So the measurements are public and the pixels are not.

## Why not GitHub for the data

* GitHub rejects any file over 100 MB and warns past 50 MB. One `.npz` is fine; 1,200 are not.
* A repository is soft-capped around 5 GB. The 56 model checkpoints alone are about 4 GB.
* Git stores full history, so re-exporting crops once would double the repository's size
  permanently.

Hugging Face is built for exactly this: large binary files over LFS, with a private flag.

## Why not Drive for everything

Drive is where Colab writes while it works, and that stays. But Drive is not citable, not
versioned and not something a reviewer can be pointed at. The Hugging Face repo is the archival
copy; Drive is the scratch space.

## The rule for anything new

| Artefact | Home | Public? |
|---|---|---|
| `src/`, `tools/`, `tests/`, notebooks | GitHub | yes |
| `PREREGISTRATION.md`, `docs/` | GitHub | yes |
| `analysis/*.csv` — derived measurements | GitHub | yes |
| `crops/*.npz` — contains REFUGE pixels | Hugging Face | **no** |
| `runs/*/best.pt` — model weights | Hugging Face | **no** |
| `results/*.csv`, `gate.json` — derived | both | yes on GitHub |
| Anything RIGA-derived | neither, until the NC clause is checked | **no** |

When the paper is submitted, the Hugging Face repo can be opened to reviewers individually
without making it public.

## Tokens

The write token lives in **Colab → Secrets** as `HF_TOKEN`, and in nothing else. It is not in
this repository, not in a notebook cell, and not in any committed file. `.gitignore` blocks
`*token*.txt` so the local credentials file can never be committed by accident.

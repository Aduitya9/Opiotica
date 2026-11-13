# Derived measurements — provenance and attribution

Everything in this folder is **derived numbers**: vertical cup-to-disc ratios, bounding-box
coordinates and referral decisions computed from publicly released expert annotations. **No
image pixels are redistributed here.** That distinction is deliberate — see
[../docs/storage.md](../docs/storage.md).

| File | Derived from | Licence of the source | What it holds |
|---|---|---|---|
| `r1_labels.csv` | REFUGE / REFUGE-MultiRater | REFUGE challenge terms | the clinical diagnosis per eye, 1,195 clean of 1,200 |
| `r5_fusion_vcdr.csv` | REFUGE-MultiRater | REFUGE challenge terms | vCDR under six fusion rules |
| `r6_official_merge.csv` | REFUGE-MultiRater | REFUGE challenge terms | per-expert vCDR + the official merge |
| `chaksu_decisions.csv` | **Chákṣu** | **CC BY 4.0** | five experts' glaucoma-suspect decisions |

## Required attribution

**Chákṣu** is CC BY 4.0, which requires attribution wherever it is reused:

> Kumar, J. R. H., Seelamantula, C. S., Gagan, J. H., Kamath, Y. S., Kuzhuppilly, N. I. R.,
> Vivekanand, U., Gupta, P., & Patil, S. *Chákṣu: A glaucoma specific fundus image database.*
> Scientific Data (2023). Dataset: <https://doi.org/10.6084/m9.figshare.20123135>

**REFUGE** and **REFUGE-MultiRater** are used under the REFUGE challenge's own terms
(<https://refuge.grand-challenge.org/>). The Hugging Face mirror's card states apache-2.0, but
the original challenge terms govern; this project treats REFUGE **imagery** as
non-redistributable and publishes only measurements derived from it.

**RIGA** is **CC BY-NC**. It appears in the paper; nothing RIGA-derived goes into any released
artefact, or that artefact inherits the non-commercial clause.

## Reproducing these files

Each `rN_*.py` regenerates its own CSV from the released annotations:

```bash
python -I analysis/r6_official_merge.py <path to the extracted REFUGE-Multirater folder>
```

`tests/test_vcdr.py` re-derives every number the pre-registration quotes from
`r6_official_merge.csv`, so a drift in any of them fails the test suite rather than silently
changing the paper.

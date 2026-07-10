# Data

This directory holds the datasets used to evaluate EntroPath. **Its contents are
not committed to the repository** — only this file is tracked. Synthetic data is
regenerated at runtime; the single-cell datasets are downloaded once from the
DTNE data share (link below).

## Synthetic benchmarks — no download needed

The synthetic benchmarks from the paper are regenerated deterministically from a
fixed seed, so nothing needs to be stored here. The generator functions live in
the `utils/` directory. With `random_state=42` the output should be bit-for-bit
reproducible.

## Single-cell datasets — download from the DTNE data share

The biological datasets (Paul15, Nestorowa, Embryoid Body, Pancreas, Lymphoid,
Root Atlas) are those released by the authors of DTNE. We do **not** redistribute
them; download them from:

https://drive.google.com/drive/folders/1UFKBWFJ7BhzcABpa4DZXssthQuTGMfmU

Place the files in this folder so the notebooks resolve them (adjust the names
below to match the share):

```
data/
├── README.md          # this file (tracked)
├── paul15/            # e.g. paul15.h5ad
├── nestorowa/
├── pancreas/
├── lymphoid/
├── embryoid_body/
└── root_atlas/
```

Everything here except `README.md` is git-ignored, so downloaded files stay local
and are never committed.
# Ablation experiments

This folder holds the ablation studies for the EntroPath paper. It maps each
ablation to the code that produces it, where its outputs land, and which paper
figure/table it feeds. Some ablations are self-contained scripts; others derive
from re-running the standard synthetic benchmark notebooks with a config toggle —
this README is the single place that records which is which.

> Adjust the notebook filenames / paths below to match your repo; placeholders are
> marked `<...>`.

## Conventions

- **Name by content, not by A-number.** Files use descriptive names
  (`_landmarks`, `_kernel`) because the paper's A1/A2/… numbering shifts between
  revisions.
- **Fixed depth.** Every script assumes it sits two directories under the repo
  root and resolves `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`.
  Keep all scripts at `experiments/ablation/<file>` so this holds.
- **Outputs.** Caches go to `results/<name>/`, figures to `figures/<name>/`.
- **Caching.** Self-contained scripts cache to CSV and re-plot from cache
  (`FORCE_RERUN=False`); set `QUICK_TEST=False` for the real run.
- **Shared recipe.** EntroPath runs with `k_neighbors=15`, `kernel="gaussian"`
  (synthetic), full-rank unless landmarks are the thing under test.

## Index

### Diffusion depth `k`  — MAIN TEXT
- **Code:** `ablation_A1_diffusion_depth_entropath.ipynb` (notebook — the one
  non-script ablation).
- **Run:** set `QUICK_TEST=False`, execute all cells.
- **Outputs:** `results/ablation_k/{ablation_k.csv, ablation_k_largek.csv}`;
  `figures/ablation_k/{ablation_k_main, ablation_k_largek}.{pdf,png}`.
- **Paper:** main-text ablation paragraph; `fig:ablation-k` (fidelity + VNE knee,
  and the large-`k` / `prop:largek` panel).

### Landmark approximation — APPENDIX
- **Code:** `run_ablation_landmarks.py` (self-contained: sweep → CSV → figures).
- **Run:** `QUICK_TEST=False`, then `python experiments/ablation/run_ablation_landmarks.py`.
- **Outputs:** `results/ablation_landmarks/ablation_landmarks.csv`;
  `figures/ablation_landmarks/ablation_landmarks_{quality,trust,runtime,examples}.{pdf,png}`.
- **Paper:** appendix subsubsection `app:ablation-lm`; `fig:ablation-lm`
  (quality + runtime). The standalone `trust` and `examples` figures are produced
  but not shown in the paper.
- **Qualitative companion (MAIN TEXT):** `swiss_roll_landmarks.py` →
  `swiss_roll_fps.png`, the 50k embedding used as `fig:scalability`.

### Sensitivity to `k_NN` — APPENDIX
- **Plot code:** `plot_knn_ablation.py` (reads pre-computed level-2 JSONs).
- **Data provenance:** *no dedicated sweep script.* The JSONs come from re-running
  the synthetic benchmark notebooks at `k_NN ∈ {5,10,15,20}` with
  `GEODESIC_TYPE="analytic"`:
  - `<swiss_roll notebook>` → stem `swiss_roll_uniform`
  - `<non_uniform_swiss_roll notebook>` → stem `non_uniform_swiss_roll_nonuniform`
  - JSONs land in `results/knn_ablation/{stem}_{geodesic}_level2_knn{k}.json`.
- **Run plot:** `python experiments/ablation/plot_knn_ablation.py --geodesic analytic --metric spearman_row`.
- **Paper:** appendix subsubsection `app:ablation-knn` + figure.
- **Note:** always `--geodesic analytic` — the shortest-path ground truth is
  circular for Isomap / Shortest Path (they score ~1.0 by construction).

### Kernel choice (Gaussian vs. α-decay) — APPENDIX
- **No dedicated script.** The three-row table (non-uniform Swiss roll, sphere,
  dense tree; **shortest-path** protocol, `k_NN=15`, 30 seeds, `spearman_row` +
  trustworthiness) is the EntroPath row from the standard synthetic benchmark
  notebooks, run **twice** — once with `kernel="gaussian"`, once with
  `kernel="alpha_decay"` (`decay=40`).
- **Source files (EntroPath row of each):**
  - `results/<dataset>/shortest_path/<dataset>_shortest_path_level2.json`
    for `non_uniform_swiss_roll_nonuniform`, `sphere`, `dense_tree`.
  - The Gaussian and α-decay columns are both verified to match these JSONs to
    3 d.p. (shortest-path, 30 seeds).
- **Protocol note:** this table uses **shortest-path** (consistent with the main
  synthetic tables). Unlike the kNN ablation it does *not* use analytic, because
  it contains only EntroPath rows — the Isomap/Shortest-Path circularity that
  forces analytic elsewhere does not apply here.
- **Kernel toggle (reproducibility gap):** there is currently no `kernel`
  parameter in the notebooks — the kernel is changed by editing the repo default,
  which is undocumented. Fix: add an explicit `KERNEL = "gaussian"` at the top of
  each synthetic notebook, passed into the EntroPath constructor, and regenerate
  the α-decay rows with `KERNEL="alpha_decay"`. Keep **one** set of notebooks; do
  not maintain parallel Gaussian/α-decay copies (the saved JSONs are the record).
- **Paper:** appendix subsubsection `app:ablation-kernel`; `tab:ablation-kernel`
  (LaTeX in `<paper>/ablation_A5_A6.tex`).
- **Optional upgrade:** a `run_ablation_kernel.py` (EntroPath only, both kernels
  set explicitly, three datasets, shortest-path, 30 seeds) would make this
  one-command and re-verify the Gaussian column. Not an arXiv blocker — the
  numbers are already saved and verified.

## Reproducibility status

| Ablation | One-command repro | Notes |
|---|---|---|
| Diffusion depth `k` | yes (notebook) | run all cells |
| Landmarks | yes (`run_ablation_landmarks.py`) | cached CSV |
| `k_NN` sensitivity | partial | notebook re-run → JSON, then `plot_knn_ablation.py` |
| Kernel | no | shortest-path; numbers saved + verified; toggle currently a repo edit — make `KERNEL` explicit; optional `run_ablation_kernel.py` |

## Not in this folder

- `fig:bottleneck` is a conceptual TikZ figure in the paper's method section
  (illustrating the MERW bottleneck effect, `rem:bottleneck`), not generated by
  any ablation code here.

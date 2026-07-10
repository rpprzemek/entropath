#!/usr/bin/env python
"""
Noise-robustness ablation for EntroPath.

Tests the Related-Work claim that EntroPath's path-ensemble (log-sum-exp over many
paths) smooths shortcut noise while targeting the same geodesic geometry -- i.e.
that it degrades more gracefully under input noise than single-path methods
(Isomap, Shortest Path).

Design (uniform Swiss roll, ANALYTIC geodesic):
  * The analytic geodesic is computed from the true (t, h) parameters, so it is
    completely immune to the input noise. Any score drop as sigma rises is therefore
    PURELY method degradation -- not ground-truth degradation. A shortest-path
    ground truth would confound the two. Uniform (not non-uniform) isolates the
    noise axis from density heterogeneity.
  * PRIMARY metric = Level-1 (distance matrix) row-wise Spearman vs the analytic
    geodesic. This tests the distance representation directly, without the MDS step.
    Level-2 (embedding) is recorded as a secondary confirmation.
  * Comparison set = the distance-based methods: Euclidean floor, Isomap /
    Shortest Path (single-path, expected to fail fast), and the diffusion family
    incl. EntroPath (path-ensemble, expected to hold).

PILOT FIRST. Set QUICK_TEST=True to run sigma in {0.05, 0.5} on a few seeds and
confirm EntroPath drops LESS than Isomap before committing to the full
6 x 30 x 8 grid. If the effect is absent or reversed, the ablation contradicts the
Related-Work sentence and you want to know cheaply (same logic as the Beta decile check).

Cannot run in the prep sandbox (needs entropath/utils); written against the notebook's
own API (get_methods / extract_method_distances / level1_metrics / level2_metrics).
"""

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # experiments/ablation/<file>
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from utils.benchmark_utils import (
    get_methods, level1_metrics, level2_metrics, LEVEL1_METHODS, demap,
)
from utils.swiss_roll_utils import make_swiss_roll_with_analytic_geodesic
from utils.helpers import extract_method_distances

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

# ============================== CONFIG ==============================
N            = 2000
HEIGHT_SCALE = 3.0
K_NN         = 15
K_TRUST      = 50
K_DEMAP      = 30

# Noise sweep, calibrated to the manifold scale (arc length ~10 units; 1.0 ~ 10%).
# 0 = ceiling, 0.05 = the standard, up to 1.0 = severe (do NOT go past -- structure
# is destroyed for everyone and the comparison loses signal).
NOISE_GRID   = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS        = list(range(42, 72))          # 30 seeds; same seeds across sigma (paired)

# Distance-based comparison set (headline). Euclidean is added separately as the floor.
# Set to None to run every method get_methods() returns (adds embedding-only methods,
# ~1.5x cost, no Level-1 rows for them).
METHODS_SUBSET = ["Isomap", "Shortest Path", "Diffusion Maps",
                  "PHATE", "HeatGeo", "DTNE", "EntroPath"]

FORCE_RERUN  = True
QUICK_TEST   = False                          # pilot: False for the full grid
if QUICK_TEST:
    NOISE_GRID = [0.05, 0.5]
    SEEDS      = list(range(42, 45))         # 3 seeds

FIG_DIR = PROJECT_ROOT / "figures" / "ablation_noise"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = PROJECT_ROOT / "results" / "ablation_noise"; RES_DIR.mkdir(parents=True, exist_ok=True)
CSV     = RES_DIR / "ablation_noise.csv"

# EntroPath emphasised; Euclidean is the noise floor (grey).
STYLE = {
    "Euclidean":      dict(color="0.6",     marker="x", ls=":"),
    "Isomap":         dict(color="#8c564b", marker="v"),
    "Shortest Path":  dict(color="#e377c2", marker="P"),
    "Diffusion Maps": dict(color="#9467bd", marker="o"),
    "PHATE":          dict(color="#2ca02c", marker="s"),
    "HeatGeo":        dict(color="#ff7f0e", marker="^"),
    "DTNE":           dict(color="#1f77b4", marker="D"),
    "EntroPath":      dict(color="#d62728", marker="*", lw=2.8, markersize=12, zorder=5),
}


# ============================== SWEEP ==============================
def _run_method(name, method, X, X_clean, D_geo):
    """Return (l1_spearman | nan, l2_spearman | nan) for one fitted method."""
    l1 = np.nan
    emb = method.fit_transform(X)
    l2 = level2_metrics(X, emb, D_geo, k=K_TRUST)["spearman_row"]
    D_method = extract_method_distances(method, name)
    if D_method is not None and name in LEVEL1_METHODS:
        l1 = level1_metrics(D_method, D_geo)["spearman_row"]
    return l1, l2


def run_sweep():
    rows = []
    for noise in NOISE_GRID:
        print(f"\n=== noise sigma = {noise} ===")
        for seed in SEEDS:
            X, t, D_geo, X_clean = make_swiss_roll_with_analytic_geodesic(
                n_samples=N, noise=noise, random_state=seed, height_scale=HEIGHT_SCALE)

            # Euclidean Level-1 floor (raw distances, no denoising).
            l1_eucl = level1_metrics(squareform(pdist(X)), D_geo)["spearman_row"]
            rows.append(dict(noise=noise, seed=seed, method="Euclidean",
                             l1_spearman=l1_eucl, l2_spearman=np.nan))

            methods = get_methods(seed=seed, k_nn=K_NN)
            names = METHODS_SUBSET if METHODS_SUBSET else list(methods)
            for name in names:
                t0 = time.perf_counter()
                try:
                    l1, l2 = _run_method(name, methods[name], X, X_clean, D_geo)
                except Exception as e:
                    print(f"    {name} FAILED: {e}")
                    l1, l2 = np.nan, np.nan
                rows.append(dict(noise=noise, seed=seed, method=name,
                                 l1_spearman=l1, l2_spearman=l2))
                print(f"  sigma={noise:<4} seed={seed} {name:14s} "
                      f"L1={l1:.3f} L2={l2:.3f}  {time.perf_counter()-t0:5.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"\nsaved {CSV}")
    return df


# ============================== PLOTS ==============================
def _order(df, col, exclude=()):
    """Methods with data for `col`, minus `exclude`; EntroPath drawn last (on top)."""
    present = [m for m in df.method.unique()
               if m not in exclude and df[df.method == m][col].notna().any()]
    return sorted(present, key=lambda m: (m == "EntroPath", m))

def plot_degradation(df, col="l1_spearman", tag="l1", ylabel=None, exclude=(), labels=None):
    ylabel = ylabel or ("distance-level Spearman vs analytic geodesic"
                        if tag == "l1" else "embedding Spearman vs analytic geodesic")
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name in _order(df, col, exclude):          # <-- pass exclude through
        sub = df[df.method == name].groupby("noise")[col]
        m, s = sub.mean().sort_index(), sub.std().sort_index()
        st = {"lw": 1.6, "markersize": 6, **STYLE.get(name, {})}
        ax.plot(m.index, m.values, label=labels.get(name, name), **st)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values,
                        alpha=0.12, color=st.get("color"))
    ax.axvline(0.05, color="0.7", ls=":", lw=1)
    ax.text(0.05, ax.get_ylim()[0], " standard", color="0.5", fontsize=8, va="bottom")
    ax.set_xlabel(r"input noise $\sigma$"); ax.set_ylabel(ylabel)
    ax.set_title(f"Noise robustness (uniform Swiss roll, $N={N}$)")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ablation_noise_{tag}.{ext}", bbox_inches="tight", dpi=400)
    plt.show()


def pilot_check(df):
    """Cheap insurance: does EntroPath drop LESS than Isomap across the sweep?"""
    lo, hi = min(df.noise), max(df.noise)
    print(f"\n=== pilot direction check (L1 Spearman, sigma {lo} -> {hi}) ===")
    for name in ["EntroPath", "Isomap", "Shortest Path", "Euclidean"]:
        sub = df[df.method == name].groupby("noise")["l1_spearman"].mean()
        if lo in sub and hi in sub:
            drop = sub[lo] - sub[hi]
            print(f"  {name:14s}: {sub[lo]:.3f} -> {sub[hi]:.3f}   drop {drop:+.3f}")
    ep = df[df.method == "EntroPath"].groupby("noise")["l1_spearman"].mean()
    iso = df[df.method == "Isomap"].groupby("noise")["l1_spearman"].mean()
    if lo in ep and hi in ep and lo in iso and hi in iso:
        verdict = (ep[lo] - ep[hi]) < (iso[lo] - iso[hi])
        print(f"  --> EntroPath degrades {'LESS' if verdict else 'MORE'} than Isomap "
              f"{'(hypothesis supported)' if verdict else '(HYPOTHESIS NOT SUPPORTED)'}")


# ============================== MAIN ==============================
if __name__ == "__main__":
    if CSV.exists() and not FORCE_RERUN:
        print(f"loading cached {CSV} (set FORCE_RERUN=True to re-sweep)")
        df = pd.read_csv(CSV)
    else:
        df = run_sweep()
    pilot_check(df)
    plot_degradation(df, col="l1_spearman", tag="l1",
                 exclude=["Diffusion Maps", "Shortest Path"],
                 labels={"Isomap": "Isomap / Shortest Path"})
    plot_degradation(df, col="l2_spearman", tag="l2")   # secondary
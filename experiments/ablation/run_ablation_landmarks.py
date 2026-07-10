#!/usr/bin/env python
"""
A4 -- Landmark / Nystrom ablation for EntroPath.

Sweeps the landmark count M and the selection method (fps / kmeans / random) on the
Swiss roll, measuring (i) embedding quality vs the analytic geodesic and (ii) wall-clock
time, against the full-rank EntroPath ceiling. Caches to CSV; regenerates figures from
cache so plotting can be re-run without re-sweeping.

IMPORTANT: the grid must span the M=2000 design point. The landmark requirement is set
by manifold complexity, not N -- the swiss roll spiral needs ~2000 landmarks to resolve,
so a grid that stops below 2000 only shows the rising part of the curve and understates
the method. The default grid below brackets the design point (250..4000).

  python run_ablation_landmarks.py     # QUICK_TEST first, then set QUICK_TEST=False

Cannot be executed in the prep sandbox (needs entropath/utils); written against the
real APIs. Quality uses utils.benchmark_utils.level2_metrics, evaluated on a fixed
point subsample per seed for cost control.
"""

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # adjust to your repo layout
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from entropath import EntroPath
except ImportError:
    from entropath.embedding import EntroPath
from utils.benchmark_utils import level2_metrics
from utils.swiss_roll_utils import make_swiss_roll_with_analytic_geodesic

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

# ============================== CONFIG ==============================
N            = 5000                    # full-rank ceiling stays tractable; M=2000 design point is in-range
NOISE        = 0.05
HEIGHT_SCALE = 3.0
K_NN         = 15
KERNEL       = "gaussian"              # match get_methods' EntroPath
SEEDS        = list(range(42, 52))     # 10 seeds for an appendix ablation; bump for camera-ready

# Grid MUST bracket the M=2000 design point so the plateau is visible.
M_GRID       = [250, 500, 1000, 2000, 3000, 4000]
METHODS      = ["fps", "kmeans", "random"]    # random = floor; fps vs kmeans = the comparison
INCLUDE_FULL_RANK = True               # full-rank EntroPath as the quality/runtime ceiling
EVAL_SUBSAMPLE    = 1500               # points used for the level2 metric (cost control)
EXAMPLE_MS        = [500, 1000, 2000]  # M-progression for the qualitative example figure

FORCE_RERUN  = False                   # if False and CSV exists, skip the sweep and just plot
QUICK_TEST   = False                    # True: 2 seeds, M in {500,2000}. Set False for the real run.
if QUICK_TEST:
    SEEDS   = SEEDS[:2]
    M_GRID  = [500, 2000]              # include a near-design-point M even in the smoke test

FIG_DIR = PROJECT_ROOT / "figures" / "ablation_landmarks"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = PROJECT_ROOT / "results" / "ablation_landmarks"; RES_DIR.mkdir(parents=True, exist_ok=True)
CSV     = RES_DIR / "ablation_landmarks.csv"


# ============================== METRIC ==============================
def quality(X, emb, D_geo, idx):
    """Your Level-2 metric on a fixed subsample; returns (spearman_row, trust)."""
    m = level2_metrics(X[idx], emb[idx], D_geo[np.ix_(idx, idx)])
    return m["spearman_row"], m["trust"]


# ============================== MODEL ==============================
def make_model(seed, use_landmarks, n_landmarks=None, method=None):
    kw = dict(k_neighbors=K_NN, kernel=KERNEL, t_power="auto",
              n_pca=None, random_state=seed, verbose=0)
    if use_landmarks:
        return EntroPath(use_landmarks=True, n_landmarks=n_landmarks,
                         landmarks_method=method, mds_solver="smacof", **kw)
    return EntroPath(use_landmarks=False, **kw)

def fit_timed(model, X):
    t0 = time.perf_counter()
    Y = model.fit_transform(X)
    return Y, time.perf_counter() - t0


# ============================== SWEEP ==============================
def run_sweep():
    rows = []
    for seed in SEEDS:
        X, t, D_geo, X_clean = make_swiss_roll_with_analytic_geodesic(
            n_samples=N, noise=NOISE, random_state=seed, height_scale=HEIGHT_SCALE)
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=min(EVAL_SUBSAMPLE, N), replace=False)

        if INCLUDE_FULL_RANK:
            Y, secs = fit_timed(make_model(seed, use_landmarks=False), X)
            q, tr = quality(X, Y, D_geo, idx)
            rows.append(dict(seed=seed, method="full", M=N, quality=q, trust=tr, seconds=secs))
            print(f"  seed {seed} | full-rank        : rho={q:.3f} trust={tr:.3f}  {secs:6.1f}s")

        for method in METHODS:
            for M in M_GRID:
                Y, secs = fit_timed(
                    make_model(seed, use_landmarks=True, n_landmarks=M, method=method), X)
                q, tr = quality(X, Y, D_geo, idx)
                rows.append(dict(seed=seed, method=method, M=M, quality=q, trust=tr, seconds=secs))
                print(f"  seed {seed} | {method:6s} M={M:<5d}: rho={q:.3f} trust={tr:.3f}  {secs:6.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"\nsaved {CSV}")
    return df


# ============================== PLOTS ==============================
def _agg(df, method, col):
    sub = df[df.method == method].groupby("M")[col]
    return sub.mean().sort_index(), sub.std().sort_index()

def plot_quality(df, col="quality", ylabel=r"embedding $\rho$ vs. analytic geodesic", tag="quality"):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for j, method in enumerate(METHODS):
        m, s = _agg(df, method, col)
        ax.plot(m.index, m.values, marker="o", lw=2, color=f"C{j}", label=method)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values, alpha=0.15, color=f"C{j}")
    if (df.method == "full").any():
        full = df[df.method == "full"][col]
        ax.axhline(full.mean(), color="0.3", ls="--", lw=1.6, label=f"full-rank = {full.mean():.3f}")
    ax.axvline(2000, color="0.6", ls=":", lw=1.2)
    ax.text(2000, ax.get_ylim()[0], " design point", color="0.4", fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("number of landmarks $M$ (log)")
    ax.set_ylabel(ylabel); ax.set_title(f"Landmark count vs. {tag} (Swiss roll, $N={N}$)")
    ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ablation_landmarks_{tag}.{ext}", bbox_inches="tight", dpi=400)
    plt.show()

def plot_runtime(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for j, method in enumerate(METHODS):
        m, _ = _agg(df, method, "seconds")
        ax.plot(m.index, m.values, marker="s", lw=2, color=f"C{j}", label=method)
    if (df.method == "full").any():
        full = df[df.method == "full"]["seconds"]
        ax.axhline(full.mean(), color="0.3", ls="--", lw=1.6, label=f"full-rank = {full.mean():.1f}s")
    ax.axvline(2000, color="0.6", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of landmarks $M$ (log)"); ax.set_ylabel("wall-clock time (s, log)")
    ax.set_title(f"Landmark count vs. runtime (Swiss roll, $N={N}$)")
    ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ablation_landmarks_runtime.{ext}", bbox_inches="tight", dpi=400)
    plt.show()

def plot_examples(seed=None, method="fps"):
    """Qualitative M-progression at one seed: full-rank vs fps at increasing M, coloured by t.
    Shows the embedding sharpening toward full-rank as M approaches the design point."""
    seed = SEEDS[0] if seed is None else seed
    X, t, D_geo, X_clean = make_swiss_roll_with_analytic_geodesic(
        n_samples=N, noise=NOISE, random_state=seed, height_scale=HEIGHT_SCALE)
    configs = [("full-rank", make_model(seed, use_landmarks=False))]
    for M in EXAMPLE_MS:
        configs.append((f"{method}  (M={M})",
                        make_model(seed, use_landmarks=True, n_landmarks=M, method=method)))
    fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 4.2))
    for ax, (name, model) in zip(np.atleast_1d(axes), configs):
        Y = model.fit_transform(X)
        ax.scatter(Y[:, 0], Y[:, 1], c=t, s=3, alpha=0.6, cmap="Spectral")
        ax.set_title(name, fontweight="bold"); ax.set_axis_off()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"ablation_landmarks_examples.{ext}", bbox_inches="tight", dpi=300)
    plt.show()


def summary(df):
    print("\n=== A4 summary ===")
    if (df.method == "full").any():
        full = df[df.method == "full"]
        print(f"full-rank: rho={full['quality'].mean():.3f} trust={full['trust'].mean():.3f} "
              f"{full['seconds'].mean():.1f}s")
    for method in METHODS:
        sub = df[df.method == method]
        for M in sorted(sub.M.unique()):
            r = sub[sub.M == M]
            print(f"  {method:6s} M={M:<5d}: rho={r['quality'].mean():.3f}+/-{r['quality'].std():.3f} "
                  f"trust={r['trust'].mean():.3f}  {r['seconds'].mean():.1f}s")


# ============================== MAIN ==============================
if __name__ == "__main__":
    if CSV.exists() and not FORCE_RERUN:
        print(f"loading cached {CSV} (set FORCE_RERUN=True to re-sweep)")
        df = pd.read_csv(CSV)
    else:
        df = run_sweep()
    summary(df)
    plot_quality(df, col="quality", ylabel=r"embedding $\rho$ vs. analytic geodesic", tag="quality")
    plot_quality(df, col="trust", ylabel="trustworthiness", tag="trust")
    plot_runtime(df)
    plot_examples()

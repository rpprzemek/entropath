"""
bio_utils.py
============

Shared helpers for biological scRNA-seq benchmark notebooks (Pancreas,
Nestorowa, Paul15, Embryoid Body, ...).

Builds on ``benchmark_utils.py`` with three additions specific to single-cell
benchmarks where no clean reference manifold is available:

* DTNE-style subsampled DEMaP (Wei et al., 2025), instead of full-matrix
  DEMaP on the noiseless manifold;
* cluster-purity metrics (ARI, NMI, trustworthiness) on cell-type labels,
  interpreted as proxies for trajectory-preservation quality;
* embedding-row / embedding-grid plots with a shared categorical legend,
  plus pseudotime downstream visualisation (EntroPath, DTNE).

The seven-method comparison set matches the bio section of the EntroPath
paper. For the full twelve-method synthetic benchmark see
``benchmark_utils.get_methods`` directly.
"""

import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from utils.benchmark_utils import demap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from entropath import EntroPath
from heatgeo.embedding import HeatGeo
from dtne import DTNE
import umap
import phate

from utils.helpers import compute_shortest_path_geodesic



# ── Bio-specific configuration ────────────────────────────────────────────────

BIO_PLOT_ORDER = ['PCA', 't-SNE', 'UMAP', 'PHATE',
                  'HeatGeo','DTNE', 'EntroPath']


BIO_PANEL_SIZE = 2.6

# Smaller markers than the synthetic POINT_KWARGS (s=15), since bio embeddings
# are denser and we want cluster structure to read at small panel size.
BIO_POINT_KWARGS = dict(
    s=2,
    alpha=0.85,
    linewidths=0,
    edgecolors='white',
    rasterized=False,
)


# ── Methods factory ───────────────────────────────────────────────────────────

def get_bio_methods(seed, k_nn=15):
    return {
        "PCA":       PCA(n_components=2),
        "t-SNE":     TSNE(perplexity=30, random_state=seed),
        "UMAP":      umap.UMAP(n_neighbors=k_nn, random_state=seed),
        "PHATE":     phate.PHATE(knn=k_nn, random_state=seed, verbose=False),
        "HeatGeo":   HeatGeo(knn=k_nn),
        "DTNE":      DTNE(k_neighbors=k_nn),
        "EntroPath": EntroPath(k_neighbors=k_nn, random_state=seed,
                               kernel="alpha_decay", mds_solver="smacof",
                               landmarks_method="kmeans", verbose=False), # use_landmarks=True
    }


# ── Run loop ──────────────────────────────────────────────────────────────────

def run_methods(methods, X, verbose=True):
    """Fit every method on X, return embeddings and per-method runtimes.

    Methods that raise are recorded with ``runtimes[name] = None`` and
    omitted from the embeddings dict. Prints EntroPath's ``t_power_`` attribute
    if present, matching the convention in the pancreas/nestorowa notebooks.

    Returns
    -------
    embeddings : dict[str, (n, 2) ndarray]
    runtimes   : dict[str, float | None]
    """
    embeddings, runtimes = {}, {}
    for name, method in methods.items():
        if verbose:
            print(f"  {name:18s}", end=" ", flush=True)
        t0 = time.time()
        try:
            emb = method.fit_transform(X)
            runtimes[name] = time.time() - t0
            embeddings[name] = emb
            extra = ""
            if name == "EntroPath" and hasattr(method, "t_power_"):
                extra = f"  (t_power = {method.t_power_})"
            if verbose:
                print(f"done ({runtimes[name]:6.1f}s){extra}")
        except Exception as e:
            runtimes[name] = None
            if verbose:
                print(f"FAILED: {type(e).__name__}: {e}")
    if verbose:
        n_ok = sum(v is not None for v in runtimes.values())
        print(f"\n=== Done — {n_ok}/{len(methods)} methods succeeded ===")
    return embeddings, runtimes


# ── DEMaP subsampling (DTNE protocol) ─────────────────────────────────────────

def demap_subsampled(D_geo_full=None, X_highd=None, embedding=None,
                     n_subsample=2000, n_repeats=50,
                     knn=30, random_state=42):
    """DTNE-style DEMaP via subsampling.

    Builds the geodesic kNN graph ONCE on the full reference data, then
    averages Spearman correlations over n_repeats random subsamples of
    the precomputed distance matrix. Matches the canonical DEMaP
    convention (Moon et al. 2019) and DTNE's protocol.

    Parameters
    ----------
    D_geo_full : (n, n) ndarray, optional
        Precomputed full-data geodesic distance matrix. Pass this if
        you've already computed it once for use across multiple methods,
        avoiding redundant computation. If None, computed from X_highd.
    X_highd : (n, d) ndarray, optional
        High-dimensional reference data. Required if D_geo_full is None.
        Ignored if D_geo_full is provided.
    embedding : (n, 2) ndarray
        Low-dimensional embedding to evaluate.
    n_subsample : int, default=2000
        Cells per subsample (DTNE convention).
    n_repeats : int, default=50
        Number of subsample repetitions (DTNE convention).
    knn : int, default=30
        Neighbors for the high-D kNN graph. Only used when D_geo_full
        is None.
    random_state : int, default=42

    Returns
    -------
    mean_rho, std_rho : float, float
        Mean and (population) std of Spearman correlations across
        n_repeats subsamples.
    """
    if D_geo_full is None:
        if X_highd is None:
            raise ValueError("Must provide either D_geo_full or X_highd")
        D_geo_full = compute_shortest_path_geodesic(X_highd, k_geo=knn)
    
    n = D_geo_full.shape[0]
    n_subsample = min(n_subsample, n)
    
    rng = np.random.default_rng(random_state)
    correlations = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=n_subsample, replace=False)

        # Condensed (upper-triangle) form — ~2x fewer elements than .flatten()
        D_geo_sub = squareform(D_geo_full[idx][:, idx], checks=False)
        D_emb_sub = pdist(embedding[idx])

        rho, _ = spearmanr(D_geo_sub, D_emb_sub)
        correlations.append(rho)
    
    return float(np.mean(correlations)), float(np.std(correlations, ddof=1))


# ── Metrics table ─────────────────────────────────────────────────────────────

def continuity(X_high, X_low, n_neighbors=5):
    """Dual of trustworthiness — penalizes missing high-D neighbors in low-D.

    Trustworthiness asks: are the k nearest neighbors in the embedding also
    neighbors in the high-D space? Continuity asks the reverse: are the k
    nearest neighbors in high-D still neighbors in the embedding?

    Mathematically continuity(X, Y) = trustworthiness(Y, X); reporting both
    gives a more complete picture of local-neighborhood preservation than
    either alone.

    Parameters
    ----------
    X_high : (n, d_high) ndarray
        High-dimensional reference.
    X_low : (n, d_low) ndarray
        Low-dimensional embedding.
    n_neighbors : int, default=5
        Neighborhood size for the metric.

    Returns
    -------
    float in [0, 1]
        1.0 means all high-D neighbors are preserved in the embedding.
    """
    return trustworthiness(X_low, X_high, n_neighbors=n_neighbors)



def compute_bio_metrics(embeddings, X_input, label_codes=None,
                       n_clusters=None,
                       X_ref=None, 
                       runtimes=None,
                       demap_n_subsample=2000, demap_n_repeats=50,
                       demap_knn=50, trust_k=15,
                       random_state=42, plot_order=None,
                       verbose=True):
    """Per-method bio metrics table.

    Computes:
        - DEMaP (subsampled, DTNE protocol): mean + std over `n_repeats`
          subsamples of size `n_subsample`. Spearman correlation between
          embedding distances and geodesic distances on a kNN graph of
          X_ref. Higher is better (range [0, 1]).
        - ARI / NMI from Agglomerative clustering of the 2D embedding
          against `label_codes`.
        - Trustworthiness: are the embedding's k nearest neighbors also
          neighbors in X_input? Higher is better (range [0, 1]).
        - Continuity: are X_input's k nearest neighbors also neighbors in
          the embedding? Higher is better (range [0, 1]).
        - Runtime in seconds, if provided.

    Two separate high-D references because they measure different things:

    - **X_input** is what the methods saw as input (e.g. PCA-30 or PCA-100).
      Used for trustworthiness and continuity: these measure *local*
      neighborhood preservation, and the fair comparison is "did the method
      preserve what it saw?"
    - **X_ref** is the strictest reference available (e.g. raw expression,
      full LSI, full transcriptome). Used for DEMaP: this measures global
      geodesic fidelity, and the honest comparison is against the underlying
      biological manifold rather than a method-friendly projection.

    On most datasets X_input == X_ref is fine (e.g. Pancreas/Nestorowa where
    the canonical input *is* PCA-30 and no rawer representation is benchmarked
    against). On Embryoid Body and Lymphoid the distinction matters: passing
    raw sqrt-expression / full LSI as X_ref aligns DEMaP with DTNE's published
    appendix convention, while X_input remains the PCA representation.

    Parameters
    ----------
    embeddings : dict[str, (n, 2) ndarray]
        Method name -> 2D embedding.
    X_input : (n, d_input) ndarray
        Input passed to the methods (e.g. PCA-30). Used for trustworthiness
        and continuity.
    X_ref : (n, d_ref) ndarray
        High-dimensional reference for DEMaP (e.g. raw sqrt-expression).
        Pass `X_input` here if you want all metrics on the same reference
        (older convention).
    label_codes : (n,) int array
        Integer cell-type codes for ARI / NMI.
    n_clusters : int
        Target cluster count for Agglomerative clustering.
    runtimes : dict[str, float] | None
        If provided, added as the "Runtime (s)" column.
    demap_n_subsample : int, default=2000
        Subsample size for DEMaP.
    demap_n_repeats : int, default=50
        Number of subsample repetitions for DEMaP (mean + std).
    demap_knn : int, default=50
        kNN graph size for DEMaP geodesics. Higher k reduces risk of
        disconnected components on heterogeneous data.
    trust_k : int, default=15
        Neighborhood size for trustworthiness and continuity. Matches the
        cross-dataset K_NN default.
    random_state : int, default=42
        Seed for DEMaP subsampling.
    plot_order : list[str] | None
        Restricts and orders output rows. Defaults to `embeddings` insertion
        order.
    verbose : bool, default=True

    Returns
    -------
    pd.DataFrame
        Indexed by method name. Columns: DEMaP (mean), DEMaP (std),
        ARI, NMI, Trustworthiness, Continuity, Runtime (s).
    """
    # Backward-compat: if only X_input is provided, use it as both
    if X_ref is None:
        X_ref = X_input

    # Validate required args (now-optional in signature, but still required)
    if label_codes is None or n_clusters is None:
        raise ValueError(
            "label_codes and n_clusters are required. "
            "They were made keyword-only with no default to preserve "
            "backward compatibility with the legacy 4-positional call."
        )

    if plot_order is None:
        plot_order = list(embeddings.keys())

    if verbose:
        print("Computing bio metrics:")
        print(f"  DEMaP: n_subsample={demap_n_subsample}, "
              f"n_repeats={demap_n_repeats}, knn={demap_knn}")
        print(f"  Trustworthiness/Continuity: k={trust_k}")
        if X_input is X_ref:
            print(f"  X_input == X_ref (single reference, shape {X_input.shape})")
        else:
            print(f"  X_input shape: {X_input.shape} "
                  f"(for trustworthiness/continuity)")
            print(f"  X_ref shape:   {X_ref.shape} (for DEMaP)")
        print()

    rows = []
    for name in plot_order:
        if name not in embeddings:
            continue
        emb = embeddings[name]
        if verbose:
            print(f"  {name:18s}", end=" ", flush=True)

        pred = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='average',
        ).fit_predict(emb)

        T = trustworthiness(X_input, emb, n_neighbors=trust_k)
        C = continuity(X_input, emb, n_neighbors=trust_k)

        demap_mean, demap_std = demap_subsampled(
            X_highd=X_ref, embedding=emb,
            n_subsample=demap_n_subsample, n_repeats=demap_n_repeats,
            knn=demap_knn, random_state=random_state,
        )

        if verbose:
            print(f"DEMaP = {demap_mean:.3f} ± {demap_std:.3f}, "
                  f"T = {T:.3f}, C = {C:.3f}")

        rt = runtimes.get(name) if runtimes is not None else None
        rows.append({
            'Method':          name,
            'DEMaP (mean)':    demap_mean,
            'DEMaP (std)':     demap_std,
            'ARI':             adjusted_rand_score(label_codes, pred),
            'NMI':             normalized_mutual_info_score(label_codes, pred),
            'Trustworthiness': T,
            'Continuity':      C,
            'Runtime (s)':     rt if rt is not None else np.nan,
        })

    return pd.DataFrame(rows).set_index('Method')


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory-fidelity: geodesic-from-root and correlation triple
# Matches DTNE's nest_plot.ipynb evaluation protocol.
# ─────────────────────────────────────────────────────────────────────────────


def geodesic_from_root(X, root_idx, k_neighbors=20):
    """Geodesic distance from ``root_idx`` to all other rows of ``X``.

    Builds a symmetric kNN graph on Euclidean distances and runs Dijkstra
    from the root cell. Matches DTNE's ``adjacency_dist_matrix`` protocol
    (`nest_plot.ipynb`).

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Feature matrix. For trajectory-fidelity evaluation this is
        typically each method's 2D embedding; passing the high-D PCA
        gives the Isomap-style reference geodesic.
    root_idx : int
        Index of the root cell.
    k_neighbors : int, default 20
        Number of neighbors per cell in the kNN graph.

    Returns
    -------
    geo : ndarray, shape (n,)
        Shortest-path distances from ``root_idx``. Cells unreachable
        from the root (disconnected components) get ``np.inf``;
        ``correlation_triple`` masks these out.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(X)
    knn_dists, knn_indices = nbrs.kneighbors(X)

    indptr = np.arange(0, (n + 1) * k_neighbors, k_neighbors)
    k_matrix = sparse.csr_matrix(
        (knn_dists.ravel(), knn_indices.ravel(), indptr),
        shape=(n, n),
    )
    # Symmetrize: edge weight = max of the two directed kNN distances
    # (DTNE convention; equivalent to "either is a neighbor of the other").
    adjacency = k_matrix.maximum(k_matrix.T)

    geo = sparse.csgraph.dijkstra(
        csgraph=adjacency, directed=False,
        indices=root_idx, return_predecessors=False,
    )

    n_unreachable = int(np.isinf(geo).sum())
    if n_unreachable > 0:
        # Common on bio data with very small k_neighbors; usually a sign
        # the graph fragmented into components. Caller decides what to do.
        import warnings
        warnings.warn(
            f"geodesic_from_root: {n_unreachable}/{n} cells unreachable "
            f"from root_idx={root_idx} at k_neighbors={k_neighbors}. "
            "Correlations will mask these out.",
            stacklevel=2,
        )
    return geo


def correlation_triple(x, y):
    """Return (Pearson, Spearman, Kendall) correlation between ``x`` and ``y``.

    Non-finite entries (``nan``/``inf``) in either array are masked out
    before correlating. Returns ``(nan, nan, nan)`` if fewer than 3
    finite paired observations remain.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x[mask], y[mask]
    p, _ = pearsonr(xm, ym)
    s, _ = spearmanr(xm, ym)
    k, _ = kendalltau(xm, ym)
    return float(p), float(s), float(k)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_embedding_row(embeddings, plot_order, cell_colors, zip_types,
                       runtimes=None, panel_size=BIO_PANEL_SIZE,
                       point_kwargs=None, legend_title="Cell type",
                       save_path=None, show=True):
    """Single-row layout with shared bottom-centre legend.

    Parameters
    ----------
    embeddings : dict[str, (n, 2) ndarray]
    plot_order : list[str]
        Panels are rendered in this order; missing methods get blank panels
        with the method title (matches pancreas.ipynb behaviour).
    cell_colors : sequence of length n
    zip_types : dict[label -> color]
        Drives the shared legend.
    runtimes : dict[str, float] | None
        If given, appends "(t.ts)" to the panel title.
    save_path : Path | str | None
        Saves ``save_path.pdf`` and ``save_path.png`` (400 DPI).
    """
    if point_kwargs is None:
        point_kwargs = BIO_POINT_KWARGS
    n_panels = len(plot_order)
    ASPECT = 4 / 3
    fig, axes = plt.subplots(
        1, n_panels, figsize=(panel_size * n_panels, panel_size / ASPECT + 1.0),
        #figsize=(panel_size * n_panels, panel_size + 1.0),
    )
    if n_panels == 1:
        axes = np.array([axes])
    for ax, name in zip(axes, plot_order):
        if name in embeddings:
            emb = embeddings[name]
            ax.scatter(emb[:, 0], emb[:, 1], c=cell_colors, **point_kwargs)
            #ax.set_aspect("equal") # , adjustable="box", datalim
        title = name
        if runtimes is not None and runtimes.get(name) is not None:
            title += f"\n({runtimes[name]:.1f}s)"
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    _attach_shared_legend(fig, zip_types, legend_title, y=-0.06)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if save_path is not None:
        _save_pdf_png(fig, save_path)
    if show:
        plt.show()
    return fig, axes


def plot_embedding_grid(embeddings, plot_order, cell_colors, zip_types,
                        n_cols=4, runtimes=None, panel_size=BIO_PANEL_SIZE,
                        point_kwargs=None, legend_title="Cell type",
                        save_path=None, show=True):
    """Multi-row grid layout (default 4 columns), shared bottom-centre legend.

    Parameters mirror ``plot_embedding_row``. Methods missing from
    ``embeddings`` are skipped (no blank panel); trailing axes in the last
    row are hidden via ``set_visible(False)``.
    """
    if point_kwargs is None:
        point_kwargs = BIO_POINT_KWARGS
    available = [n for n in plot_order if n in embeddings]
    n_rows = -(-len(available) // n_cols)  # ceiling division
    ASPECT = 4 / 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_size * n_cols, (panel_size / ASPECT) * n_rows + 1.0),
        #figsize=(panel_size * n_cols, panel_size * n_rows + 1.0),
    )
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, available):
        emb = embeddings[name]
        ax.scatter(emb[:, 0], emb[:, 1], c=cell_colors, **point_kwargs)
        #ax.set_aspect("equal") # , adjustable="box", datalim
        title = name
        if runtimes is not None and runtimes.get(name) is not None:
            title += f"  ({runtimes[name]:.1f}s)"
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[len(available):]:
        ax.set_visible(False)
    _attach_shared_legend(fig, zip_types, legend_title, y=-0.04)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path is not None:
        _save_pdf_png(fig, save_path)
    if show:
        plt.show()
    return fig, axes


def plot_pseudotime_panels(method, method_name, embedding, root_idx,
                           labels, cell_colors, zip_types,
                           order_method='_order_cells',
                           panel_size=BIO_PANEL_SIZE,
                           point_kwargs=None,
                           save_path=None, show=True):
    """Two-panel pseudotime visualisation for a single method.

    Panel 1 — ground-truth cell types.
    Panel 2 — pseudotime (rank-normalised), with colourbar.

    The ``order_method`` parameter handles the EntroPath/DTNE attribute-name
    divergence (``_order_cells`` vs ``order_cells``) — promote to a public
    name in EntroPath and this parameterisation becomes unnecessary.

    Returns
    -------
    pseudotime : (n,) ndarray | None
        ``None`` if the method lacks ``order_method``.
    """
    if point_kwargs is None:
        point_kwargs = BIO_POINT_KWARGS

    if not hasattr(method, order_method):
        print(f"{method_name} pseudotime ordering not available — skipping.")
        return None

    root_label = labels[root_idx]
    print(f"Root cell {root_idx}: type = {root_label}")

    order_fn = getattr(method, order_method)
    pseudotime = np.asarray(
        order_fn(root_cells=[root_idx], normalization='rank'),
    ).ravel()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * panel_size + 1.0, panel_size + 0.5),
    )

    # Panel 1: ground-truth cell types
    axes[0].scatter(embedding[:, 0], embedding[:, 1],
                    c=cell_colors, **point_kwargs)
    axes[0].set_title('Ground-truth cell types', fontsize=11, pad=6)

    # Panel 2: pseudotime
    sc_handle = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                                c=pseudotime, cmap='Spectral', **point_kwargs)
    axes[1].set_title(
        f"{method_name} pseudotime\n(root = cell {root_idx}, {root_label})",
        fontsize=11, pad=6,
    )
    plt.colorbar(sc_handle, ax=axes[1], fraction=0.05, pad=0.02,
                 label='pseudotime (rank)')

    for ax in axes:
        #ax.set_aspect("equal") # , adjustable="box", datalim
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        _save_pdf_png(fig, save_path)
    if show:
        plt.show()

    return pseudotime


def plot_pseudotime_boxplot(pseudotime, labels, label_categories, zip_types,
                            method_name, root_label,
                            save_path=None, show=True):
    """Boxplot of pseudotime by cell type — quality check on ordering.

    Root population should have the smallest pseudotime; terminal populations
    the largest. Box colours match ``zip_types``.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(label_categories))
    boxplot_data = [pseudotime[labels == cat] for cat in label_categories]
    bp = ax.boxplot(boxplot_data, positions=positions,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, cat in zip(bp['boxes'], label_categories):
        patch.set_facecolor(zip_types[cat])
    ax.set_xticks(positions)
    ax.set_xticklabels(label_categories, rotation=45, ha='right')
    ax.set_ylabel(f"{method_name} pseudotime (rank)")
    ax.set_title(f"Pseudotime by cell type (root = {root_label})")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        _save_pdf_png(fig, save_path)
    if show:
        plt.show()
    return fig, ax


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_bio_artifacts(embeddings, runtimes, labels, label_codes, zip_types,
                       X_shape, config, results_dir, basename):
    """Pickle the full benchmark state under ``{basename}_embeddings.pkl``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = results_dir / f"{basename}_embeddings.pkl"
    with open(artifact_path, 'wb') as f:
        pickle.dump({
            'embeddings':  embeddings,
            'runtimes':    runtimes,
            'labels':      labels,
            'label_codes': label_codes,
            'zip_types':   zip_types,
            'X_shape':     X_shape,
            'config':      config,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact_path


def save_metrics_csv(df, results_dir, basename):
    """Save metrics DataFrame to ``{basename}_metrics.csv`` (4 decimal places)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{basename}_metrics.csv"
    df.to_csv(metrics_path, float_format='%.4f')
    return metrics_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_pdf_png(fig, save_path):
    """Save figure to both vector (PDF) and raster (PNG, 400 DPI) at save_path."""
    save_path = Path(save_path)
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.png'), dpi=400, bbox_inches='tight')


def _attach_shared_legend(fig, zip_types, title, y=-0.02):
    handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor=color, markeredgewidth=0,
                          markersize=8, label=label)
               for label, color in zip_types.items()]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 8), bbox_to_anchor=(0.5, y),
               frameon=False, fontsize=9, title=title)

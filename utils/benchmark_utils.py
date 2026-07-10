"""
benchmark_utils.py
==================

Shared evaluation, formatting, and constants for all manifold-learning
benchmark notebooks (Swiss Roll, Swiss Hole, artificial tree, torus,
real-data datasets).

Dataset-specific generation (make_swiss_roll, make_torus, etc.) lives
in the per-dataset modules under utils/.

Evaluation conventions follow DEMaP (Moon et al., 2019), HeatGeo
(Huguet et al., 2023), and DTNE (Wei et al., 2025):
ground-truth geodesics are computed on the noiseless manifold and
embeddings are evaluated against rank-based and neighborhood metrics.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path, connected_components
from zadu import zadu

import phate
import umap

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from utils.baseline_methods import ShortestPathEmb, DiffusionMapEmb
from entropath import EntroPath

from heatgeo.embedding import HeatGeo
from dtne import DTNE

from utils.helpers import compute_shortest_path_geodesic



K_NN_DEFAULT = 15
USE_LANDMARKS_DEFAULT = False



def get_methods(seed, k_nn=K_NN_DEFAULT):
    """
    Instantiate all baseline methods plus EntroPath for benchmarking.

    Parameters
    ----------
    seed : int
        Random seed passed to all stochastic methods.
    k_nn : int
        Number of nearest neighbours for kNN-graph-based methods.

    Returns
    -------
    dict[str, object]
        Method name -> fitted-ready estimator.
    """
    return {
        # Linear
        'PCA':                 PCA(n_components=2),
        'MDS':                 MDS(n_components=2, dissimilarity='euclidean',
                                   random_state=seed),
        # Local
        'UMAP':                umap.UMAP(n_components=2, n_neighbors=k_nn,
                                         random_state=seed),
        't-SNE':               TSNE(n_components=2, init='pca',
                                    random_state=seed),
        # Geodesic
        'Isomap':              Isomap(n_components=2, n_neighbors=k_nn),
        'Shortest Path':       ShortestPathEmb(k=k_nn, random_state=seed),
        # Spectral
        'Laplacian Eigenmaps': SpectralEmbedding(n_components=2,
                                                 affinity='nearest_neighbors',
                                                 n_neighbors=k_nn,
                                                 eigen_solver='arpack',
                                                 random_state=seed),
        'Diffusion Maps':      DiffusionMapEmb(k=k_nn, tau=10, alpha=1.0,
                                               random_state=seed),
        # Diffusion-distance
        'PHATE':               phate.PHATE(knn=k_nn, random_state=seed,
                                           n_landmark=None, verbose=False),
        'HeatGeo':              HeatGeo(knn=k_nn), # random_state disabled due to inconsistent support across methods/backends
        'DTNE':                 DTNE(k_neighbors=k_nn),
        'EntroPath':            EntroPath(k_neighbors=k_nn, use_landmarks=False,
                                          kernel="gaussian",
                                          random_state=seed, verbose=False),
    }


# ── Method groupings ──────────────────────────────────────────────────────────

METHODS = ['PCA', 'MDS', 'UMAP', 't-SNE',
           'Isomap', 'Shortest Path',
           'Laplacian Eigenmaps', 'Diffusion Maps',
           'PHATE', 'HeatGeo', 'DTNE', 'EntroPath']

# Methods that expose a meaningful distance matrix for Level 1 evaluation
LEVEL1_METHODS = ['Euclidean', 'Isomap', 'Shortest Path',
                  'Diffusion Maps', 'PHATE',
                  'HeatGeo', 'DTNE', 'EntroPath']

# Canonical plot order: linear -> local -> geodesic -> spectral -> diffusion

METHOD_FAMILIES = {
    'Linear':              ['PCA', 'MDS'],
    'Local':               ['UMAP', 't-SNE'],
    'Geodesic':            ['Isomap', 'Shortest Path'],
    'Spectral':            ['Laplacian Eigenmaps', 'Diffusion Maps'],
    'Diffusion-distance':  ['PHATE', 'HeatGeo', 'DTNE', 'EntroPath'],
}


# ── Metric metadata ───────────────────────────────────────────────────────────

METRIC_INFO = {
    'spearman_row':    'Spearman (row)',
    'pearson_row':     'Pearson (row)',
    'trust':           'Trustworthiness',
    'continuity':      'Continuity',
    'mrre_false':      'MRRE false (↓)',
    'mrre_missing':    'MRRE missing (↓)',
    'spearman_global': 'Spearman (global)',
    'pearson_global':  'Pearson (global)',
    'demap':           'DEMaP',
}

# Indicates whether higher values are better (True) or lower (False).
# Used for table formatting and boxplot orientation conventions.
METRIC_HIGHER_IS_BETTER = {
    'spearman_row':    True,
    'pearson_row':     True,
    'trust':           True,
    'continuity':      True,
    'mrre_false':      False,
    'mrre_missing':    False,
    'spearman_global': True,
    'pearson_global':  True,
    'demap':           True,
}

# Default trustworthiness / continuity / MRRE neighborhood size.
# Matches DTNE and HeatGeo evaluation protocols.
DEFAULT_K_TRUST = 50
KNN_PRECISION_KS = (5, 15, 50, 100)


# ── Core metric helpers ───────────────────────────────────────────────────────

def _rowwise_corr(D1, D2, method='spearman'):
    """
    DTNE/HeatGeo-style: average per-row correlation, diagonal excluded.
    """
    N    = D1.shape[0]
    vals = []
    for i in range(N):
        a = np.delete(D1[i], i)
        b = np.delete(D2[i], i)
        if method == 'spearman':
            r, _ = spearmanr(a, b)
        elif method == 'pearson':
            r, _ = pearsonr(a, b)
        else:
            raise ValueError("method must be 'spearman' or 'pearson'")
        vals.append(r)
    return float(np.nanmean(vals))


# ── Diagnostic-only metrics (kept available, not part of main eval) ──────────

def stress_kruskal(D_target, D_emb):
    """
    Kruskal stress-1 with optimal isotropic scaling. Scale-invariant in D_emb.

    Diagnostic use only — not included in level1_metrics / level2_metrics,
    and not reported in the main paper, since the diffusion-distance methods
    we benchmark against (PHATE, HeatGeo, DTNE) do not use stress as a primary
    metric. See Smelser et al. (2024) for known issues with stress under
    cross-method scale differences.
    """
    upper = np.triu_indices(len(D_target), k=1)
    d_t = D_target[upper]
    d_e = D_emb[upper]
    alpha = np.dot(d_t, d_e) / np.dot(d_e, d_e)
    return float(np.sqrt(
        np.sum((d_t - alpha * d_e)**2) / np.sum(d_t**2)
    ))


def knn_precision(D_true, D_pred, ks=(5, 15, 50, 100)):
    """
    Multi-scale neighborhood preservation (HeatGeo / Kobak-Berens style).

    Returns precision@k = |N_k(true) ∩ N_k(pred)| / k, averaged over points.
    With equal-size neighbor sets, precision == recall.

    Diagnostic use only — not included in level1_metrics / level2_metrics.
    Available for HeatGeo-style multi-scale evaluation if needed.
    """
    n = D_true.shape[0]
    out = {}
    for k in ks:
        if k >= n:
            continue
        nbr_true = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(D_true)
        nbr_pred = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(D_pred)
        adj_true = nbr_true.kneighbors_graph(n_neighbors=k).astype(bool)
        adj_pred = nbr_pred.kneighbors_graph(n_neighbors=k).astype(bool)
        overlap = adj_true.multiply(adj_pred).sum()
        out[f'p@{k}'] = float(overlap / (k * n))
    return out


# ── Public metric API ─────────────────────────────────────────────────────────

def demap(data_true, embedding, knn=30):
    """DEMaP per Moon et al. (2019).
    
    Spearman ρ between kNN-shortest-path geodesic on clean reference data and
    Euclidean distances in the method's embedding.
    
    Parameters
    ----------
    data_true : (n, d) ndarray
        Clean (noise-free) reference data.
    embedding : (n, d') ndarray
        Low-dimensional embedding produced by the method.
    knn : int, default=30
        Number of neighbors for the ground-truth geodesic graph
        (PHATE convention).
    
    Returns
    -------
    rho : float
        Spearman rank correlation in [-1, 1].
    """
    D_geo = compute_shortest_path_geodesic(data_true, k_geo=knn)
    D_emb = squareform(pdist(embedding))
    return float(spearmanr(D_geo.flatten(), D_emb.flatten()).correlation)


def level2_metrics(X, emb, D_geo, k=DEFAULT_K_TRUST):
    """
    Full embedding evaluation, DEMaP/HeatGeo/DTNE style.

    Parameters
    ----------
    X     : (n, d) input data (used by zadu for trust/continuity/MRRE)
    emb   : (n, 2) embedding
    D_geo : (n, n) ground-truth geodesic distance matrix
            (computed on the noiseless manifold per DEMaP convention)
    k     : neighbourhood size for trustworthiness / continuity / MRRE

    Returns
    -------
    dict with keys matching METRIC_INFO
    """
    D_emb = squareform(pdist(emb, metric='euclidean'))
    upper = np.triu_indices(len(X), k=1)

    spec = [
        {'id': 'tnc',  'params': {'k': k}},
        {'id': 'mrre', 'params': {'k': k}},
    ]
    scores = zadu.ZADU(spec, X).measure(emb)
    tnc, mrre = scores

    return {
        'spearman_row':    _rowwise_corr(D_emb, D_geo, 'spearman'),
        'pearson_row':     _rowwise_corr(D_emb, D_geo, 'pearson'),
        'trust':           tnc['trustworthiness'],
        'continuity':      tnc['continuity'],
        'mrre_false':      mrre['mrre_false'],
        'mrre_missing':    mrre['mrre_missing'],
        'spearman_global': float(spearmanr(D_geo[upper], D_emb[upper])[0]),
        'pearson_global':  float(pearsonr(D_geo[upper], D_emb[upper])[0]),
    }


def level1_metrics(D_method, D_geo):
    """
    Evaluate distance matrix quality before embedding (HeatGeo-style).
    
    D_method : (n, n) distance matrix from the method's internal representation
    D_geo    : (n, n) ground-truth geodesic distance matrix
    """
    upper = np.triu_indices(len(D_method), k=1)
    return {
        'spearman_row':    _rowwise_corr(D_method, D_geo, 'spearman'),
        'pearson_row':     _rowwise_corr(D_method, D_geo, 'pearson'),
        'spearman_global': float(spearmanr(
                               D_geo[upper], D_method[upper])[0]),
        'pearson_global':  float(pearsonr(
                               D_geo[upper],D_method[upper])[0]),
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(values):
    """Format list of floats as 'mean ± std' string."""
    if values is None or len(values) == 0:
        return '—'
    return f'{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}'


# ── Saving artifacts ─────────────────────────────────────────────────────────

def save_synthetic_artifacts(
    embeddings, runtimes, X_3d, X_noisy, labels,
    D_geo, level2_results, config, results_dir, basename,
):
    """Save synthetic-benchmark state for offline replotting."""
    from pathlib import Path
    import pickle
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = results_dir / f"{basename}_artifacts.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({
            "embeddings":     embeddings,      # dict[str, (n,2) ndarray]
            "runtimes":       runtimes,
            "X_3d":           X_3d,            # ground-truth manifold (n,3)
            "X_noisy":        X_noisy,         # noisy input fed to methods
            "labels":         labels,          # 1D color/parametric label
            "D_geo":          D_geo,           # (n,n) ground-truth geodesic
            "level2_results": level2_results,  # dict[method -> metrics dict]
            "config":         config,          # n_samples, noise, seed, ...
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact_path
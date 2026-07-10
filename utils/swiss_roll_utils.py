import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn import datasets
from utils.helpers import compute_shortest_path_geodesic


# ── Helpers ─────────────────────────────────────────────────────────────

def _geodesic_from_unrolled(t, X_sub):
    """Analytic geodesic from intrinsic coordinates (t, height)."""
    unrolled = np.stack([t, X_sub[:, 1]], axis=1)
    return squareform(pdist(unrolled, metric='euclidean'))


def _analytic_geodesic_from_unrolled(t, h):
    """Analytic geodesic distance on the Swiss roll via arc length.
    
    For X = (t cos t, h, t sin t), the manifold unrolls isometrically to a
    flat 2D strip with coordinates (s(t), h), where
        s(t) = (1/2) [t sqrt(1 + t^2) + arcsinh(t)]
    is the arc length along the spiral.
    """
    s = 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))
    unrolled = np.stack([s, h], axis=1)
    return squareform(pdist(unrolled, metric='euclidean'))


# ── Dataset generation ──────────────────────────────────────────────────

def make_swiss_roll_with_shortest_path_geodesic(n_samples=2000, noise=0.05, 
                                                 height_scale=3.0, random_state=42, 
                                                 k_geo=15):
    rng = np.random.RandomState(random_state)
    X_clean, t = datasets.make_swiss_roll(n_samples, noise=0.0, random_state=random_state)
    X_clean[:, 1] *= height_scale
    X = X_clean + noise * rng.standard_normal(X_clean.shape)
    
    D_geo = compute_shortest_path_geodesic(
        X_clean, k_geo=k_geo, seed=random_state
    )
    return X, t, D_geo, X_clean


def make_swiss_roll_with_analytic_geodesic(n_samples=2000, noise=0.05,
                                  height_scale=3.0, random_state=42):
    """DEMaP-style uniform Swiss roll."""
    rng = np.random.RandomState(random_state)
    X_clean, t = datasets.make_swiss_roll(
        n_samples, noise=0.0, random_state=random_state)
    X_clean[:, 1] *= height_scale
    h_clean = X_clean[:, 1]
    X = X_clean + noise * rng.standard_normal(X_clean.shape)
    D_geo = _analytic_geodesic_from_unrolled(t, h_clean)
    return X, t, D_geo, X_clean


# ── Non-uniform Swiss roll ──────────────────────────────────────────────

def _generate_non_uniform_swiss_roll(n_samples, noise, height_scale,
                                     alpha, beta, random_state, pool_mult=5):
    """Swiss roll sampled with Beta(alpha, beta) density along the roll parameter t.

    alpha > beta  -> mass at large t (outer, tightly-wound turns)
    alpha < beta  -> mass at small t (inner turns)
    alpha = beta = 1 -> uniform

    Paper config: alpha=1, beta=4 (density (1-t_norm)^3, oversamples inner turns),
    pool_mult=5. These reproduce the published figures/tables exactly; changing
    pool_mult resamples the RNG stream and will alter results.
    """
    rng = np.random.RandomState(random_state)

    X_clean_pool, t_pool = datasets.make_swiss_roll(
        n_samples * pool_mult, noise=0.0, random_state=random_state)   # 5x, see note
    X_clean_pool[:, 1] *= height_scale

    t_norm = (t_pool - t_pool.min()) / (t_pool.max() - t_pool.min())
    # Beta(alpha, beta) density kernel, up to normalization:
    weights = t_norm ** (alpha - 1) * (1.0 - t_norm) ** (beta - 1)
    weights = np.clip(weights, 1e-8, None)
    weights /= weights.sum()

    chosen = rng.choice(len(X_clean_pool), size=n_samples,
                        replace=False, p=weights)

    X_clean = X_clean_pool[chosen]
    t = t_pool[chosen]
    X = X_clean + noise * rng.standard_normal(X_clean.shape)
    return X, t, X_clean


def make_non_uniform_swiss_roll_with_analytic_geodesic(
    n_samples=2000, noise=0.05, height_scale=3.0,
    alpha=1.0, beta=4.0, random_state=42,
):
    """Beta(alpha, beta) Swiss roll with analytic arc-length geodesic."""
    X, t, X_clean = _generate_non_uniform_swiss_roll(
        n_samples, noise, height_scale, alpha, beta, random_state)
    h_clean = X_clean[:, 1]
    D_geo = _analytic_geodesic_from_unrolled(t, h_clean)
    return X, t, D_geo, X_clean


def make_non_uniform_swiss_roll_with_shortest_path_geodesic(
    n_samples=2000, noise=0.05, height_scale=3.0,
    alpha=1.0, beta=4.0, random_state=42, k_geo=15,
):
    """Beta(alpha, beta) Swiss roll with shortest-path geodesic."""
    X, t, X_clean = _generate_non_uniform_swiss_roll(
        n_samples, noise, height_scale, alpha, beta, random_state)
    D_geo = compute_shortest_path_geodesic(
        X_clean, k_geo=k_geo, seed=random_state)
    return X, t, D_geo, X_clean


# ── Swiss hole ──────────────────────────────────────────────────────────

def make_swiss_hole_with_shortest_path_geodesic(n_samples=2000, noise=0.05,
                                  height_scale=3.0, random_state=42,
                                  k_geo=15):
    """DEMaP-style Swiss hole with hole-aware kNN-shortest-path geodesic."""
    rng = np.random.RandomState(random_state)
    
    # Clean coordinates with hole
    X_clean, t = datasets.make_swiss_roll(
        n_samples, noise=0.0, hole=True, random_state=random_state)
    X_clean[:, 1] *= height_scale
    
    # Add noise after clean coords established
    X = X_clean + noise * rng.standard_normal(X_clean.shape)

    D_geo = compute_shortest_path_geodesic(
        X_clean, k_geo=k_geo, seed=random_state,
    )
    
    return X, t, D_geo, X_clean
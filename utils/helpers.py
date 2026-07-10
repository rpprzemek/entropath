"""
helpers.py
==========

Utility functions for benchmarking manifold learning methods,
including geodesic distance computations. 
"""


import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial.distance import squareform, pdist


def extract_method_distances(method, name):
    """Return the distance matrix from a fitted method, or None if not exposed."""
    if hasattr(method, 'get_dist'):
        return method.get_dist()
    
    if name == 'EntroPath' and hasattr(method, 'distances_'):
        return method.distances_
    if name == 'DTNE' and hasattr(method, 'dists'):
        return method.dists
    if name == 'HeatGeo' and hasattr(method, 'dist'):
        return method.dist
    if name == 'Isomap' and hasattr(method, 'dist_matrix_'):
        return method.dist_matrix_
    if name == 'PHATE' and hasattr(method, 'diff_potential'):
        return squareform(pdist(method.diff_potential))
    
    return None


def compute_shortest_path_geodesic(X_clean, k_geo=15, seed=None):
    """Compute shortest-path geodesic distances on a kNN graph.

    DEMaP-style ground-truth geodesic. Works for any manifold where
    clean reference coordinates are available.

    Parameters
    ----------
    X_clean : (n, d) ndarray
        Clean reference coordinates (noise-free).
    k_geo : int, default=15
        Number of nearest neighbors for graph construction.
    seed : int, optional
        Seed identifier for error messages (helps identify problematic runs).

    Returns
    -------
    D_geo : (n, n) ndarray
        Pairwise shortest-path distances. Symmetric, zero diagonal.

    Raises
    ------
    RuntimeError
        If the kNN graph has disconnected components, indicating unreliable
        geodesics. Increase k_geo or n_samples to fix.
    """
    G = kneighbors_graph(X_clean.astype(np.float32), n_neighbors=k_geo,
                         mode="distance", include_self=False)
    G = G.maximum(G.T)

    n_comp, _ = connected_components(G, directed=False)
    if n_comp > 1:
        prefix = f"seed {seed}: " if seed is not None else ""
        raise RuntimeError(
            f"{prefix}kNN graph has {n_comp} components "
            f"(k_geo={k_geo}, n={X_clean.shape[0]}). "
            f"Increase k_geo or n_samples to ensure connectivity."
        )

    D_geo = shortest_path(G, method="D", directed=False) # 'D' = Dijkstra
    D_geo = 0.5 * (D_geo + D_geo.T)
    np.fill_diagonal(D_geo, 0.0)
    return D_geo
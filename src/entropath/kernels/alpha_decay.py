# merw_phate/kernels/alpha_decay.py
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, coo_matrix
import numpy as np

from .registry import register_kernel   # relative import


@register_kernel("alpha_decay")
def build_affinity_alpha_decay(
    X, k=15, decay=40, normalize=False, sym_mode="max", random_state=None, **kwargs
):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)          # (n, k)

    adjacency_knn_indices = indices.copy()           # full indices

    distances = distances[:, 1:]                     # drop self
    indices   = indices[:, 1:]
    k_eff = k - 1

    sigma = np.maximum(distances[:, -1], 1e-12)      # adaptive bandwidth (n,)
    sig_i = sigma[:, np.newaxis]                     # (n, 1)
    sig_j = sigma[indices]                           # (n, k_eff)

    # ── alpha-decay kernel: exp(-(d/sigma)^decay), mean of directed ──
    kij = np.exp(-(distances / sig_i) ** decay)       # (n, k_eff)
    kji = np.exp(-(distances / sig_j) ** decay)
    vals = 0.5 * (kij + kji)

    row = np.repeat(np.arange(n), k_eff)
    col = indices.ravel()
    K = coo_matrix((vals.ravel(), (row, col)), shape=(n, n))

    if sym_mode == "max":
        K = K.maximum(K.T)
    elif sym_mode == "avg":
        K = (K + K.T) * 0.5
    K = K.tocsr()

    if normalize:
        d = np.array(K.sum(axis=1)).flatten()
        K = K.multiply(1.0 / np.maximum(d, 1e-12)[:, None])

    return K, adjacency_knn_indices, sigma
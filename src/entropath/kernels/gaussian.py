# merw_phate/kernels/gaussian.py
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from .registry import register_kernel   # relative import


@register_kernel("gaussian")
def build_affinity_gaussian(
    X, k=15, normalize=False, sym_mode="max", random_state=None, **kwargs
):
    #with timed_step("affinities", indent_level=1, logger=logger, verbose=self.verbose):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)          # (n, k)

    # KEEP full indices for DTNE compatibility
    adjacency_knn_indices = indices.copy()
    #self.adjacency_knn_indices_ = adjacency_knn_indices

    # remove self for kernel computation
    distances = distances[:, 1:]
    indices   = indices[:, 1:]
    k_eff = k - 1

    #sigma = distances[:, -1].clip(1e-12)             # (n,)
    sigma = np.maximum(distances[:, -1], 1e-12)

    # broadcasted kernel computation
    d2 = distances ** 2                              # (n, k_eff)
    sig_i = sigma[:, np.newaxis]                     # (n, 1)
    sig_j = sigma[indices]                           # (n, k_eff)

    vals = np.exp(-d2 / (sig_i * sig_j))             # (n, k_eff)

    # flatten for COO
    row = np.repeat(np.arange(n), k_eff)
    col = indices.ravel()
    #data = vals.ravel()

    K = coo_matrix((vals.ravel(), (row, col)), shape=(n, n))

    # symmetrize with max (your current choice)
    #K = K.maximum(K.T).tocsr()
    #K_assym = K.tocsr()  # keep asymmetric for now, symmetrize later if needed
    
    if sym_mode == "max":
        K = K.maximum(K.T)
    elif sym_mode == "avg":
        K = (K + K.T) * 0.5

    K = K.tocsr()

    if normalize:
        d = np.array(K.sum(axis=1)).flatten()
        K = K.multiply(1.0 / np.maximum(d, 1e-12)[:, None])

    # DTNE style normalization
    #d = np.array(K.sum(axis=1)).flatten()
    #d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    #D_inv_sqrt = sparse.diags(d_inv_sqrt)
    #K = D_inv_sqrt @ K @ D_inv_sqrt

    return K, adjacency_knn_indices, sigma




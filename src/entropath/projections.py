import numpy as np
from sklearn.neighbors import NearestNeighbors


def project_points_simple(X, X_lm, Z_lm, **kwargs):

    nn = NearestNeighbors(
        n_neighbors=50, # it was 10
        algorithm="auto"
    ).fit(X_lm)

    dist, idx = nn.kneighbors(X)

    weights = 1.0 / (dist + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    Z = np.zeros((X.shape[0], Z_lm.shape[1]))

    for i in range(X.shape[0]):
        Z[i] = np.sum(
            weights[i][:, None] * Z_lm[idx[i]],
            axis=0
        )

    return Z, weights, idx


def project_points_improved(X, X_lm, Z_lm, **kwargs):
        
    nn = NearestNeighbors(
        n_neighbors=50, # min(30, len(X_lm)),
        algorithm="auto"
    ).fit(X_lm)

    dist, idx = nn.kneighbors(X)

    # adaptive Gaussian weights
    sigma = np.median(dist, axis=1, keepdims=True) + 1e-12
    weights = np.exp(-(dist ** 2) / (sigma ** 2))

    weights /= weights.sum(axis=1, keepdims=True)

    Z = (weights[:, :, None] * Z_lm[idx]).sum(axis=1)

    return Z, weights, idx
        
    
def project_points_improved_adaptive(X, X_lm, Z_lm, k_neighbors=None, projection_bandwidth_exponent=None, k_project=None, **kwargs): #return_weights=False
    """Improved adaptive-bandwidth projection (exactly mirrors _build_affinity_fast kernel).

    - sigma_i = distance to k-th NN of query point **w.r.t. landmarks only**
    - sigma_j = distance to k-th NN of each landmark **w.r.t. landmarks only**
    - Uses exp(-d^2 / (sigma_i · sigma_j)) on the k nearest landmarks
    - Fully vectorized, no Python loops
    - Larger k (50–100) = smoother at high diffusion powers (recommended)
    """
    #if k_projection is None:
    #    k_projection = max(50, k_neighbors)          # auto-scale with your affinity k (or set manually)

    # Precompute local scales for ALL landmarks (once)
    nn_lm = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto").fit(X_lm)
    dist_lm, _ = nn_lm.kneighbors(X_lm)                  # (n_lm, k_aff_)
    sigma_lm = np.maximum(dist_lm[:, -1], 1e-12)         # (n_lm,)

    # k-NN from queries -> landmarks
    nn = NearestNeighbors(n_neighbors=k_project, algorithm="auto").fit(X_lm)
    dist, idx = nn.kneighbors(X)                         # (n_query, k)

    # Adaptive scales for query points (w.r.t. landmarks)
    sigma_query = np.maximum(dist[:, -1], 1e-12)         # (n_query,)

    # Adaptive Gaussian kernel on the k nearest only
    sig_i = sigma_query[:, np.newaxis]                   # (n_query, 1)
    sig_j = sigma_lm[idx]                                # (n_query, k)
    d2 = dist ** 2                                       # (n_query, k)

    if projection_bandwidth_exponent is not None:
        vals = np.exp(-d2 / ((sig_i * sig_j)**projection_bandwidth_exponent + 1e-12))         # +1e-12 prevents rare overflow when sigma is tiny
    else:
        vals = np.exp(-d2 / (sig_i * sig_j + 1e-12))

    # row-normalize to probabilities
    weights = vals / (vals.sum(axis=1, keepdims=True) + 1e-12)

    # Interpolate embedding
    Z = (weights[:, :, None] * Z_lm[idx]).sum(axis=1)   # (n_query, dim)

    return Z, weights, idx


""" def project_points_Pnm(diff_op, labels, Y_landmarks):
    
    Tk: (N x N) MERW power transition matrix
    labels: cluster assignments (N,)
    Y_landmarks: (M x d)
    

    N = diff_op.shape[0]
    M = Y_landmarks.shape[0]

    # Build PNM (point -> landmark)
    PNM = np.zeros((N, M))

    for j in range(M):
        idx = np.where(labels == j)[0]
        PNM[:, j] = diff_op[:, idx].sum(axis=1)

    # Project
    Y = PNM @ Y_landmarks

    return Y """
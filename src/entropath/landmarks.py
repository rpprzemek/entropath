import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def select_landmarks_random(
    X,
    n_landmarks=2000,
    random_state=None,
):
    n = X.shape[0]
    
    if n_landmarks >= n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    landmark_idx = rng.choice(n, size=n_landmarks, replace=False)
        
    return np.sort(landmark_idx)
    

def select_landmarks_kmeans(
    X_graph,
    n_landmarks=2000,
    random_state=None,
    fill_missing=True,
    logger=None,
    verbose=False,
):
    """
    Select landmarks using MiniBatchKMeans.

    Parameters
    ----------
    X_graph : array (n_samples, n_features)
    n_landmarks : int
    random_state : int or None
    fill_missing : bool
        If True, fill empty clusters with random points
    logger : optional logger
    verbose : bool

    Returns
    -------
    landmark_idx : array (n_landmarks,)
    """

    n = X_graph.shape[0]

    if n_landmarks >= n:
        return np.arange(n)
    
    # -------------------------
    # KMeans clustering
    # -------------------------

    kmeans = MiniBatchKMeans(
        n_clusters=n_landmarks,
        random_state=random_state,
        batch_size=min(3 * n_landmarks, n),
        n_init='auto'
    )
    
    labels = kmeans.fit_predict(X_graph)
    centers = kmeans.cluster_centers_

    # --------------------------------------------------
    # Pre-group indices (O(n))
    # --------------------------------------------------
    clusters = [[] for _ in range(n_landmarks)]
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)

    landmark_idx = []

    # -------------------------
    # Select closest point per cluster
    # -------------------------

    #min_cluster_size = 5

    for i in range(n_landmarks):
        cluster_points = clusters[i]
        
        if not cluster_points:
            continue

        cluster_points = np.array(cluster_points)

                # same as original (numerically stable)
        diff = X_graph[cluster_points] - centers[i]
        distances = np.einsum("ij,ij->i", diff, diff)

        closest_idx = cluster_points[np.argmin(distances)]
        landmark_idx.append(closest_idx)

    # --------------------------------------------------
    # Fill missing clusters
    # --------------------------------------------------
    missing = n_landmarks - len(landmark_idx)
    
    if missing > 0 and fill_missing:
        rng = np.random.default_rng(random_state)
        
        used = np.zeros(n, dtype=bool)
        used[landmark_idx] = True
        
        remaining = np.flatnonzero(~used)

        replace = len(remaining) < missing # -> False
        extra = rng.choice(remaining, size=missing, replace=replace)
        
        landmark_idx.extend(extra)

    return np.sort(np.array(landmark_idx))
    

def select_landmarks_kmeans_einsum(
    X,
    n_landmarks=2000,
    random_state=None,
    fill_missing=False,
):
    """
    K-means based landmark selection:
    - Selects points closest to each cluster center
    - Guarantees one landmark per cluster
    - Fills missing clusters randomly if needed
    """

    n = X.shape[0]

    if n_landmarks >= n:
        return np.arange(n)

    # Run MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_landmarks,
        random_state=random_state,
        batch_size=min(3 * n_landmarks, n),
        n_init='auto',
    )
    
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Pre-group points per cluster (O(n))
    clusters = [[] for _ in range(n_landmarks)]
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)

    landmark_idx = []

    #rng = np.random.default_rng(random_state)

    # Select closest point per cluster    
    for i in range(n_landmarks):
        cluster_points = clusters[i]
        if not cluster_points:
            continue

        cluster_points = np.array(cluster_points)
        
        diff = X[cluster_points] - centers[i]
        distances = np.einsum("ij,ij->i", diff, diff) #fast vectorized way to compute squared distances
        
        closest_idx = cluster_points[np.argmin(distances)]
        landmark_idx.append(closest_idx)

            
    # Fill missing clusters        
    missing = n_landmarks - len(landmark_idx)
    
    if missing > 0 and fill_missing:
        rng = np.random.default_rng(random_state)
        
        used = np.zeros(n, dtype=bool)
        used[landmark_idx] = True
        
        remaining = np.flatnonzero(~used)

        replace = len(remaining) < missing # -> False
        extra = rng.choice(remaining, size=missing, replace=replace) #replace=False
        
        landmark_idx.extend(extra)

    return np.sort(np.array(landmark_idx))



def select_landmarks_fps(
    X,
    n_landmarks=2000,
    random_state=None,
):
    """
    Greedy farthest-point sampling
    X : (n, d) – low dimensional coordinates work best
    """
    
    n = X.shape[0]
    
    if n_landmarks >= n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    
    # Start with random point
    selected = [rng.integers(0, n)]
    min_dist = np.full(n, np.inf)
    
    for _ in range(1, n_landmarks):
        # Update min distance to selected set
        dist_to_last = np.linalg.norm(X - X[selected[-1]], axis=1)
        min_dist = np.minimum(min_dist, dist_to_last)
        
        # Pick point with largest min-distance
        next_idx = np.argmax(min_dist)
        if min_dist[next_idx] < 1e-8:  # all points already covered
            break
            
        selected.append(next_idx)
    
    return np.sort(np.array(selected))
    

def select_landmarks_hybrid(
    X,
    n_landmarks=2000,
    random_state=None,
):
    n = X.shape[0]

    # Safety: avoid degenerate case
    if n_landmarks >= n:
        return np.arange(n)

    # PCA (only if high-dimensional)
    if X.shape[1] > 15:
        pca = PCA(n_components=10, random_state=random_state)
        X_low = pca.fit_transform(X)
    else:
        X_low = X
            
    # FPS (core method)
    return select_landmarks_fps(
        X_low,
        n_landmarks=n_landmarks,
        random_state=random_state,
    )


def select_landmarks_kmeans_robust(
    X_graph,
    n_landmarks=2000,
    random_state=None,
    fill_missing=True,
    trim_percentile=100,
    min_cluster_size=4,
    use_medoid=True,
    n_candidates=150,
    verbose=False,
):
    """
    Robust landmark selection with optional approximate medoid.
    """
    n = X_graph.shape[0]
    if n_landmarks >= n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)

    # KMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_landmarks,
        random_state=random_state,
        batch_size=min(3 * n_landmarks, n),
        n_init='auto',
        reassignment_ratio=0.02,
    )
    
    labels = kmeans.fit_predict(X_graph)
    centers = kmeans.cluster_centers_

    clusters = [[] for _ in range(n_landmarks)]
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)

    landmark_idx = []

    for i in range(n_landmarks):
        cluster_points = np.array(clusters[i])
        
        if len(cluster_points) < min_cluster_size:
            continue

        # Trim outliers
        diff = X_graph[cluster_points] - centers[i]
        distances = np.einsum("ij,ij->i", diff, diff)
        
        if trim_percentile < 100:
            thresh = np.percentile(distances, trim_percentile)
            mask = distances <= thresh
            trimmed_idx = cluster_points[mask]
        else:
            trimmed_idx = cluster_points

        if len(trimmed_idx) == 0:
            continue

        # === Select representative ===
        if use_medoid and len(trimmed_idx) > 1:
            best = select_approximate_medoid(
                X_graph, 
                trimmed_idx, 
                rng,
                n_candidates=n_candidates
            )
        else:
            # Fallback: closest to center
            best = trimmed_idx[np.argmin(distances[mask] if trim_percentile < 100 else distances)]

        landmark_idx.append(best)

    # Fill missing
    missing = n_landmarks - len(landmark_idx)
    if missing > 0 and fill_missing:
        used = np.zeros(n, dtype=bool)
        used[landmark_idx] = True
        remaining = np.flatnonzero(~used)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=missing, replace=len(remaining) < missing)
            landmark_idx.extend(extra)

    return np.sort(np.array(landmark_idx))



def select_approximate_medoid(
    X: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int = 150,
    n_inner_samples: int = 400
) -> int:
    """
    Fast approximate medoid selection.
    """
    n = len(indices)
    
    if n <= 30:
        return _exact_medoid(X, indices)
    
    # Sample candidates (points to evaluate as medoids)
    if n <= n_candidates:
        candidates = indices
    else:
        candidates = rng.choice(indices, size=n_candidates, replace=False)
    
    # Sample points for distance approximation
    if n <= n_inner_samples:
        inner_idx = indices
    else:
        inner_idx = rng.choice(indices, size=n_inner_samples, replace=False)
    
    X_inner = X[inner_idx]
    
    best_idx = int(candidates[0])
    min_sum_dist = np.inf
    
    for cand in candidates:
        diff = X[cand] - X_inner
        sum_dist = np.einsum('ij,ij->i', diff, diff).sum()   # squared euclidean
        
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            best_idx = int(cand)
            
    return best_idx


def _exact_medoid(X: np.ndarray, indices: np.ndarray) -> int:
    """Exact medoid for small clusters (fixed)."""
    if len(indices) == 1:
        return int(indices[0])
    
    X_cluster = X[indices]                    # shape: (m, d)
    m = len(X_cluster)
    
    # Compute pairwise squared distances efficiently
    diff = X_cluster[:, None, :] - X_cluster[None, :, :]   # shape: (m, m, d)
    dist_matrix = np.einsum('ijk,ijk->ij', diff, diff)     # FIXED: 'ijk' not 'ijkl'
    
    # Or even better (cleaner and often faster):
    # dist_matrix = np.sum((X_cluster[:, None, :] - X_cluster[None, :, :]) ** 2, axis=-1)
    
    sum_dists = dist_matrix.sum(axis=1)
    best_local = np.argmin(sum_dists)
    
    return int(indices[best_local])
# baseline_methods.py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.manifold import MDS


class ShortestPathEmb:
    """
    Isomap-style baseline: Euclidean kNN graph shortest path + MDS.
    Mirrors HeatGeo's ShortestPath implementation (other_emb.py).

    Parameters
    ----------
    k           : number of nearest neighbours for graph construction
    n_components: embedding dimensionality
    random_state: passed to MDS
    """

    def __init__(self, k=15, n_components=2, random_state=None):
        self.k            = k
        self.n_components = n_components
        self.random_state = random_state
        self.dist_        = None

    def _graph_shortest_path(self, X):
        G = kneighbors_graph(
            X, n_neighbors=self.k,
            mode='distance',
            include_self=False
        )
        D = shortest_path(G, method='auto', directed=False)
        # Guard against disconnected components
        if np.isinf(D).any():
            finite_max = D[np.isfinite(D)].max()
            D[np.isinf(D)] = finite_max * 2
        return D

    def fit(self, X):
        self.dist_ = self._graph_shortest_path(X)
        return self

    def transform(self, X=None):
        if self.dist_ is None:
            raise RuntimeError("Model not fitted.")
        return MDS(
            n_components=self.n_components,
            dissimilarity='precomputed',
            random_state=self.random_state
        ).fit_transform(self.dist_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform()

    def get_dist(self):
        """Return the precomputed shortest-path distance matrix."""
        if self.dist_ is None:
            raise RuntimeError("Model not fitted.")
        return self.dist_


class DiffusionMapEmb:
    """
    Diffusion Map baseline matching HeatGeo's DiffusionMap implementation.

    Uses:
    - Adaptive Gaussian kernel (self-tuning bandwidth via k-th neighbour)
    - Alpha normalisation for density correction
    - Eigendecomposition of P^tau
    - Returns top n_components non-trivial eigenvectors weighted by eigenvalues

    Parameters
    ----------
    k           : number of nearest neighbours
    tau         : diffusion time (power of transition matrix)
    alpha       : density normalisation (0=none, 1=full correction)
    n_components: embedding dimensionality
    random_state: unused, kept for API consistency
    """

    def __init__(self, k=15, tau=10, alpha=1.0,
                 n_components=2, random_state=None):
        self.k            = k
        self.tau          = tau
        self.alpha        = alpha
        self.n_components = n_components
        self.random_state = random_state
        self.embedding_   = None
        self.dist_        = None

    def _build_transition_matrix(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        dists, indices = nbrs.kneighbors(X)
        n = X.shape[0]

        # Self-tuning Gaussian kernel
        sigma = dists[:, -1]  # bandwidth = distance to k-th neighbour
        W = np.zeros((n, n))
        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                w = np.exp(
                    -dists[i, j_idx]**2 / (sigma[i] * sigma[j])
                )
                W[i, j] = w
                W[j, i] = w  # symmetrise

        # Alpha normalisation (density correction)
        q = W.sum(axis=1)
        W = W / np.outer(q**self.alpha, q**self.alpha)

        # Row-normalise to get transition matrix P
        d = W.sum(axis=1)
        P = W / d[:, None]
        return P

    def fit(self, X):
        P  = self._build_transition_matrix(X)
        Pt = np.linalg.matrix_power(P, self.tau)

        evals, evecs = np.linalg.eig(Pt)
        evals = np.real(evals)
        evecs = np.real(evecs)

        # Sort by magnitude, skip trivial eigenvalue (largest, ≈1)
        order = np.argsort(np.abs(evals))[::-1]
        idx   = order[1:self.n_components + 1]

        self.embedding_ = evals[idx] * evecs[:, idx]

        # Diffusion distance matrix for correlation metrics
        emb_full        = evals[order] * evecs[:, order]
        self.dist_      = squareform(pdist(emb_full, metric='euclidean'))
        return self

    def transform(self, X=None):
        if self.embedding_ is None:
            raise RuntimeError("Model not fitted.")
        return self.embedding_

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_

    def get_dist(self):
        """Return the full diffusion distance matrix (for correlation metrics)."""
        if self.dist_ is None:
            raise RuntimeError("Model not fitted.")
        return self.dist_
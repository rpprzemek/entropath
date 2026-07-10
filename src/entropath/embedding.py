import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform, cdist



import time
from contextlib import contextmanager

from entropath.utils import timed_step, IndentFormatter, alpha_normalize, _matmul, _compute_power
from entropath.kernels import KERNEL_REGISTRY
from entropath.mds import sgd_mds, sgd_mds_metric, classic, smacof, classic_mds_robust, classic_pca
from entropath.power_selection import von_neumann_entropy

from entropath.landmarks import (
    select_landmarks_kmeans,
    select_landmarks_kmeans_einsum,
    select_landmarks_fps,
    select_landmarks_hybrid,
    select_landmarks_random,
)

from entropath.projections import (
    project_points_simple,
    project_points_improved_adaptive,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(IndentFormatter("%(tabs)s%(message)s"))

logger.addHandler(handler)
logger.propagate = False



class BaseEmbedding:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    # --- Randomness handling ---
    def _get_rng(self):
        """Create a reproducible random number generator."""
        return np.random.default_rng(self.random_state)

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X=None):
        if not hasattr(self, "embedding_"):
            raise RuntimeError("Model not fitted.")
        return self.embedding_

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_



class EntroPath(BaseEmbedding):
    """Maximum Entropy Path Ensemble Embedding for manifold learning.

    EntroPath constructs a kNN affinity graph ``A`` on the input data and
    forms the spectrally normalized affinity

        A_tilde = A / lambda_max

    where ``lambda_max`` is the top (Perron) eigenvalue of ``A``. The
    t-step diffusion operator ``A_tilde ** t`` is computed and turned into
    log-transition dissimilarities

        D_sq_ij = -log( (A_tilde ** t)_ij )        # squared form
        D_ij    = sqrt( D_sq_ij )                   # linear (Varadhan) form

    which are embedded via metric MDS. The diffusion power ``t`` is selected
    automatically from the Von Neumann entropy (VNE) of the maximum-entropy
    random walk (MERW) transition matrix derived from ``A``; users can
    override it with an explicit integer ``t_power``. For large datasets a
    landmark approximation reduces the MDS problem to ~``n_landmarks`` points
    while preserving local fidelity in the short-time (Varadhan) regime.

    The spectral rescaling by ``lambda_max`` is the maximum-entropy
    normalization of Burda et al. [1]_: it gives ``A_tilde`` spectral radius
    1 so the powers ``A_tilde ** t`` stay bounded in [0, 1], and endows the
    associated random walk with maximum path entropy on the graph.
    Physically, ``D_ij`` is the negative log-likelihood of an MERW path
    ensemble of length ``t`` between points ``i`` and ``j`` -- a path
    free-energy under Boltzmann edge energies. In the Varadhan short-time
    limit this dissimilarity approximates the Riemannian geodesic distance on
    the underlying manifold [5]_, giving EntroPath a manifold-learning
    interpretation analogous to PHATE [2]_, HeatGeo [3]_, and DTNE [4]_.

    .. note::
        The Varadhan approximation carries a global prefactor
        ``sqrt(4 * t / lambda_max)``. It is **deliberately omitted** from the
        returned distances: a constant rescaling of every dissimilarity
        leaves the metric-MDS solution unchanged (it is absorbed into the MDS
        stress scaling), so dropping it keeps distances comparable across
        ``t_power`` values without affecting the embedding. The stored
        ``distances_`` are therefore exactly ``sqrt(-log(A_tilde ** t))``.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the output embedding.

    k_neighbors : int, default=15
        Number of nearest neighbors for the kNN affinity graph. Matches UMAP
        and DTNE conventions and is robust across dataset sizes from ~1.6k
        (Nestorowa) to ~16k cells (EB). (An ``"auto"`` adaptive-k path is
        sketched in the source but is currently disabled; pass an integer.)

    t_power : int or {"auto", "von_neumann_entropy"}, default="auto"
        Diffusion power. An integer fixes ``t`` directly. ``"auto"`` (and the
        explicit alias ``"von_neumann_entropy"``) selects ``t`` from the knee
        of the VNE curve of the MERW transition matrix. The resolved value is
        stored in ``t_power_``.

    metric : str, default="precomputed"
        Distance metric passed to the underlying MDS solver. Use
        ``"precomputed"`` when feeding distance matrices directly (the typical
        case for EntroPath).

    kernel : {"gaussian", "alpha_decay", "cauchy"}, default="gaussian"
        Affinity kernel for the kNN graph. ``"alpha_decay"`` is the
        PHATE-style kernel controlled by ``decay`` (its bandwidth comes from
        the per-point kNN distance; there is no separate ``alpha`` argument).
        ``"cauchy"`` is the Cauchy kernel controlled by ``cauchy_decay``.
        ``"gaussian"`` is the standard Gaussian kernel and is the only kernel
        that consumes ``kernel_norm`` and ``sym_mode`` (see those parameters).

    mds_solver : {"auto", "smacof", "sgd", "classic", "classic_mds_robust", "classic_pca", "sklearn"}, default="auto"
        MDS solver. ``"auto"`` resolves to ``"smacof"`` for landmark sets and
        for full problems with ``n <= 5000``, and to ``"sgd"`` for full
        problems with ``n > 5000``. ``"smacof"`` finds better local minima on
        diffusion-distance MDS but is iterative;
        ``"classic"`` use eigendecomposition and are deterministic; ``"sgd"``
        is a PHATE-style stochastic solver. The resolved choice is stored in
        ``mds_solver_``.

    mds_dissimilarity : {"squared", "linear"}, default="squared"
        Form of the dissimilarity fed to MDS as the target:
        - ``"squared"`` : embed ``D_sq = -log(A_tilde ** t)`` directly
          (default; emphasizes large-distance fidelity, better on
          branching/tree-structured data).
        - ``"linear"``  : embed ``D = sqrt(-log(A_tilde ** t))``
          (Varadhan-derived geodesic form, better on smooth manifolds).
        Note this only changes the MDS *target*; ``distances_`` always stores
        the linear form ``D``.

    max_iter : int, default=3000
        Maximum iterations for iterative MDS solvers (SMACOF, SGD). Ignored by
        closed-form solvers.

    n_pca : int or None, default=100
        Number of PCA components for input preprocessing. If
        ``X.shape[1] > n_pca``, PCA is applied before graph construction
        (PHATE convention). Set to None to disable.

    kernel_norm : bool, default=False
        Passed to the **gaussian** kernel as its ``normalize`` option (routed
        via ``kernel_params`` as ``gaussian__normalize``). Has no effect for
        the ``alpha_decay`` or ``cauchy`` kernels.

    sym_mode : {"max", "avg"}, default="max"
        Symmetrization of the asymmetric kNN affinity, passed to the
        **gaussian** kernel (``gaussian__sym_mode``): ``"max"`` takes the
        element-wise maximum (PHATE convention), ``"avg"`` the arithmetic
        mean. Has no effect for the ``alpha_decay`` or ``cauchy`` kernels,
        which apply their own symmetrization.

    t_max : int or "auto", default="auto"
        Upper bound for the diffusion-power sweep during automatic ``t_power``
        selection. ``"auto"`` currently resolves to a fixed ``150`` regardless
        of dataset size (see ``_auto_t_max``); the resolved value is stored in
        ``t_max_``. A warning is emitted if the selected ``t`` lands within
        10% of ``t_max`` (raise ``t_max`` or ``n_landmarks`` if so).

    epsilon : float, default=1e-21
        Floor applied to ``A_tilde ** t`` before the logarithm, clipping
        negative round-off so ``-log`` is well defined.

    lambda_max : float or None, default=None
        **Currently inert.** ``lambda_max_`` is always computed from the
        dominant eigenpair of ``A`` during fitting; this constructor argument
        is stored but not read. Kept for API stability / future override.

    smooth_entropy : bool, default=False
        **Currently inert.** The VNE selector is presently called with
        ``smooth=False`` hard-coded, so this flag has no effect. Kept for
        future use (smoothing a noisy entropy curve on small/sparse data).

    n_landmarks : int, default=2000
        Target number of landmarks for the landmark approximation. Effective
        when ``use_landmarks`` resolves to True.

    use_landmarks : bool or None, default=None
        Landmark control. If None, landmarks are enabled automatically when
        ``0 < n_landmarks < n_samples`` and disabled otherwise. Set True/False
        to override. Landmarks act as a regularizer in addition to providing
        speedup; on biological benchmarks the landmark variant matches or
        exceeds the full-rank variant on DEMaP (see paper).
    
    landmarks_method : {"kmeans", "kmeans_einsum", "fps", "hybrid", "random"}, default="kmeans"
        Landmark selection algorithm. The ``"kmeans"`` variants use cluster
        centers; ``"fps"`` (farthest-point sampling) is deterministic given a
        seed and produces well-distributed landmarks; ``"random"`` is the
        baseline.

    required_landmarks : array-like of int or None, default=None
        Optional point indices forced to be landmarks. They are unioned with
        the selected landmark set, guaranteeing specific points (e.g. known
        roots/terminals) survive into the landmark MDS.

    projection_method : {"improved_adaptive", "simple"}, default="improved_adaptive"
        Out-of-sample projection of non-landmark points after landmark MDS.
        ``"improved_adaptive"`` uses per-point bandwidths; ``"simple"`` uses
        uniform projection.

    k_project : int or "auto", default=50
        Number of nearest landmarks used to project each non-landmark point.
        ``"auto"`` scales with ``n_samples`` / ``n_landmarks`` (see
        ``_auto_k_proj``); the resolved value is stored in ``k_project_``.

    projection_bandwidth_exponent : float or None, default=None
        Optional exponent on the projection bandwidth
        ``exp(-d^2 / (sigma_i * sigma_j) ** projection_bandwidth_exponent)``.
        None disables it (equivalent to 1, plain product bandwidth).
        Experimental.

    decay : float, default=40
        Decay parameter for the ``alpha_decay`` kernel (higher = sharper local
        kernel). Used only when ``kernel="alpha_decay"``; ignored for
        ``"gaussian"`` and ``"cauchy"``.

    cauchy_decay : float, default=40
        Decay parameter for the ``cauchy`` kernel. Used only when
        ``kernel="cauchy"``.

    root_cells : ndarray of int or None, default=None
        Optional indices of root cells for pseudotime ordering. If None,
        pseudotime is computed on demand via ``_order_cells(root_cells=...)``.

    terminal_cells : ndarray of int or None, default=None
        Optional indices of terminal cells, used as endpoint priors in the
        pseudotime terminal correction.

    plot : bool, default=False
        If True, generate diagnostic plots during fitting (e.g. the VNE
        curve). For interactive use only.

    verbose : int, default=1
        Verbosity: 0 silent, 1 per-step timing and high-level messages,
        2 debug-level detail.

    logger : logging.Logger or None, default=None
        Custom logger. If None, uses the module-level logger.

    random_state : int or None, default=None
        Random seed. Affects PCA preprocessing, SMACOF/SGD initialization,
        and stochastic landmark selection.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The fitted low-dimensional embedding.

    distances_ : ndarray of shape (n_samples, n_samples)
        The EntroPath dissimilarity matrix ``sqrt(-log(A_tilde ** t_power_))``
        (full-rank path). This is the distance matrix consumed by external
        distance-fidelity evaluation. The landmark path stores the
        landmark-space counterpart in ``distances_lm_`` instead.

    A_tilde_ : ndarray
        Spectrally normalized affinity ``A / lambda_max_`` (full-rank path).

    A_tilde_k_ : ndarray
        The diffusion operator ``A_tilde ** t_power_`` before the ``-log``
        (full-rank path); landmark counterpart: ``A_tilde_k_lm_``.

    T_ : ndarray
        MERW transition matrix used for VNE power selection (full-rank path);
        landmark counterpart: ``T_lm_``.

    v_ : ndarray
        Dominant (Perron) eigenvector of ``A`` from the MERW construction;
        the MERW stationary distribution satisfies ``pi_i ~ v_i ** 2``.

    lambda_max_ : float
        Top eigenvalue of ``A`` used in the spectral rescaling.

    t_power_ : int
        Resolved diffusion power.

    t_max_ : int
        Resolved upper bound for the diffusion-power sweep.

    mds_solver_ : str
        MDS solver actually used (after ``"auto"`` resolution).

    pca_ : sklearn.decomposition.PCA or None
        Fitted PCA model used for preprocessing, or None if PCA was not
        applied.

    k_neighbors_ : int
        Resolved number of neighbors used for the affinity graph.

    use_landmarks_ : bool
        Whether landmarks were used in this fit.

    n_landmarks_ : int
        Effective number of landmarks used (0 if landmarks disabled).

    landmark_idx_ : ndarray of int or None
        Indices of selected landmarks (None if landmarks disabled).

    k_project_ : int
        Resolved number of landmarks used for projecting non-landmark points
        (set only when landmarks are used).

    Pnm : ndarray of shape (n_samples, n_landmarks)
        Row-stochastic point->landmark projection operator (landmark path).

    diff_time_ : ndarray
        Pseudotime, set by ``_order_cells``.

    labels_ : ndarray
        Cluster labels, set by ``_cluster_cells``.

    Examples
    --------
    >>> from entropath import EntroPath
    >>> import numpy as np
    >>> X = np.random.randn(1000, 30)
    >>> model = EntroPath(k_neighbors=15, random_state=42)
    >>> Y = model.fit_transform(X)
    >>> Y.shape
    (1000, 2)
    >>> D = model.distances_          # EntroPath dissimilarity matrix
    >>> t = model.t_power_            # VNE-selected diffusion power

    For large datasets, enable landmarks for both speed and regularization:

    >>> model = EntroPath(
    ...     k_neighbors=15,
    ...     use_landmarks=True,
    ...     n_landmarks=2000,
    ...     mds_solver="smacof",
    ...     random_state=42,
    ... )
    >>> Y = model.fit_transform(X_large)

    Notes
    -----
    The algorithm proceeds in five stages:

    1. Optional PCA preprocessing (when input is high-dimensional).
    2. kNN affinity graph ``A`` with the chosen kernel.
    3. Spectral rescaling ``A_tilde = A / lambda_max`` (top eigenvalue of
       ``A``); the maximum-entropy normalization of Burda et al. [1]_, which
       keeps ``A_tilde ** t`` bounded in [0, 1].
    4. Log-transition dissimilarity ``D_sq = -log(A_tilde ** t)`` with ``t``
       chosen from the VNE knee of the MERW transition matrix. The MDS target
       is ``D_sq`` (squared form) or ``sqrt(D_sq)`` (linear/Varadhan form);
       ``distances_`` always stores ``sqrt(D_sq)``.
    5. For ``n_samples > n_landmarks``: landmark MDS on a subset, then
       back-projection of the remaining points.

    The MERW transition matrix is used *only* to select the diffusion power;
    the embedding kernel itself is the rescaled affinity ``A_tilde ** t``.
    This separation lets ``t`` be tuned by the information-theoretic entropy
    criterion while keeping the embedding's interpretation tied to the kNN
    graph rather than the random-walk operator.

    References
    ----------
    .. [1] Burda, Z., Duda, J., Luck, J.M., Waclaw, B. (2009).
           "Localization of the Maximal Entropy Random Walk."
           Physical Review Letters, 102(16), 160602.

    .. [2] Moon, K.R., et al. (2019). "Visualizing structure and transitions
           in high-dimensional biological data." Nature Biotechnology,
           37(12), 1482-1492.

    .. [3] Huguet, G., et al. (2024). "A Heat Diffusion Perspective on
           Geodesic Preserving Dimensionality Reduction." NeurIPS.

    .. [4] Wei, M., et al. (2025). "Diffusion-based Trajectory and Nonlinear
           Embedding (DTNE) for single-cell data." [Update venue per actual
           publication.]

    .. [5] Varadhan, S.R.S. (1967). "On the behavior of the fundamental
           solution of the heat equation with variable coefficients."
           Communications on Pure and Applied Mathematics, 20(2), 431-455.

    See Also
    --------
    BaseEmbedding : Parent class providing the fit_transform interface.
    """

    def __init__(
        self,
        n_components: int = 2,
        k_neighbors: int = 15,  # "auto" for Zelnik-Manor style adaptive k selection
        t_power: int | str = "auto",
        metric: str = "precomputed",
        kernel: str = "gaussian",
        mds_solver: str = "auto",
        mds_dissimilarity: str = "squared", #linear, squared
        max_iter: int = 3000,
        n_pca: int | None = 100,  # None to disable PCA preprocessing, or positive integer for number of PCA dimensions
        kernel_norm: bool = False,
        sym_mode: str = "max",
        t_max: int | str = "auto",
        epsilon: float = 1e-21,
        lambda_max: float | None = None,
        smooth_entropy: bool = False,
        n_landmarks: int = 2000,
        use_landmarks: bool | None = None,
        landmarks_method: str = "kmeans", # kmeans, fps, hybrid, random
        required_landmarks: ArrayLike | np.ndarray | None = None,
        projection_method: str = "improved_adaptive",
        k_project: int | str = 50, # auto
        projection_bandwidth_exponent: float | None = None,
        decay: float = 40,
        cauchy_decay: float = 40,
        root_cells: np.ndarray | None = None,
        terminal_cells: np.ndarray | None = None,
        plot: bool = False,
        verbose: int = 1,   # 0 silent, 1 normal, 2 debug
        logger: logging.Logger | None = None,
        random_state: int | None = None,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.k_neighbors = k_neighbors
        self.t_power = t_power
        self.metric = metric
        self.kernel = kernel
        self.mds_solver = mds_solver
        self.mds_dissimilarity = mds_dissimilarity
        self.max_iter = max_iter
        self.n_pca = n_pca
        self.kernel_norm = kernel_norm
        self.sym_mode = sym_mode
        self.t_max = t_max
        self.epsilon = epsilon
        self.lambda_max = lambda_max
        self.smooth_entropy = smooth_entropy
        self.n_landmarks = n_landmarks
        self.use_landmarks = use_landmarks
        self.required_landmarks = required_landmarks
        self.projection_method = projection_method
        self.k_project = k_project
        self.landmarks_method = landmarks_method
        self.projection_bandwidth_exponent = projection_bandwidth_exponent
        self.decay = decay
        self.cauchy_decay = cauchy_decay
        self.root_cells = root_cells
        self.terminal_cells = terminal_cells
        self.plot = plot
        self.verbose = verbose
        self.logger = logger

    def get_params(self, deep=True):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.endswith("_") and k != "logger"
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _validate_params(self):
        # String-enum parameters
        _VALID = {
            "mds_solver":        {"auto", "smacof", "sgd", "classic",
                                  "classic_mds_robust", "classic_pca", "sklearn"},
            "mds_dissimilarity": {"squared", "linear"},
            "kernel":             {"gaussian", "alpha_decay", "cauchy"},
            "sym_mode":           {"max", "avg"},
            "landmarks_method":   {"kmeans", "kmeans_einsum", "fps",
                                  "hybrid", "random"},
            "projection_method":  {"simple", "improved_adaptive"},
        }
        for name, allowed in _VALID.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {sorted(allowed)}, got {value!r}"
                )

        # Positive integers
        for name in ("n_components", "max_iter", "n_landmarks"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        # "auto" or positive integer
        for name in ("k_neighbors", "t_power", "t_max", "k_project"):
            value = getattr(self, name)
            if value == "auto":
                continue
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{name} must be 'auto' or a positive integer, got {value!r}"
                )

        # Positive float
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.decay <= 0:
            raise ValueError(f"decay must be > 0, got {self.decay}")
        if self.cauchy_decay <= 0:
            raise ValueError(f"cauchy_decay must be > 0, got {self.cauchy_decay}")


    @property
    def kernel_params(self):
        return {
            "gaussian__normalize": self.kernel_norm,
            "gaussian__sym_mode": self.sym_mode,
            "alpha_decay__decay": self.decay,
            "cauchy__decay": self.cauchy_decay,
        }



    def fit(self, X):
        """Fit the EntroPath model on input data X.

        Constructs the kNN affinity graph, MERW transition matrix,
        diffusion power, free-energy dissimilarity matrix, and metric MDS
        embedding. If `use_landmarks=True` (explicit or auto), uses
        a landmark approximation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data. If `X.shape[1] > self.n_pca` and
            `self.n_pca is not None`, PCA is applied as a
            preprocessing step.

        Returns
        -------
        self : EntroPath
            Fitted estimator. The embedding is stored in
            `self.embedding_`.

        """
        self._validate_params()
        if self.verbose <= 0:
            logger.setLevel(logging.WARNING)
        n_samples = X.shape[0]

        self._log(f"Running EntroPath on {n_samples} observations and {X.shape[1]} variables.", indent=0)
        with timed_step("EntroPath", indent_level=0, logger=logger, verbose=self.verbose):
            # ------------------------------------------------------------------
            # 1. Decide landmark strategy (explicit > implicit)
            # ------------------------------------------------------------------
            if self.use_landmarks is not None:
                # Explicit override from user
                use_landmarks = self.use_landmarks

                if not use_landmarks:
                    n_landmarks_effective = None
                    self._log("Landmarks disabled", indent=1)
                else:
                    if self.n_landmarks is None:
                        n_landmarks_effective = min(2000, n_samples)
                    else:
                        n_landmarks_effective = min(self.n_landmarks, n_samples)
                    self._log(
                        f"Using landmarks: method={self.landmarks_method}, "
                        f"n_landmarks={n_landmarks_effective}",
                        indent=2,
                    )

            else:
                # Automatic behavior
                if self.n_landmarks is None:
                    use_landmarks = False
                    n_landmarks_effective = None
                elif self.n_landmarks <= 0 or self.n_landmarks >= n_samples:
                    use_landmarks = False
                    n_landmarks_effective = None
                else:
                    use_landmarks = True
                    n_landmarks_effective = self.n_landmarks
                    self._log(
                        f"Using landmarks: method={self.landmarks_method}, "
                        f"n_landmarks={n_landmarks_effective}",
                        indent=1,
                    )

            # ------------------------------------------------------------------

            self.k_neighbors_ = self.k_neighbors

            if self.t_max == "auto":
                n_graph_nodes = n_landmarks_effective if use_landmarks else X.shape[0]
                self.t_max_ = self._auto_t_max()
            else:
                self.t_max_ = self.t_max

            # --- PCA preprocessing ---
            if self.n_pca is not None and X.shape[1] > self.n_pca:
                pca = PCA(n_components=self.n_pca, random_state=self.random_state)
                X_graph = pca.fit_transform(X)
                self.pca_ = pca
                self._log(
                    f"Preprocessing PCA: {X.shape[1]} -> {self.n_pca} dimensions "
                    f"(set n_pca=None to disable)",
                    indent=1,
                )
            else:
                X_graph = X #.copy()  #safer than in-place modification
                self.pca_ = None
        
            # Select landmarks (ONLY if enabled)
            if use_landmarks:
                if self.k_project == "auto":
                    self.k_project_ = self._auto_k_proj(
                        n_samples=n_samples,
                        n_landmarks=n_landmarks_effective,
                    )
                else:
                    self.k_project_ = self.k_project

                landmark_idx = self._select_landmarks(
                    X_graph,
                    n_landmarks=n_landmarks_effective
                )
                if self.required_landmarks is not None:
                    forced = np.atleast_1d(np.asarray(self.required_landmarks, dtype=int))
                    landmark_idx = np.union1d(landmark_idx, forced)
            else:
                landmark_idx = None

            # Store state
            self.use_landmarks_ = use_landmarks
            self.n_landmarks_ = len(landmark_idx) if landmark_idx is not None else 0
            self.landmark_idx_ = landmark_idx

            # Run embedding pipeline
            if use_landmarks:
                Z = self._fit_landmark_pipeline_simple(X_graph, landmark_idx, use_landmarks = True)
            else:
                Z = self._fit_full_pipeline(X_graph, use_landmarks = False)

            self.embedding_ = Z

        return self


    def _fit_full_pipeline(self, X, use_landmarks=False):
        with timed_step("affinities", indent_level=1, logger=logger, verbose=self.verbose):
            A, _, _ = self._build_affinity(X=X, k=self.k_neighbors_, **self._get_kernel_params())
        
        with timed_step("MERW transition matrix", indent_level=1, logger=logger, verbose=self.verbose):
            T = self._merw_transition_matrix(A, tol=0)[0] #1e-9
        self.T_ = T
        A_tilde = A / self.lambda_max_
        self.A_tilde_ = A_tilde
        with timed_step("optimal diffusion power", indent_level=1, logger=logger, verbose=self.verbose):
            self.t_power_ = self._select_power(X, T)
        Z, self.A_tilde_k_, self.distances_ = self._fit_full_diffusion(A_tilde, use_landmarks=use_landmarks)
        return Z

    def _fit_landmark_pipeline_simple(self, X, landmark_idx, use_landmarks=True):
        X_lm = X[landmark_idx]
        logger.debug(f"X_lm shape after selection: {X_lm.shape}")

        # --- graph only on landmarks ---
        with timed_step("affinities", indent_level=1, logger=logger, verbose=self.verbose):            
            A_lm, _, _ = self._build_affinity(X=X_lm, k=self.k_neighbors_, **self._get_kernel_params())
        logger.debug(f"A_lm shape after affinity: {A_lm.shape}")
        with timed_step("MERW transition matrix", indent_level=1, logger=logger, verbose=self.verbose):
            T_lm = self._merw_transition_matrix(A_lm, tol=0)[0] # it was 1e-9
        # store MERW on landmarks
        self.T_lm_ = T_lm
        A_lm_tilde = A_lm / self.lambda_max_
        self.A_lm_tilde_ = A_lm_tilde
        logger.debug(f"T_lm final shape before VNE: {T_lm.shape}")

        # select diffusion power on landmarks
        with timed_step("optimal diffusion power", indent_level=1, logger=logger, verbose=self.verbose):
            self.t_power_ = self._select_power(X_lm, T_lm)

        # run standard embedding
        Z_lm, self.A_tilde_k_lm_, self.distances_lm_ = self._fit_full_diffusion(A_lm_tilde, use_landmarks=use_landmarks) ####test A_lm_tilde it was T_lm

        # project all points
        with timed_step("projecting points", indent_level=1, logger=logger, verbose=self.verbose):
            Z, weights, idx = self._project_points(X, X_lm, Z_lm, method=self.projection_method, k_neighbors=self.k_neighbors_, projection_bandwidth_exponent=self.projection_bandwidth_exponent, k_project=self.k_project_)

        self.landmarks_ = landmark_idx
        self.landmark_embedding_ = Z_lm

        self.Pnm = self._compute_Pnm(weights, idx, X.shape[0], X_lm.shape[0])

        return Z


    def _spectral_coords(self, A, n_svd=100):
        """Top-n_svd spectral coordinates of the symmetric affinity A.

        PHATE-style: eigen-decomposition of the diffusion operator, scaled by
        eigenvalues so Euclidean distance approximates diffusion distance.
        Clustering happens in this space (sparse regions are contracted here,
        so k-means is far less likely to carve out tiny sparse clusters).
        """
        n = A.shape[0]
        k = min(n_svd, n - 1)
        # A is symmetric -> eigsh is appropriate and fast on the sparse kernel.
        vals, vecs = eigsh(A, k=k, which="LA")
        # sort descending (eigsh returns ascending)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # spectral coordinates: eigenvectors scaled by eigenvalues
        coords = vecs * np.maximum(vals, 0.0)
        return coords
    

    def _project_points(self, X, X_lm, Z_lm, method="improved_adaptive", **kwargs):
        """
        General method to project points using different strategies.
    
        Parameters:
            X, X_lm, Z_lm: data arrays
            method: str, selects which projection method to use
            kwargs: additional parameters for specific methods
        Returns:
            Z, weights, idx: projected coordinates, interpolation weights, and landmark indices
        """
        if method == "simple":
            return project_points_simple(X, X_lm, Z_lm, **kwargs)
        elif method == "improved_adaptive":
            return project_points_improved_adaptive(X, X_lm, Z_lm, **kwargs)
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
    
    def _compute_Pnm(self, weights, idx, n_samples, n_landmarks):
        """
        * Pnm​(i,j)=probability that point i connects to landmark j
        * Pnm = weights from projection step
        """
        with timed_step("Pnm operator", indent_level=1, logger=logger, verbose=self.verbose):
            Pnm = np.zeros((n_samples, n_landmarks))
            rows = np.arange(n_samples)[:, None]

            Pnm[rows, idx] = weights

            return Pnm


    def _fit_full_diffusion(self, A_tilde, use_landmarks=None):
        with timed_step("fitting full diffusion", indent_level=1, logger=logger, verbose=self.verbose):
            with timed_step("diffusion power", indent_level=2, logger=logger, verbose=self.verbose):
                A_tilde_k = _compute_power(A_tilde, self.t_power_)

            A_tilde_k_clipped = np.clip(A_tilde_k, self.epsilon, None)   # negative roundoff -> 1e-21, then log fine
            D_sq = -np.log(A_tilde_k_clipped)
            np.fill_diagonal(D_sq, 0)
            # drop 4k from the distance scaling, as it is invariant to the MDS solution and can be absorbed into the MDS stress scaling.
            # This also makes the distances more interpretable and comparable across different t_power values.
            D = np.sqrt(D_sq)

            target_dissimilarity = D_sq if self.mds_dissimilarity == "squared" else D
            Z = self._embed_mds(target_dissimilarity, use_landmarks=use_landmarks)

            return Z, A_tilde_k, D
    

    def _select_landmarks(self, X, n_landmarks=None):

        if self.n_landmarks is None:
            return None

        n_landmarks = n_landmarks or self.n_landmarks

        methods = {
            "kmeans": select_landmarks_kmeans,
            "kmeans_einsum": select_landmarks_kmeans_einsum,
            "fps": select_landmarks_fps,
            "hybrid": select_landmarks_hybrid,
            "random": select_landmarks_random,
        }

        if self.landmarks_method not in methods:
            raise ValueError(f"Unknown landmark method: {self.landmarks_method}")

        with timed_step(
            f"Selecting landmarks ({self.landmarks_method})",
            indent_level=1,
            logger=logger,
            verbose=self.verbose,
        ):
            return methods[self.landmarks_method](
                X,
                n_landmarks=n_landmarks,
                random_state=self.random_state,
            )


    def _get_kernel_params(self):
        params = {}

        for key, value in self.kernel_params.items():
            if "__" in key:
                kernel_name, param = key.split("__", 1)
                if kernel_name == self.kernel:
                    params[param] = value
            else:
                params[key] = value

        return params


    def _build_affinity(self, X, k, **extra_kwargs):
        """Unified method that selects and runs the chosen kernel."""
        try:
            func = KERNEL_REGISTRY[self.kernel]
        except KeyError:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Available: {list(KERNEL_REGISTRY.keys())}"
            )
        
        params = {
        **self._get_kernel_params(),
        **extra_kwargs,
        }

        return func(X=X, k=k, **params)


    def _log(self, msg, level="info", indent=0):
        if self.verbose <= 0:
            return

        if self.logger is None:
            print("  " * indent + msg)
        else:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(msg, extra={"indent": indent})

    
    def _merw_transition(self, A):
        eigvals, eigvecs = eigsh(A, k=1, which='LA')
        lambda_max = max(eigvals[0], 1e-12)

        psi = np.abs(eigvecs[:, 0])
        psi = np.maximum(psi, 1e-12)

        # MERW transition matrix
        T = (A / lambda_max) * (psi[np.newaxis, :] / psi[:, np.newaxis])
        T = T / T.sum(axis=1, keepdims=True)
        self.lambda_max_ = lambda_max
        return T


    def _merw_transition_sparse(self, A):

        eigvals, eigvecs = eigsh(A, k=1, which='LA')

        lambda_max = max(eigvals[0], 1e-12)

        psi = np.abs(eigvecs[:, 0])
        psi = np.maximum(psi, 1e-12)

        # MERW formula
        ratio = psi[np.newaxis, :] / psi[:, np.newaxis]

        if sparse.issparse(A):
            T = A.multiply(ratio) / lambda_max
        else:
            T = (A / lambda_max) * ratio

        # --- row normalization (sparse safe) ---
        row_sums = np.array(T.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1e-12

        if sparse.issparse(T):
            T = sparse.diags(1 / row_sums) @ T
        else:
            T = T / row_sums[:, None]
        self.lambda_max_ = lambda_max
        return T, psi


    def _merw_transition_matrix(self, A, tol=0): #1e-9
        """
        Computes MERW (maximal entropy random walk) transition matrix.
        Returns sparse or dense matrix depending on input A.
        """
        if not sparse.issparse(A):
            A = np.asarray(A)

        # Dominant eigenpair (largest algebraic eigenvalue)
        vals, vecs = eigsh(A, k=1, which='LA', tol=tol)
        lambda_max = vals[0]                # should be > 0 for connected non-neg. A

        v = np.abs(vecs[:, 0])
        v = np.maximum(v, 1e-15)

        # equivalent to A.multiply(ratio)/lambda_max; this form is slightly more stable
    
        vj = v[None, :]
        vi = v[:, None]
    
        if sparse.issparse(A):
            T = A.multiply(vj) / (lambda_max * vi)
        else:
            T = (A * vj) / (lambda_max * vi)

        # Clean any numerical garbage
        T.data[np.isnan(T.data)] = 0
        T.data[np.isinf(T.data)] = 0

        # Row normalization (sparse & dense safe)
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        row_sums = np.maximum(row_sums, 1e-12)

        if sparse.issparse(T):
            T = sparse.diags(1.0 / row_sums) @ T
        else:
            T /= row_sums[:, None]
        
        logger.debug(f"MERW diffusion final shape before VNE: {T.shape}")
        self.lambda_max_ = lambda_max
        self.v_ = v

        return T, lambda_max, v
    

    def _get_T_k(self, use_landmark=False):
        if use_landmark:
            cache_attr, psi_attr, A_tilde_k_attr, T_attr = (
                "T_lm_k_", "psi_lm_", "A_tilde_lm_k_", "T_lm_"
            )
        else:
            cache_attr, psi_attr, A_tilde_k_attr, T_attr = (
                "T_k_", "psi_", "A_tilde_k_", "T_"
            )

        # Check cache
        result = getattr(self, cache_attr, None)
        if result is not None:
            return result

        # Try spectral reconstruction (cheap path)
        psi = getattr(self, psi_attr, None)
        A_tilde_k = getattr(self, A_tilde_k_attr, None)
        if psi is not None and A_tilde_k is not None:
            result = (A_tilde_k * psi[None, :]) / psi[:, None]
        else:
            # Fall back to matrix power (expensive path)
            T = getattr(self, T_attr, None)
            if T is None:
                raise ValueError(f"Cannot compute T^k: neither {A_tilde_k_attr} "
                                f"nor {T_attr} is available.")
            result = _compute_power(T, self.t_power_)

        setattr(self, cache_attr, result)
        return result


    def _sparse_matrix_power(self, T, k):
        """
        Fast sparse matrix power using exponentiation by squaring.
        Keeps sparsity as long as possible.
        """
        if k == 0:
            return sparse.eye(T.shape[0], format=T.format, dtype=T.dtype)
        if k == 1:
            return T.copy()

        # Binary exponentiation
        result = sparse.eye(T.shape[0], format=T.format, dtype=T.dtype)

        while k > 0:
            if k % 2 == 1:
                result = result @ T
            T = T @ T
            k //= 2

        return result


    def _select_power(self, X, diff_op):

        # user provided fixed power
        if isinstance(self.t_power, int):
            self._log(f"Using user-provided diffusion power: {self.t_power}", indent=1)
            return self.t_power

        method = self.t_power

        if method == "auto":
            method = self._auto_power_method() #diff_op

        if method == "von_neumann_entropy":
            t_opt, _, _ = von_neumann_entropy(
                diff_op,
                t_max=self.t_max_,
                smooth=False,
                plot=self.plot,
                logger=logger,
            )
            if t_opt >= 0.9 * self.t_max_:
                self._log(
                    f"Optimal t={t_opt} is near t_max={self.t_max_}. "
                    f"Consider increasing t_max or n_landmarks for better selection.",
                    level="warning",
                    indent=2,
                )
            return t_opt
        else:
            raise ValueError(f"Unknown t_power method: {method}. "
                             "Only 'auto', 'von_neumann_entropy', or an integer are supported.")
    
    
    def _auto_k_proj(self, n_samples, n_landmarks):
        """
        Automatic selection of projection k (nearest landmarks for out-of-sample).
        Only called when use_landmarks=True.

        Larger k → smoother embedding (recommended for high diffusion t)
        Smaller k → more local / faithful to fine structure

        TODO: improve scaling relative to n_landmarks/n_samples ratio
        """
        if n_samples < 5_000:
            k = 30
        elif n_samples > 50_000:
            k = 100
        elif n_samples > 20_000:
            k = 75
        else:
            k = 50

        # Safety cap — avoid requesting more neighbours than landmarks allow
        # Inactive at default n_landmarks=2000 but protects small landmark counts
        k = min(k, max(30, n_landmarks // 10))

        return k

    def _auto_t_max(self):
        return 150

    def _auto_power_method(self):
        return "von_neumann_entropy"
    
    def _embed_mds(self, D, init=None, use_landmarks=None):

        n = D.shape[0]
        solver = self.mds_solver

        # --- automatic solver selection ---
        if solver == "auto":
            if use_landmarks:
                solver = "smacof"  # more stable for smaller landmark sets
            else:
                # Full dataset
                if n <= 5000: # small datasets
                    solver = "smacof"
                else:
                    solver = "sgd"
                    if n > 10000 and self.verbose:
                        self._log(
                            f"Running SGD on full n={n} without landmarks. "
                            f"This requires the full {n}x{n} distance matrix in memory. "
                            f"Consider use_landmarks=True for n > 10000.",
                            level="warning",
                            indent=2,
                        )
        
        # store chosen solver
        self.mds_solver_ = solver

        with timed_step(f"metric MDS ({solver}) - precomputed dissimilarity", indent_level=2, logger=logger, verbose=self.verbose):
            # --- classic MDS ---
            if solver == "classic":

                Z = classic(
                    D,
                    n_components=self.n_components,
                    random_state=self.random_state,
                )

            # --- SGD MDS (PHATE style) ---
            elif solver == "sgd":

                if init is None:
                    init_Y = classic_pca(D, n_components=self.n_components, random_state=self.random_state,) #_mds_robust
                else:
                    init_Y = init

                Z = sgd_mds(
                    D,
                    n_components=self.n_components,
                    random_state=self.random_state,
                    init=init_Y,
                    verbose=0,
                )

            # --- SMACOF ---
            elif solver == "smacof":

                if init is None:
                    init_Y = classic_mds_robust(D, n_components=self.n_components, random_state=self.random_state,)

                else:
                    init_Y = init

                Z = smacof(
                    D,
                    n_components=self.n_components,
                    random_state=self.random_state,
                    init=init_Y,
                    max_iter=self.max_iter
                )
            
            elif solver == "classic_mds_robust":
                Z = classic_mds_robust(
                    D,
                    n_components=self.n_components,
                    random_state=self.random_state,
                )
            
            elif solver == "classic_pca":
                Z = classic_pca(
                    D,
                    n_components=self.n_components,
                    random_state=self.random_state,
                )
            
            # --- sklearn MDS ---
            elif solver == "sklearn":

                from sklearn.manifold import MDS

                mds = MDS(
                    n_components=self.n_components,
                    metric=True,
                    dissimilarity=self.metric,
                    random_state=self.random_state,
                    normalized_stress="auto",
                )

                Z = mds.fit_transform(D)

            else:
                raise ValueError(f"Unknown mds_solver: {solver}")

            return Z


    def _normalize_pt(self, values, mode):
        if mode == "minmax":
            return (values - values.min()) / (values.max() - values.min() + 1e-12)
        elif mode == "rank":
            ranks = np.argsort(np.argsort(values))
            return ranks / max(len(values) - 1, 1)
        else:
            raise ValueError(f"normalization must be 'minmax' or 'rank', got {mode!r}")
        
    def _diffusion_dist_from(self, cells, T_k, P_nm=None):
        """Compute -log(diffusion mass) from given cells."""
        if P_nm is not None:
            # Soft (landmark) version
            weights = P_nm[cells]                       # (n_cells, m)
            probs = (weights @ T_k).sum(axis=0)         # (m,)
        else:
            # Hard version
            probs = T_k[cells, :].sum(axis=0)           # (n,)

        probs = np.clip(probs, 1e-12, None)
        return -np.log(probs)


    def _order_cells(self, normalization="minmax", **params):
        """DTNE-style pseudotime via T^k diffusion from root cells."""
        # ---- Parse params ----
        if "root_cells" in params:
            self.root_cells = params["root_cells"]
        elif self.root_cells is None:
            raise ValueError("root_cells must be specified.")

        if "terminal_cells" in params:
            self.terminal_cells = params["terminal_cells"]

        normalization = params.get("normalization", normalization)

        if not hasattr(self, "t_power_") or self.t_power_ is None:
            raise ValueError("t_power_ not found. Run fit_transform first.")

        root_cells = np.asarray(self.root_cells)

        # ---- Detect mode ----
        use_landmark = (
            hasattr(self, "T_lm_") and self.T_lm_ is not None
            and hasattr(self, "Pnm") and self.Pnm is not None
        )

        T_k = self._get_T_k(use_landmark=use_landmark)
        P_nm = self.Pnm if use_landmark else None

        # ---- Root pseudotime ----
        d_root = self._diffusion_dist_from(root_cells, T_k, P_nm=P_nm)
        pt_intermediate = self._normalize_pt(d_root, normalization)

        if use_landmark:
            diff_time = self.Pnm @ pt_intermediate
        else:
            diff_time = pt_intermediate

        # ---- Terminal correction (optional) ----
        if getattr(self, "terminal_cells", None) is not None:
            terminal_cells = np.asarray(self.terminal_cells)
            d_term = self._diffusion_dist_from(terminal_cells, T_k, P_nm=P_nm)
            d_term = self._normalize_pt(d_term, "minmax")  # always minmax for terminal

            if use_landmark:
                d_term = self.Pnm @ d_term

            # Combine: pseudotime is fraction toward terminal
            pt = diff_time / (diff_time + d_term + 1e-12)
            diff_time = self._normalize_pt(pt, normalization)

        self.diff_time_ = diff_time
        return diff_time


    def _cluster_cells(self, **params):
        """
        Cluster cells using precomputed EntroPath distances or pseudotime binning.

        Supports:
            - Distance-based clustering (recommended for manifold structure):
                * "agglo" / "hiera" (default)
                * "dbscan"
            - Trajectory-aware clustering:
                * "pseudotime" (bins self.diff_time_ into equal-width intervals)
            - Embedding-based:
                * "kmeans" (uses self.embedding_)

        Args:
            cluster_method (str, optional): Clustering method. Default: "pseudotime".
            n_clusters (int, optional): Number of clusters (for agglo, kmeans, pseudotime). Default: 8.
            eps (float, optional): DBSCAN epsilon. Default: 0.5.
            min_samples (int, optional): DBSCAN min samples. Default: 5.
            root_cells (list, optional): Only needed for "pseudotime" if self.diff_time_ does not exist yet.

        Returns:
            np.ndarray: Cluster labels (shape: n_cells,)

        Raises:
            ValueError: If required attributes or parameters are missing.
        """

        from sklearn import cluster

        # Parameter parsing with sensible defaults
        cluster_method = params.get("cluster_method", "pseudotime")
        n_clusters = params.get("n_clusters", 8)

        # ------------------------------------------------------------------
        # Special case: pseudotime binning (trajectory-aware)
        # ------------------------------------------------------------------
        if cluster_method == "pseudotime":
            if not hasattr(self, "diff_time_") or self.diff_time_ is None:
                raise ValueError("self.diff_time_ not found. Run _order_cells first.")
            
            # normalize diff_time_ to [0, 1]
            bins = np.linspace(0, 1, n_clusters + 1)
            labels = np.digitize(self.diff_time_, bins) - 1          # 0-based
            labels = np.clip(labels, 0, n_clusters - 1)

            self.labels_ = labels
            return labels

        # ------------------------------------------------------------------
        # All other methods require either distances_ or embedding_
        # ------------------------------------------------------------------
        if cluster_method == "kmeans":
            # kmeans works on the low-dimensional embedding (Euclidean)
            if not hasattr(self, "embedding_") or self.embedding_ is None:
                raise ValueError(
                    "cluster_method='kmeans' requires self.embedding_. "
                    "Run the embedding step first."
                )
            X = self.embedding_
            model = cluster.KMeans(
                n_clusters=n_clusters,
                random_state=getattr(self, "random_state", None),
                n_init="auto"
            )
            labels = model.fit_predict(X)

        else:
            # All other methods use the precomputed EntroPath distance matrix
            if not hasattr(self, "distances_") or self.distances_ is None:
                raise ValueError(
                    f"cluster_method='{cluster_method}' requires self.distances_. "
                    "Make sure the diffusion / embedding pipeline has been run."
                )
            X = self.distances_

            if cluster_method in ["agglo", "hiera"]:
                model = cluster.AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average"
                )
                labels = model.fit_predict(X)

            elif cluster_method == "dbscan":
                eps = params.get("eps", 0.5)
                min_samples = params.get("min_samples", 5)
                model = cluster.DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="precomputed"
                )
                labels = model.fit_predict(X)

            else:
                raise ValueError(
                    f"Unknown cluster_method: {cluster_method}. "
                    "Supported: agglo/hiera, dbscan, kmeans, pseudotime."
                )

        # ------------------------------------------------------------------
        # Store and return
        # ------------------------------------------------------------------
        self.labels_ = labels
        return labels


    def _embedding_diffusion_maps(self, use_landmarks=None):
        """
        Diffusion Maps embedding using the transition matrix.

        Parameters
        ----------
        use_landmarks : bool or None
            If True, use landmark operator (T_lm_ + Pnm)
            If False, use full T_
            If None, auto-detect

        Returns
        -------
        Z : array (n_samples, n_components)
        """
        from sklearn.manifold import SpectralEmbedding

        if use_landmarks is None:
            use_landmarks = hasattr(self, "T_lm_") and self.T_lm_ is not None

        if use_landmarks:
            # embedding on landmarks
            Z_lm = SpectralEmbedding(
                n_components=self.n_components,
                affinity="precomputed"
            ).fit_transform(self.T_lm_)

            # project back
            Z = self.Pnm @ Z_lm

        else:
            Z = SpectralEmbedding(
                n_components=self.n_components,
                affinity="precomputed"
            ).fit_transform(self.T_)

        return Z
    

    def _embedding_merw_limit_fast(self, X=None, centered=True):
        """
        avoids explicit pi computation by using the fact that for MERW, the stationary distribution pi is proportional to the square of the dominant eigenvector v of the adjacency matrix A (pi_i ~ v_i^2).
        This allows us to compute the embedding directly from v without needing to compute T or its powers, which can be expensive for large datasets.
        Uses: pi_i ~ v_i^2
        """
        if not hasattr(self, "v_") or self.v_ is None:
            A, _, _ = self._build_affinity(X=X, k=self.k_neighbors_, **self._get_kernel_params())
            T = self._merw_transition_matrix(self.A_, tol=0)[0]
        v = self.v_

        Z = -2.0 * np.log(v)

        if centered:
            Z = Z - Z.mean()

        return Z[:, None]




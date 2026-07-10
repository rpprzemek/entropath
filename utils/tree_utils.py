import numpy as np
import phate
from utils.helpers import compute_shortest_path_geodesic


# ── Data generation ─────────────────────────────────────────────────────

def make_artificial_tree(
    n_branch=6,
    branch_length=500,
    n_dim=10,
    sigma=2.0,
    k_geo=15,
    random_state=42,
):
    """PHATE-style synthetic branching tree with kNN-graph geodesic ground truth.

    Parameters
    ----------
    n_branch      : number of branches in the tree
    branch_length : points per branch
    n_dim         : ambient dimensionality
    sigma         : Gaussian noise std added on top of the clean tree
    k_geo         : k for the kNN graph used to compute ground-truth
                    geodesic distances (on clean coordinates)
    random_state  : RNG seed

    Returns
    -------
    X_noisy : (n, n_dim) ndarray
        Noisy tree, used as input to all methods.
    X_clean : (n, n_dim) ndarray
        Clean reference points (noise-free), for DEMaP evaluation.
    labels  : (n,) ndarray
        Integer branch index, 0..n_branch-1.
    D_geo   : (n, n) ndarray
        Pairwise shortest-path geodesic distances on the clean kNN graph.
    """
    rng = np.random.RandomState(random_state)

    # Clean latent tree (no observation noise)
    X_clean, _ = phate.tree.gen_dla(
        n_branch=n_branch,
        branch_length=branch_length,
        n_dim=n_dim,
        sigma=0,
        seed=random_state,
    )

    # Noisy observations (what algorithms see)
    X_noisy = X_clean + rng.normal(0, sigma, X_clean.shape)

    # Ground-truth geodesic via the universal helper
    D_geo = compute_shortest_path_geodesic(
        X_clean, k_geo=k_geo, seed=random_state,
    )

    labels = np.repeat(np.arange(n_branch), branch_length)
    assert labels.shape[0] == X_noisy.shape[0]

    return X_noisy, X_clean, labels, D_geo

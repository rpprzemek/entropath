import numpy as np
from utils.helpers import compute_shortest_path_geodesic



# ── Data generation ───────────────────────────────────────────────────────────

def make_torus(n_samples=2000, R=3.0, r=1.0, noise=0.0, random_state=None):
    """Uniform torus embedding in R^3.

    Parameters
    ----------
    n_samples : int
        Number of points.
    R : float
        Major radius (distance from torus center to tube center).
    r : float
        Minor radius (tube radius). Standard donut shape requires R > r.
    noise : float
        Gaussian noise std added to ambient coordinates. 0 returns clean torus.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : (n, 3) ndarray
        Torus points with optional Gaussian noise added.
    X_clean : (n, 3) ndarray
        Underlying noise-free points exactly on the torus surface.
    theta : (n,) ndarray
        Major angle in [0, 2π).
    phi : (n,) ndarray
        Minor angle in [0, 2π).
    """
    rng = np.random.default_rng(random_state)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi   = rng.uniform(0, 2 * np.pi, n_samples)

    # Clean points on the torus surface
    X_clean = np.stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ], axis=1)

    # Noisy version (same underlying points)
    if noise > 0:
        X = X_clean + rng.normal(0, noise, X_clean.shape)
    else:
        X = X_clean.copy()

    return X, X_clean, theta, phi


def make_torus_with_shortest_path_geodesic(n_samples=2000, R=3.0, r=1.0,
                                            noise=0.05, random_state=42,
                                            k_geo=15):
    """Torus with shortest-path geodesic ground truth (DEMaP convention).

    The embedded torus in R^3 has no closed-form Riemannian geodesic, so
    shortest path on a kNN graph constructed from clean coordinates serves
    as the ground truth.

    Returns
    -------
    X : (n, 3) ndarray
        Noisy torus points.
    t : (n,) ndarray
        Major angle theta (used for coloring in plots).
    D_geo : (n, n) ndarray
        Pairwise shortest-path geodesic distances on clean data.
    X_clean : (n, 3) ndarray
        Noise-free reference points (for DEMaP).
    """
    X, X_clean, theta, phi = make_torus(
        n_samples=n_samples, R=R, r=r,
        noise=noise, random_state=random_state,
    )
    D_geo = compute_shortest_path_geodesic(
        X_clean, k_geo=k_geo, seed=random_state,
    )
    return X, theta, D_geo, X_clean


# ── Non-uniform torus ───────────────────────────────────────────────────

def make_non_uniform_torus(n_samples=2000, R=3.0, r=1.0, noise=0.05,
                          concentration=4.0, random_state=42):
    """
    Non-uniform torus: Beta(1, concentration) density in theta.
    """
    rng   = np.random.RandomState(random_state)
    n_big = n_samples * 10

    theta = rng.uniform(0, 2 * np.pi, n_big)
    phi   = rng.uniform(0, 2 * np.pi, n_big)

    X_big = np.stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi)
    ], axis=1)
    X_big += rng.normal(0, noise, X_big.shape)

    theta_norm = theta / (2 * np.pi)
    weights    = (1 - theta_norm) ** (concentration - 1)
    weights    = np.clip(weights, 1e-8, None)
    weights   /= weights.sum()

    chosen = rng.choice(n_big, size=n_samples, replace=False, p=weights)
    return X_big[chosen], theta[chosen], phi[chosen]


# ── Geodesic distance approximation ─────────────────────────────────────

def torus_geodesic_approx(theta, phi, R=3.0, r=1.0):
    """
    Approximate torus geodesic using midpoint Riemannian metric.

    True metric:
        ds² = (R + r cosφ)² dθ² + r² dφ²

    We approximate using midpoint φ between pairs (with circular mean).

    Note: this is still an approximation (true geodesics are curved),
    but significantly better than flat parameter distance.
    """
    def _circular(a):
        diff = np.abs(np.subtract.outer(a, a))
        return np.minimum(diff, 2 * np.pi - diff)

    d_theta = _circular(theta)
    d_phi   = _circular(phi)

    # circular midpoint (important fix!)
    phi_mid = np.arctan2(
        np.sin(phi[:, None]) + np.sin(phi[None, :]),
        np.cos(phi[:, None]) + np.cos(phi[None, :])
    )

    R_eff = R + r * np.cos(phi_mid)

    return np.sqrt((R_eff * d_theta) ** 2 + (r * d_phi) ** 2)

import numpy as np


# ── Data generation ───────────────────────────────────────────────────────────


def make_sphere(n_samples=2000, R=1.0, noise=0.0, random_state=None):
    """Uniform sample on a sphere of radius R in R^3.
    
    Returns
    -------
    X : (n, 3) ndarray
        Sphere points with optional Gaussian noise added (off-surface
        if noise > 0).
    X_clean : (n, 3) ndarray
        Underlying noise-free points exactly on the sphere surface.
    coords : (n, 2) ndarray
        Intrinsic spherical coordinates (theta, phi) of the clean points.
    """
    rng = np.random.default_rng(random_state)
    
    # Generate clean uniform points on sphere
    X_clean = rng.standard_normal((n_samples, 3))
    X_clean = X_clean / np.linalg.norm(X_clean, axis=1, keepdims=True) * R
    
    # Intrinsic coordinates — from the truly clean points
    theta = np.arccos(np.clip(X_clean[:, 2] / R, -1, 1))
    phi = np.arctan2(X_clean[:, 1], X_clean[:, 0])
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    coords = np.stack([theta, phi], axis=1)
    
    # Add noise (off-surface)
    if noise > 0:
        X = X_clean + rng.normal(0, noise, X_clean.shape)
    else:
        X = X_clean.copy()
    
    return X, X_clean, coords


# ── Geodesic distance ─────────────────────────────────────────────────────────

def sphere_geodesic(X_or_intrinsic, R=1.0, from_intrinsic=False):
    """
    Great-circle distance pairwise on a sphere.
    
    Parameters
    ----------
    X_or_intrinsic : (n, 3) or (n, 2) ndarray
        Either 3D embedded points or intrinsic (theta, phi).
    R : float
        Sphere radius.
    from_intrinsic : bool
        If True, input is (theta, phi); if False, input is 3D xyz.
    
    Returns
    -------
    D : (n, n) ndarray
        Pairwise geodesic distances.
    """
    if from_intrinsic:
        theta, phi = X_or_intrinsic[:, 0], X_or_intrinsic[:, 1]
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        xyz = np.stack([x, y, z], axis=1)
    else:
        xyz = X_or_intrinsic
        # Normalize in case of noise
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * R
    
    # Pairwise dot products
    cos_angle = np.clip((xyz @ xyz.T) / (R**2), -1.0, 1.0)
    # Great-circle distance = R * angle
    return R * np.arccos(cos_angle)

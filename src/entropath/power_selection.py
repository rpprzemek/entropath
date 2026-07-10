import numpy as np
from scipy.sparse import issparse
from phate.vne import compute_von_neumann_entropy, find_knee_point


def von_neumann_entropy(diff_op, t_max=150, smooth=False, plot=False, ax=None, logger=None, verbose=None):
    """Calculate Von Neumann Entropy

    Determines the Von Neumann entropy of the diffusion affinities
    at varying levels of `t`. The user should select a value of `t`
    around the "knee" of the entropy curve.

    We require that 'fit' stores the value of `PHATE.diff_op`
    in order to calculate the Von Neumann entropy.

    Parameters
    ----------
    t_max : int, default: 100
        Maximum value of `t` to test

    Returns
    -------
    entropy : array, shape=[t_max]
        The entropy of the diffusion affinities for each value of `t`
    """
    t = np.arange(t_max)
    P = diff_op.copy()
    if issparse(diff_op):
        P = P.toarray()

    entropies = compute_von_neumann_entropy(P, t_max=t_max)
    if smooth:
        from scipy.signal import savgol_filter
        entropies = savgol_filter(entropies, window_length=11, polyorder=3)

    t_opt = find_knee_point(y=entropies, x=t)
    if logger:
        logger.info(f"Automatically selected MERW diffusion power = {t_opt}", extra={"indent": 2})
    
    """ from kneed import KneeLocator

    kneedle = KneeLocator(
        t, entropies,
        curve="convex",
        direction="decreasing"
    )

    t_opt = kneedle.knee """

    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        else:
            show = False
        ax.plot(t, entropies, label="Von Neumann Entropy")
        ax.scatter(t_opt, entropies[t == t_opt], marker="*", c="k", s=50)
        ax.set_xlabel("t")
        ax.set_ylabel("Von Neumann Entropy")
        ax.set_title("Optimal t = {}".format(t_opt))
        if show:
            plt.show()

    return t_opt, entropies, t

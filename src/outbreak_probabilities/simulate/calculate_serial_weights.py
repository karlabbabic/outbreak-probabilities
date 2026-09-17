# src/outbreak_probabilities/simulate/calculate_serial_weights.py
"""
Compute discrete-time weekly serial interval weights w_k from a continuous
gamma distribution g(u) using the triangular-kernel weekly discretisation
described in the PMO methods.

Return a normalized array w = [w1, w2, ..., w_kmax] that sums to 1.
"""
from functools import lru_cache
import numpy as np
from scipy.stats import gamma as scipy_gamma
from numpy.polynomial.legendre import leggauss

# Cache to avoid recomputing for same parameter tuples
@lru_cache(maxsize=64)
def compute_serial_weights(mean: float, std: float, k_max: int, nquad: int = 64, step: float = 7.0) -> np.ndarray:
    """
    Compute weekly serial interval weights using Gauss-Legendre quadrature.

    Parameters
    ----------
    mean : float
        Mean of the continuous serial interval (in days).
    std : float
        Standard deviation of the continuous serial interval (in days).
    k_max : int
        Number of weekly bins to compute (returns array length k_max).
    nquad : int, optional
        Number of Gauss-Legendre nodes to use for the fixed quadrature (default 64).
    step : float, optional
        Bin half-width in days (default 7.0 days for weekly bins).

    Returns
    -------
    w : np.ndarray
        1-D array of length k_max of normalized weights summing to 1.
    """
    # Input validation
    if not isinstance(k_max, int) or k_max < 1:
        raise ValueError("k_max must be an integer >= 1")
    if mean <= 0.0 or std <= 0.0:
        raise ValueError("mean and std must be positive numbers")

    # Gamma parameters: shape (alpha) and scale (theta)
    var = std ** 2
    shape = (mean / std) ** 2
    scale = var / mean

    g = scipy_gamma(a=shape, scale=scale)

    # Quick return for k_max == 1 (all mass in single bin)
    if k_max == 1:
        return np.array([1.0], dtype=float)

    # Prepare quadrature nodes/weights
    nodes, quad_weights = leggauss(nquad)

    w = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        # integration interval for bin k:
        # left = 7*(k-1), right = 7*(k+1), center = 7*k
        center = step * k
        left = step * (k - 1)
        right = step * (k + 1)

        half_width = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)

        # map nodes from [-1,1] to [left,right]
        u = half_width * nodes + midpoint

        # triangular kernel: 1 - |u - center|/step inside [left,right], zero outside
        tri = 1.0 - np.abs(u - center) / step
        tri[tri < 0.0] = 0.0

        # gamma pdf evaluated at quadrature points
        pdf_vals = g.pdf(u)

        integrand = tri * pdf_vals

        # Gauss-Legendre quadrature
        w_k = half_width * np.sum(quad_weights * integrand)
        w[k - 1] = w_k

    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Computed serial-interval weights sum to a non-positive or non-finite value")


    # adjust first bin so sum(w)=1 (Gittins-like adjustment)
    w[0] = max(0.0, 1.0 - w[1:].sum())
    w /= w.sum()

    print(w[0],sum(w))
    return w

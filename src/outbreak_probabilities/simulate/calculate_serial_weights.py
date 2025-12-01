# src/outbreak_probabilities/simulate/calculate_serial_weights.py
# This will compute discrete-time serial interval weights w_k
# from a continuous gamma distribution g(u)
from functools import lru_cache
import numpy as np
from scipy.stats import gamma
from numpy.polynomial.legendre import leggauss

# Use 64 leggauss nodes
@lru_cache(maxsize=64)
def compute_serial_weights(mean, std, k_max, nquad=32, step = 7.0):
    """Calculates weekly weights
    This function takes the mean and std of a disease model, and calculates the
    weights of new cases based on the Gamma distribution.
    Args:
        mean (float): observed average for Ebola
        std (float): std for Ebola
        k_max: maximum number of days 
    Returns:
        w (nparray(k_max,)): array with weights that sum to one
    Raises:
        ValueError
    """
    # Raise some errors
    if k_max < 1:
        raise ValueError("k_max must be > 1")
    if mean < 0 or std < 0:
        raise ValueError("Mean and std are parameters that must be > 1")
    # A few more errors to consider: k_max has to be an integer, mean & std data type
    # Instantiate array of weights
    w = np.zeros(k_max)

    # Calculate the probability density function g(x) using given params
    var = std ** 2
    alpha = (mean/std) ** 2
    theta = var / mean

    g = gamma(a = alpha, scale = theta)

    # k_1 is always 1
    if k_max == 1:
        return np.array([1.0], dtype=float)


    # The integral is w_k = int_(k-1)^(k+1)[1-|u-k|]g(u)du; compute for k = 2,...,k_max
    # quad() is slow because it evaluates g(u) until a certain tolerance is met;
    # instead use Gauss–Legendre quadrature
    # Instead evaluate the integrand at fixed points; this is accurate for smooth integrals
    # This gives w_k = int_(left)^(right)[1-|u-center|/step]g(u)du, with:
    #   center = step * k
    #   left = step * (k-1)
    #   right = step * (k-1)
    # and step = 7 days.

    nodes, weights = leggauss(nquad)

    for k in range(1,k_max+1):
        # Calculate the steps
        center = step * k
        left = step * (k-1)
        right = step * (k+1)
        half_width = 0.5 * (right - left)  # = step
        midpoint = 0.5 * (right + left)

        # u is where the integral is being evaluated
        u = half_width * nodes + midpoint

        # Triangular kernel calculation
        tri = 1.0 - np.abs(u - center) / step
        # Remove any -ve numbers
        tri[tri < 0.0] = 0.0

        # Define gamma pdf
        pdf_vals = g.pdf(u)

        # Find integrand
        integrand_vals = tri * pdf_vals

        # Quadrature calculation
        w_k = half_width * np.sum(weights * integrand_vals)
        w[k - 1] = w_k

    total = float(w.sum())
    if total <= 0:
        raise RuntimeError("Weights sum to a -ve number")
    # Normalize w so that they sum to 1
    w = w / total
    return w

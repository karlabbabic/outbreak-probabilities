import numpy as np
from scipy.stats import gamma
from scipy.integrate import quad

from simulate.calculate_serial_weights import compute_serial_weights

def slow_reference_weights(mean, std, k_max, step=7.0):
    """
    Slow but precise reference implementation using quad, similar to your original loop.
    """
    var = std ** 2
    shape = (mean / std) ** 2
    scale = var / mean
    g = gamma(a=shape, scale=scale)

    w = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        center = step * k
        left = max(0.0, step * (k - 1))
        right = step * (k + 1)

        def integrand(u):
            tri = 1.0 - abs(u - center) / step
            if tri < 0.0:
                return 0.0
            return tri * g.pdf(u)

        val, _ = quad(integrand, left, right, epsabs=1e-10, epsrel=1e-10)
        w[k - 1] = val

    w /= w.sum()
    return w


def test_weights_sum_to_one():
    w = compute_serial_weights(mean=15.3, std=9.3, k_max=30, nquad=32, step=7.0)
    assert w.shape == (30,)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-12


def test_weights_match_reference():
    mean = 15.3
    std = 9.3
    k_max = 10
    step = 7.0

    w_fast = compute_serial_weights(mean, std, k_max, nquad=64, step=step)
    w_slow = slow_reference_weights(mean, std, k_max, step=step)

    # They should be very close
    assert np.allclose(w_fast, w_slow, atol=1e-5, rtol=1e-5)


def test_invalid_kmax():
    import pytest
    with pytest.raises(ValueError):
        compute_serial_weights(mean=10.0, std=5.0, k_max=0)

#!/usr/bin/env python3
# src/outbreak_probabilities/pmo/analytical_refractor.py
"""
Minimal PMO estimator
"""

# Store type annotations as strings instead of evaluating them immediately.
from __future__ import annotations

import argparse
import math
import re
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp, lambertw
from scipy.stats import gamma as scipy_gamma

# defaults (paper-derived)
MEAN_SI_DAYS = 15.3
SD_SI_DAYS = 9.3
DEFAULT_R_MIN = 0.0
DEFAULT_R_MAX = 10.0


def weekly_w(max_weeks=50, mean=MEAN_SI_DAYS, sd=SD_SI_DAYS):
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    g = lambda x: scipy_gamma.pdf(x, a=shape, scale=scale)

    w = np.zeros(max_weeks, dtype=float)
    for k in range(1, max_weeks + 1):
        left = 7 * (k - 1)
        right = 7 * (k + 1)
        integrand = lambda u: (1.0 - abs(u - 7 * k) / 7.0) * g(u) if abs(u - 7 * k) <= 7 else 0.0
        val, _ = quad(integrand, left, right, epsabs=1e-9, epsrel=1e-9)
        w[k - 1] = val

    total = w.sum()
    if total <= 0:
        raise RuntimeError("Serial interval weights sum to zero")

    return w / total


def parse_initial_cases(s: str) -> List[int]:
    """ A simple parse to help with initial case input as strings.
    
    Returns a comma separated strings as a array of integers."""
    if not s:
        return []
    toks = [t for t in re.split(r"[,\s;]+", s.strip()) if t]
    return [int(t) for t in toks]


def log_likelihood_I(I_seq: List[int], R: float, w: np.ndarray) -> float:
    """Poisson renewal log-likelihood; uses gammaln for factorials.
    
    Returns an array of float vals."""
    I = np.asarray(I_seq, dtype=float)
    T = len(I)
    ll = 0.0
    for t in range(1, T):
        max_s = min(t, len(w))
        infectious = sum(I[t - s] * w[s - 1] for s in range(1, max_s + 1))
        lam = R * infectious
        if lam <= 0.0:
            if I[t] == 0:
                continue
            return -np.inf
        ll += I[t] * math.log(lam) - lam - gammaln(int(I[t]) + 1)
    return ll


def extinction_q(R: float) -> float:
    """ Calculates the probability that the process dies out."""
    if R <= 1.0:
        return 1.0
    z = -R * math.exp(-R)
    q = -lambertw(z).real / R
    return float(max(0.0, min(1.0, q)))


def PMO_given_R_general(I_seq: List[int], R: float, w: np.ndarray) -> float:
    """Compute PMO conditional on R by calculating the extinction probability 'none_prob'."""
    
    I = np.asarray(I_seq, dtype=float)
    T = len(I)

    # Get the extinction probability
    q = extinction_q(R)

    total = 0.0

    # Iterates through weeks 
    for k in range(1, T + 1):
        m = T - k
        if m <= 0:
            sum_w = 0.0
        else:
            m_use = min(m, len(w))
            sum_w = float(np.sum(w[:m_use]))
        remaining = 1.0 - sum_w

        # Calculates total remaining infectious volume
        total += I[k - 1] * remaining

    log_none = R * (q - 1.0) * total

    # exp(-800) underflows, exp(800) overflows 
    if log_none < -700:
        none_prob = 0.0
    elif log_none > 700:
        none_prob = 1.0
    else:
        none_prob = math.exp(log_none)

    none_prob = min(1.0, max(0.0, none_prob))
    return 1.0 - none_prob


def PMO_general(
    I_seq: List[int],
    w: Optional[np.ndarray] = None,
    nR: int = 2001,
    R_min: float = DEFAULT_R_MIN,
    R_max: float = DEFAULT_R_MAX,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PMO by integrating PMO(R) over the posterior of R given early case counts.

    Returns:
        PMO_val    : Integrated probability of major outbreak
        R_grid     : Grid of R values
        loglikes   : Log-likelihood log P(I_seq | R) on the grid
        pmogivenR  : PMO conditional on each R
        post       : Posterior density over R (integrates to ~1 with grid spacing)
    """

    # Build serial interval weights if not provided
    # Ensure they extend beyond the observed window
    if w is None:
        w = weekly_w(max_weeks=max(40, len(I_seq) + 5))
    if len(w) < len(I_seq):
        w = weekly_w(max_weeks=len(I_seq) + 10)

    # Discretise R over a uniform grid
    R_grid = np.linspace(R_min, R_max, nR)
    delta = R_grid[1] - R_grid[0]

    # Likelihood of observed early counts under Poisson renewal model
    loglikes = np.array(
        [log_likelihood_I(I_seq, R, w) for R in R_grid],
        dtype=float,
    )

    # If nothing fits (all likelihoods are -inf), return zero PMO
    if not np.any(np.isfinite(loglikes)):
        post = np.zeros_like(loglikes)
        pmogivenR = np.zeros_like(loglikes)
        return 0.0, R_grid, loglikes, pmogivenR, post

    # Posterior over R with uniform prior (normalized on a discrete grid)
    logpost = loglikes - logsumexp(loglikes + math.log(delta))
    post = np.exp(logpost)

    # PMO conditional on R, computed via branching-process theory
    pmogivenR = np.array(
        [PMO_given_R_general(I_seq, R, w) for R in R_grid],
        dtype=float,
    )

    # Integrate PMO(R) against the posterior of R
    PMO_val = float(np.sum(pmogivenR * post * delta))

    return PMO_val, R_grid, loglikes, pmogivenR, post


def compute_pmo_from_string(
    initial_cases: str,
    nR: int = 2001,
    R_min: float = DEFAULT_R_MIN,
    R_max: float = DEFAULT_R_MAX,
) -> Dict:
    """
    High-level helper for runner.py.

    Parses a string of early weekly case counts and returns:
    - PMO (probability of major outbreak)
    - extinction probability
    - supporting R-grid diagnostics
    """

    # Parse "1,2,0" -> [1, 2, 0]
    I_seq = parse_initial_cases(initial_cases)
    if not I_seq:
        raise ValueError("initial_cases must be a non-empty string of integers")

    # Core analytic computation
    PMO_val, R_grid, loglikes, pmogivenR, post = PMO_general(
        I_seq, nR=nR, R_min=R_min, R_max=R_max
    )

    # Package results for runner / downstream use
    return {
        "I_seq": I_seq,
        "PMO": PMO_val,
        "extinction_prob": max(0.0, min(1.0, 1.0 - PMO_val)),
        "R_grid": R_grid,
        "loglikes": loglikes,
        "pmogivenR": pmogivenR,
        "post": post,
    }

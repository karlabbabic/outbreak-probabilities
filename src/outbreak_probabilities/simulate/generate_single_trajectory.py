# ###
# **2. generate_single_trajectory.py**

# Purpose: this file produces a sequence of daily incident counts $I_t$ for t=1...k_max, where k_max is a user-specified upper limit, e.g. k_max for a 50-day simulation of cases. Either take a user-specifies first few weeks of cases or just randomly generate the second case (the first case is always 1?).

# Functions:  
# - simulate_trajectory()

#   - Input: weights `w` from `compute_serial_weights()`. Max number of days. Initial cases (if necessary, could be `=None`). R value for this trajectory. 
#   - Output: $I_{t}$, an array of integers of size k_max; array of simulated case numbers for each.
  
# - calculate_R()

#   - Input: user-specified range, e.g. `[0,5]` for `[R_min, R_max]`.
#   - Output: `R`, an integer; the basic reproduction number.
# ###


import numpy as np
from numpy.random import Generator, default_rng

def calculate_R(R_range, rng=None):
    """Sample R from uniform range
    
    """
    rmin, rmax = float(R_range[0]), float(R_range[1])

    if rmin > rmax:
        raise ValueError("R_range minimum must be <= maximum")

    if rmin == rmax:
        return rmin

    return float(rng.uniform(rmin, rmax))

def detect_is_extinct(trajectory, window):
    """Check if cases = 0
    
    """
    if window is None or window <= 0:
        return False
    if trajectory.size < window:
        return False
    return bool(np.all(trajectory[-window:] == 0))

# def simulate_trajectory(w, max_weeks,R,R_range,rng,initial_cases,extinction_window,major_threshold):
#     """Simulate a single trajectory with the renewal method

#     result : dict with keys:
#         - "trajectory" : np.ndarray shape (max_weeks,), daily counts I_t
#         - "R"          : float, reproduction number used
#         - "cumulative" : int, total cases across the trajectory
#         - "status"     : str, one of {"minor", "major", "ongoing"}
#         - "PMO"        : int, 1 if status=="major", otherwise 0
#     """

#     # Max days check
#     if max_weeks < 1:
#         raise ValueError("Max weeks must be >= 1")

#     # Convert weights to array
#     w_arr = np.asarray(w, dtype=float)

#     if w_arr.ndim != 1:
#         raise ValueError("w is not a 1D sequqnece of weights")
    
#     # Extract dimension of array
#     k_support = w_arr.size

#     # Select rng from np.random
#     if rng is None:
#         rng = default_rng()

#     # Choose reproduction number
#     if R is None:
#         if R_range is None:
#             raise ValueError("Either R or R_range must be provided")
#         R = calculate_R(R_range)
#     R = float(R)

#     # Set up initial cases
#     if initial_cases is None:
#         initial = [1]
#     else:
#         initial = list(initial_cases)
#         # Check that the first case is 1

#     # Define trajectory array
#     trajectory = np.zeros(max_weeks, dtype=int)

#     # Populate initial cases
#     L = min(len(initial), max_weeks)
#     trajectory[:L] = np.asarray(initial[:L], dtype=int)

#     # Find cumulative case count 
#     cumulative = int(trajectory[:L].sum())

#     # Early major check
#     if cumulative >= major_threshold:
#         return {
#             "trajectory": trajectory,
#             "R": R,
#             "cumulative": cumulative,
#             "status": "major",
#             "PMO": 1,
#         }
    
#     # Simulate from day L+1 to max_weeks
#     for t in range(L, max_weeks):
#         max_lag = min(k_support, t)
#         if max_lag == 0:
#             lam_base = 0.0
#         else:
#             past = trajectory[t - max_lag : t]      # I_{t-max_lag}..I_{t-1}
#             ws = w_arr[:max_lag]                   # w_1..w_maxlag
#             lam_base = float(np.dot(ws, past[::-1]))  # align w_s with I_{t-s}

#         lam = R * lam_base
#         new_cases = int(rng.poisson(lam)) if lam > 0.0 else 0

#         trajectory[t] = new_cases
#         cumulative += new_cases

#         # Major outbreak: cumulative >= threshold
#         if cumulative >= major_threshold:
#             return {
#                 "trajectory": trajectory,
#                 "R": R,
#                 "cumulative": cumulative,
#                 "status": "major",
#                 "PMO": 1,
#             }

#         # Extinction: last `extinction_window` days are all zero
#         if extinction_window is not None and detect_is_extinct(trajectory[: t + 1], extinction_window):
#             return {
#                 "trajectory": trajectory,
#                 "R": R,
#                 "cumulative": cumulative,
#                 "status": "minor",
#                 "PMO": 0,
#             }

#     # If we get here, the process neither hit major_threshold nor extinct by max_weeks
#     # Treat as ongoing with PMO = 0 at the trajectory level (you may handle separately later).
#     return {
#         "trajectory": trajectory,
#         "R": R,
#         "cumulative": cumulative,
#         "status": "ongoing",
#         "PMO": 0,
#     }

from typing import Optional, Sequence, Dict, Any
import numpy as np
from numpy.random import Generator, default_rng

def calculate_R(R_range: Sequence[float], rng: Optional[Generator] = None) -> float:
    if rng is None:
        rng = default_rng()

    if len(R_range) != 2:
        raise ValueError("R_range must be a length-2 sequence (R_min, R_max)")

    rmin, rmax = float(R_range[0]), float(R_range[1])
    if rmin > rmax:
        raise ValueError("R_range minimum must be <= maximum")

    if rmin == rmax:
        return rmin

    return float(rng.uniform(rmin, rmax))


def _is_extinct_window(trajectory: np.ndarray, window: int) -> bool:
    if window is None or window <= 0:
        return False
    if trajectory.size < window:
        return False
    return bool(np.all(trajectory[-window:] == 0))


def simulate_trajectory(
    w: Sequence[float],
    max_weeks: int,
    R: Optional[float] = None,
    R_range: Optional[Sequence[float]] = None,
    initial_cases: Optional[Sequence[int]] = None,
    rng: Optional[Generator] = None,
    extinction_window: Optional[int] = None,
    major_threshold: int = 100,
) -> Dict[str, Any]:
    """
    Simulate a single trajectory under the Poisson renewal model.

    IMPORTANT:
    - We ALWAYS simulate out to max_weeks.
    - major_threshold is ONLY used to classify the trajectory as 'major' (PMO=1)
      if cumulative >= major_threshold at any time.
    - We do NOT stop simulating when the threshold is reached.
    """
    if rng is None:
        rng = default_rng()

    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1:
        raise ValueError("w must be a 1-D sequence of serial weights")
    k_support = w_arr.size

    if max_weeks < 1:
        raise ValueError("max_weeks must be >= 1")

    # Choose R
    if R is None:
        if R_range is None:
            raise ValueError("Either R or R_range must be provided")
        R = calculate_R(R_range, rng=rng)
    R = float(R)

    # Initial cases
    if initial_cases is None:
        initial = [1]
    else:
        initial = list(initial_cases)

    trajectory = np.zeros(max_weeks, dtype=int)
    L = min(len(initial), max_weeks)
    trajectory[:L] = np.asarray(initial[:L], dtype=int)

    cumulative = int(trajectory[:L].sum())

    # Flags we track over the full run
    major_flag = cumulative >= major_threshold
    extinct_flag = False

    # Simulate days L+1 .. max_weeks
    for t in range(L, max_weeks):
        max_lag = min(k_support, t)
        if max_lag == 0:
            lam_base = 0.0
        else:
            past = trajectory[t - max_lag : t]        # I_{t-max_lag}..I_{t-1}
            ws = w_arr[:max_lag]                     # w_1..w_maxlag
            lam_base = float(np.dot(ws, past[::-1])) # align w_s with I_{t-s}

        lam = R * lam_base
        new_cases = int(rng.poisson(lam)) if lam > 0.0 else 0

        trajectory[t] = new_cases
        cumulative += new_cases

        # Update flags but DO NOT stop the simulation
        if cumulative >= major_threshold:
            major_flag = True

        if extinction_window is not None and _is_extinct_window(trajectory[: t + 1], extinction_window):
            extinct_flag = True
            # Once extinct, future intensity is 0 anyway.
            # You can break and fill zeros (already zeros) or just continue; we'll break for speed.
            # Fill remaining days with zeros (already default) and break.
            break

    # If we broke early for extinction, the tail of trajectory is already zeros (default).

    # Classify at the END of simulation window
    if major_flag:
        status = "major"
        pmo = 1
    elif extinct_flag:
        status = "minor"
        pmo = 0
    else:
        status = "ongoing"
        pmo = 0

    return {
        "trajectory": trajectory,
        "R": R,
        "cumulative": cumulative,
        "status": status,
        "PMO": pmo,
    }

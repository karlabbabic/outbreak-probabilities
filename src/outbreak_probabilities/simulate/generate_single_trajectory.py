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

def calculate_R(R_range, rng = None):
    """ Calculates R using rng from [R_min, R_max]
    
    """
    # Could implement different rng if necessary
    if rng is None:
        rng = default_rng()

    # Checking R values
    if len(R_range) != 2:
        raise ValueError("R_range must be a length-2 sequence (R_min, R_max)")

    rmin, rmax = float(R_range[0]), float(R_range[1])
    if rmin > rmax:
        raise ValueError("R_range minimum must be <= maximum")

    if rmin == rmax:
        return rmin

    return float(rng.uniform(rmin, rmax))

# Sliding window check for extinction
def _is_extinct_window(trajectory, window):
    if window is None or window <= 0:
        return False
    if trajectory.size < window:
        return False
    return bool(np.all(trajectory[-window:] == 0))


def simulate_trajectory(
    w,
    max_weeks,
    R = None,
    R_range = None,
    initial_cases = None,
    rng = None,
    extinction_window = None,
    major_threshold = 100,
):
    """
    Simulate a single trajectory under the Poisson renewal model.
    - This will always simulate out to max_weeks, even after the threshold for outbreak / extinction met
    - Use user specified major_threshold to classify traj. as 'major' (PMO=1) if cumulative cases > major_threshold
    """

    # When drawing from a single Poission distribution RNG should be the same key, but user may specify
    if rng is None:
        rng = default_rng()

    # Initialize weights array
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1:
        raise ValueError("w must be a 1-D sequence of serial weights")
    
    # Do some error checking
    if max_weeks < 1:
        raise ValueError("max_weeks must be >= 1")

    # Add some other basic checking for input variables, even those by default = None. 

    # Select R, either user specified constant or randomly from a distribution 
    if R is None:
        if R_range is None:
            raise ValueError("Either R or R_range must be provided")
        R = calculate_R(R_range, rng=rng)
    R = float(R)

    # Define initial cases if specified, otherwise start with I_1 = 1.
    if initial_cases is None:
        initial = [1]
    else:
        initial = list(initial_cases)

    trajectory = np.zeros(max_weeks, dtype=int)
    L = min(len(initial), max_weeks)
    trajectory[:L] = np.asarray(initial[:L], dtype=int)

    # Define cumulative case counts
    cumulative = int(trajectory[:L].sum())

    # Flags we track over the full run
    major_flag = cumulative >= major_threshold
    extinct_flag = False

    # Simulate days L+1 .. max_weeks
    for t in range(L, max_weeks):
        max_lag = min(w_arr.size, t)
        if max_lag == 0:
            lam_base = 0.0
        else:
            past = trajectory[t - max_lag : t]
            ws = w_arr[:max_lag]
            lam_base = float(np.dot(ws, past[::-1]))
        lam = R * lam_base
        new_cases = int(rng.poisson(lam)) if lam > 0.0 else 0
        trajectory[t] = new_cases
        cumulative += new_cases

        # Update flags but DO NOT stop the simulation
        if cumulative >= major_threshold:
            major_flag = True
            break

        if extinction_window is not None and _is_extinct_window(trajectory[: t + 1], extinction_window):
            extinct_flag = True
            # Once extinct populate break, because trajectory[] is already filled with zeros. 
            # Note that if something fails it will return zeros. 
            break

    # Classify at the end of simulation window
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

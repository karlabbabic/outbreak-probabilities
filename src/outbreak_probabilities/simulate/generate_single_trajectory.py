import numpy as np
from numpy.random import default_rng

def calculate_R(R_range, rng=None, dist="uniform", dist_params=None):
    """
    Draw a single R value (float).
    - R_range: (R_min, R_max)
    - dist: "uniform" | "normal" | "lognormal"
    Returns float R.
    """
    if rng is None:
        rng = default_rng()
    rmin, rmax = float(R_range[0]), float(R_range[1])
    if rmin == rmax:
        return float(rmin)
    if dist == "uniform":
        return float(rng.uniform(rmin, rmax))
    if dist == "normal":
        params = {} if dist_params is None else dict(dist_params)
        mu = params.get("mean", 0.5 * (rmin + rmax))
        sd = params.get("sd", (rmax - rmin) / 4.0)
        return float(max(rmin, min(rmax, rng.normal(mu, sd))))
    if dist == "lognormal":
        if rmin <= 0:
            raise ValueError("R_range lower bound must be > 0 for lognormal")
        log_min, log_max = np.log(rmin), np.log(rmax)
        return float(np.exp(rng.uniform(log_min, log_max)))
    raise ValueError(f"Unknown dist '{dist}'")

def _is_extinct_window(trajectory, window):
    if not window or window <= 0:
        return False
    if trajectory.size < window:
        return False
    return bool(np.all(trajectory[-window:] == 0))

def simulate_trajectory(
    w,
    max_weeks=50,
    R=None,
    R_range=None,
    initial_cases=None,
    rng=None,
    extinction_window=None,
    major_threshold=100,
    stop_on_major=True,
):
    """
    Poisson-renewal single trajectory simulator.

    Important convention: w[0] is the serial weight for a 1-step lag
      i.e. contribution to I_t from I_{t-1} is w[0], from I_{t-2} is w[1], etc.

    Returns dict with keys:
      'trajectory' (length max_weeks, trailing weeks zero if stopped early),
      'R' (float),
      'cumulative' (int),
      'status' ('major'|'minor'|'ongoing'),
      'PMO' (0 or 1),
      't_end' (index of last filled week +1).
    """
    if rng is None:
        rng = default_rng()
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1:
        raise ValueError("w must be 1-D")
    if max_weeks < 1:
        raise ValueError("max_weeks must be >= 1")

    if R is None:
        if R_range is None:
            raise ValueError("Either R or R_range must be provided")
        R = float(calculate_R(R_range, rng=rng))
    else:
        R = float(R)

    if initial_cases is None:
        initial = [1]
    else:
        initial = list(initial_cases)

    trajectory = np.zeros(int(max_weeks), dtype=int)
    L = min(len(initial), int(max_weeks))
    trajectory[:L] = np.asarray(initial[:L], dtype=int)
    cumulative = int(trajectory[:L].sum())

    major_flag = cumulative >= major_threshold
    extinct_flag = False
    t_end = L  # next index to fill

    for t in range(L, int(max_weeks)):
        max_lag = min(w_arr.size, t)
        if max_lag == 0:
            lam_base = 0.0
        else:
            # past = [I_{t-max_lag}, ..., I_{t-1}]  -- older .. recent
            past = trajectory[t - max_lag : t]
            # ws = [w1, w2, ..., w_max_lag] corresponding to I_{t-1}, I_{t-2}, ...
            ws = w_arr[:max_lag]
            # Align ws[0] with most recent past element -> reverse past
            lam_base = float(np.dot(ws, past[::-1]))

        lam = R * lam_base
        new_cases = int(rng.poisson(lam)) if lam > 0.0 else 0
        trajectory[t] = new_cases
        cumulative += new_cases
        t_end = t + 1

        if cumulative >= major_threshold:
            major_flag = True
            if stop_on_major:
                break

        if extinction_window is not None and _is_extinct_window(trajectory[: t + 1], extinction_window):
            extinct_flag = True
            break

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
        "t_end": t_end,
    }

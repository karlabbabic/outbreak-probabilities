# 
# **3. batch_processing.py**

# Purpose: this file will call `generate_single_trajectory` as many times as needed to simulate a user-defined number of tracks. It should also write these results to a .csv or a Python `tempfile`. It should write to a CSV with headers day_1,...,day_max; may also include R.


# Functions:
# - batch_process()
  
#   - Input: the number of tracks `N`, and the maximum number of days.
#   - Output: a 2D array, printed to a csv.
# 

import numpy as np
import csv
import tempfile
from numpy.random import default_rng
from pathlib import Path

# Import APIs from other files in folder
from .generate_single_trajectory import simulate_trajectory, calculate_R
from .calculate_serial_weights import compute_serial_weights

def default_csv_path(use_tempfile = True):
    """Define the filepath of csv
    
    
    """
    if use_tempfile:
        tf = tempfile.NamedTemporaryFile(prefix="simulated_cases_",suffix=".csv")
        p = Path(tf.name)
        tf.close()
        return p
    else:
        return Path("simulated_cases.csv")

def generate_batch(
    N,
    w,
    max_weeks,
    R_range,
    initial_cases=None,
    extinction_window=None,
    major_threshold=100,
    out_path=None,
    use_tempfile=True,
    seed=None,
):
    """Simulate N individual trajectories
    """
    rng = default_rng(seed)
    
    # Change rng if specified
    if rng == None:
        rng = default_rng(seed)
    else:
        rng = default_rng(seed)

    # Do file pathing
    if out_path is None:
        csv_path = default_csv_path(use_tempfile=use_tempfile)
    else:
        csv_path = Path(out_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup csv headers
    header = ["sim_id", "R_draw"] + [f"week_{d}" for d in range(1, max_weeks + 1)] + [
        "cumulative_cases",
        "status",
        "PMO",
    ]
    # define ND array for all trajectories
    trajectories = np.zeros((N, max_weeks), dtype=int)

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for sim_id in range(1, N + 1):
            # Draw R via calculate_R using the same rng for reproducibility
            R = calculate_R(R_range, rng=rng)

            result = simulate_trajectory(
                w=w,
                max_weeks=max_weeks,
                R=R,
                R_range=None,
                initial_cases=initial_cases,
                rng=rng,
                extinction_window=extinction_window,
                major_threshold=major_threshold,
            )

            traj = result["trajectory"]
            cumulative = int(result["cumulative"])
            status = result["status"]
            pmo_flag = int(result["PMO"])

            trajectories[sim_id - 1, :] = traj

            # weeks
            row = [sim_id, float(R), *traj.tolist(), cumulative, status, pmo_flag]
            writer.writerow(row)

    return trajectories, csv_path

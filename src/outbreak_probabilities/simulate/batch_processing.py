# 
# **3. batch_processing.py**

# Purpose: this file will call `generate_single_trajectory` as many times as needed to simulate a user-defined number of tracks. It should also write these results to a .csv or a Python `tempfile`. It should write to a CSV with headers day_1,...,day_max; may also include R.


# Functions:
# - batch_process()
  
#   - Input: the number of tracks `N`, and the maximum number of days.
#   - Output: a 2D array, printed to a csv.
# 

"""
child_rng is used in the rng for calculate_R and simulate_trajectory, so the randomness of that
trajectory comes from the single int, child_rng/
"""

# batch_processing.py: improved generate_batch with per-trajectory seeds and R distribution support
from numpy.random import default_rng
from pathlib import Path
import numpy as np
import csv
import tempfile

from .generate_single_trajectory import simulate_trajectory, calculate_R

def default_csv_path(use_tempfile=True):
    if use_tempfile:
        tf = tempfile.NamedTemporaryFile(prefix="simulated_cases_", suffix=".csv")
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
    R_dist="uniform",
    R_dist_params=None,
    generate_full=False,
):
    """
    Simulate N trajectories.
    - Writes per-row: sim_id, sim_seed, R_draw, week_1..week_max, cumulative_cases, status, PMO
    - Uses a master RNG seeded by `seed`. For each trajectory draws a per-traj integer seed and uses it to create a child RNG.
    - R_dist and R_dist_params forwarded to calculate_R.
    """
    master_rng = default_rng(seed)

    # Validate/convert R_range
    try:
        R_min = float(R_range[0])
        R_max = float(R_range[1])
    except Exception as e:
        raise ValueError("R_range must be a length-2 numeric sequence") from e

    if out_path is None:
        csv_path = default_csv_path(use_tempfile=use_tempfile)
    else:
        csv_path = Path(out_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Number of weeks to store in the CSV
    N_WRITE_WEEKS = 5

    # Header includes sim_seed and R_draw
    header = ["sim_id", "sim_seed", "R_draw"] + [f"week_{d}" for d in range(1, N_WRITE_WEEKS+1)] + [
        "cumulative_cases",
        "status",
        "PMO",
    ]

    # # Header includes sim_seed and R_draw
    # header = ["sim_id", "sim_seed", "R_draw"] + [f"week_{d}" for d in range(1, max_weeks + 1)] + [
    #     "cumulative_cases",
    #     "status",
    #     "PMO",
    # ]

    trajectories = np.zeros((N, max_weeks), dtype=int)
    csv_path.write_text("")

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header_len = len(header)

        # metadata rows: keep w and R_range info (first two columns blank/aligned)
        w_list = list(w)
        row_w = ["", "w"] + w_list
        row_w += [""] * (header_len - len(row_w))
        row_w = row_w[:header_len]

        row_R = ["", "R_range", R_min, R_max]
        row_R += [""] * (header_len - len(row_R))
        row_R = row_R[:header_len]

        # Also store the master seed used (helpful)
        row_master_seed = ["", "master_seed", seed]
        row_master_seed += [""] * (header_len - len(row_master_seed))
        row_master_seed = row_master_seed[:header_len]

        writer.writerow(row_w)
        writer.writerow(row_R)
        writer.writerow(row_master_seed)
        writer.writerow(header)

        for sim_id in range(1, N + 1):
            # draw an integer seed for this trajectory (64-bit positive)
            sim_seed = int(master_rng.integers(low=0, high=2**63 - 1, dtype=np.int64))

            # create an independent RNG for this trajectory
            child_rng = default_rng(sim_seed)

            # draw R using the child RNG and specified distribution
            R = calculate_R((R_min, R_max), rng=child_rng, dist=R_dist, dist_params=R_dist_params)

            result = simulate_trajectory(
                w=w,
                max_weeks=max_weeks,
                R=R,
                R_range=None,
                initial_cases=initial_cases,
                rng=child_rng,
                extinction_window=extinction_window,
                major_threshold=major_threshold,
            )

            # By default only write the first N_WRITE_WEEKS to the csv
            if generate_full==True:
                traj = result["trajectory"]
            else:
                traj = result["trajectory"][0:N_WRITE_WEEKS]

            cumulative = int(result["cumulative"])
            status = result["status"]
            pmo_flag = int(result["PMO"])

            row = [sim_id, sim_seed, float(R), *traj.tolist(), cumulative, status, pmo_flag]
            writer.writerow(row)

    return trajectories, csv_path

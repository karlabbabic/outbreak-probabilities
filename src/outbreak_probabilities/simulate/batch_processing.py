# Replace old generate_batch with this corrected version
from numpy.random import default_rng
from pathlib import Path
import numpy as np
import csv
import tempfile
import json

from .generate_single_trajectory import simulate_trajectory, calculate_R

def default_csv_path(use_tempfile=True):
    if use_tempfile:
        tf = tempfile.NamedTemporaryFile(prefix="simulated_cases_", suffix=".csv", delete=False)
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
    write_weeks=5,
    stop_on_major=True,
):
    """
    Simulate N trajectories and write CSV + weights JSON.
    Returns (trajectories_array, csv_path).
    trajectories_array: shape (N, max_weeks), always populated (zeros after stop).
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

    # Defensive: ensure w covers at least max_weeks; if not, warn (could recompute)
    w = np.asarray(w, dtype=float)
    if len(w) < max_weeks:
        # Prefer failing loudly so caller notices; change to padding if you prefer
        raise RuntimeError(
            f"Serial interval weights length {len(w)} < simulation horizon max_weeks {max_weeks}. "
            "Recompute weights with larger k_max."
        )

    # Number of weeks to store in the CSV, by default 5
    N_WRITE_WEEKS = int(write_weeks)
    if N_WRITE_WEEKS < 1:
        N_WRITE_WEEKS = 1
    N_WRITE_WEEKS = min(N_WRITE_WEEKS, max_weeks)

    # Header includes sim_seed and R_draw (only first N_WRITE_WEEKS of weeks)
    header = ["sim_id", "sim_seed", "R_draw"] + [f"week_{d}" for d in range(1, N_WRITE_WEEKS + 1)] + [
        "cumulative_cases",
        "status",
        "PMO",
    ]

    # Prepare return trajectories buffer and ensure it's zero-filled by default
    trajectories = np.zeros((int(N), int(max_weeks)), dtype=int)

    # Write JSON weights once (next to CSV) so consumers have canonical weights
    weights_json_path = csv_path.with_suffix(".weights.json")
    weights_payload = {
        "weights": list(map(float, w)),
        "metadata": {
            "k_max": int(len(w)),
            "step_days": 7.0,
            "R_range": [R_min, R_max],
            "major_threshold": major_threshold,
            "extinction_window": extinction_window,
            "note": "Weekly triangular-kernel discretisation of gamma serial interval",
        },
    }
    with weights_json_path.open("w") as f:
        json.dump(weights_payload, f, indent=2)

    # Prepare CSV and write header and metadata rows
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header_len = len(header)

        # metadata rows: keep w and R_range info (first two columns blank/aligned)
        # Put a short preview of weights to avoid an extremely wide csv row
        row_w = ["", "w_preview"] + [float(x) for x in w[:N_WRITE_WEEKS]]
        row_w += [""] * (header_len - len(row_w))
        row_w = row_w[:header_len]

        row_R = ["", "R_range", R_min, R_max]
        row_R += [""] * (header_len - len(row_R))
        row_R = row_R[:header_len]

        row_master_seed = ["", "master_seed", seed]
        row_master_seed += [""] * (header_len - len(row_master_seed))
        row_master_seed = row_master_seed[:header_len]

        writer.writerow(row_w)
        writer.writerow(row_R)
        writer.writerow(row_master_seed)
        writer.writerow(header)

        # Now simulate each trajectory, store full into 'trajectories' array, write CSV row
        for sim_idx in range(int(N)):
            sim_id = sim_idx + 1
            sim_seed = int(master_rng.integers(low=0, high=2**63 - 1, dtype=np.int64))
            child_rng = default_rng(sim_seed)

            R = calculate_R((R_min, R_max), rng=child_rng, dist=R_dist, dist_params=R_dist_params)

            # Call simulate_trajectory with explicit stop_on_major behavior if supported
            result = simulate_trajectory(
                w=w,
                max_weeks=int(max_weeks),
                R=R,
                R_range=None,
                initial_cases=initial_cases,
                rng=child_rng,
                extinction_window=extinction_window,
                major_threshold=major_threshold,
            )

            # result["trajectory"] should be length max_weeks (zeros if stopped early)
            traj_full = np.asarray(result["trajectory"], dtype=int)
            if traj_full.shape[0] > max_weeks:
                # In case simulate_trajectory returned longer sequence, truncate
                traj_full = traj_full[:max_weeks]
            elif traj_full.shape[0] < max_weeks:
                # If shorter, pad with zeros to ensure consistent shape
                pad = np.zeros(int(max_weeks) - traj_full.shape[0], dtype=int)
                traj_full = np.concatenate([traj_full, pad])

            # Store full trajectory into buffer
            trajectories[sim_idx, :] = traj_full

            # Decide what to write to CSV
            if generate_full:
                traj_to_write = traj_full.tolist()
            else:
                traj_to_write = traj_full[:N_WRITE_WEEKS].tolist()

            cumulative = int(result.get("cumulative", int(traj_full.sum())))
            status = result.get("status", "")
            pmo_flag = int(result.get("PMO", 0))

            row = [sim_id, sim_seed, float(R), *traj_to_write, cumulative, status, pmo_flag]
            writer.writerow(row)

    # Return full trajectories buffer (N x max_weeks) and csv path
    return trajectories, csv_path

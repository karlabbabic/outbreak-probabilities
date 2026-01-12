"""
This file is for testing only. Run it to see that the simulated trajectories appear.
"""

"""
This file is for testing only.
Run it to see that the simulated trajectories appear
and that R_draw and sim_seed are written to the CSV.
"""

import numpy as np
from calculate_serial_weights import compute_serial_weights
from batch_processing import generate_batch

# 1. Serial weights (continuous mean 15.3 days, sd 9.3 days, weekly bins)
w = compute_serial_weights(
    mean=15.3,
    std=9.3,
    k_max=10,
    nquad=32,
    step=7.0
)

# 2. Generate a batch of trajectories
trajectories, csv_path = generate_batch(
    N=5000000,
    w=w,
    max_weeks=15,
    R_range=(0.0, 10.0),
    initial_cases=[1],
    extinction_window=10,
    major_threshold=100,
    seed=42,                 # master seed (scenario-level)
    R_dist="uniform",        # default, but explicit is good
    R_dist_params=None,
    out_path="data/test_simulations.csv",
    use_tempfile=False,
)

print("Trajectories shape:", trajectories.shape)
print("CSV written to:", csv_path)

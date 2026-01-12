"""
This file is for testing only. Run it to see that the simulated trajectories appear.
"""

import numpy as np
from calculate_serial_weights import compute_serial_weights
from batch_processing import generate_batch

# 1. Serial weights (continuous mean 15.3 days, sd 9.3 days, weekly bins)
w = compute_serial_weights(mean=15.3, std=9.3, k_max=10, nquad=32, step=7.0)

# 2. Generate a batch of 500 trajectories of length 50 days
trajectories, csv_path = generate_batch(
    N=1000000,
    w=w,
    max_weeks=10,
    R_range=(0, 10.0),
    initial_cases=[1],
    extinction_window=2,      # e.g. 2 weeks of zeros → extinct
    major_threshold=100,
    seed=42,
    out_path="data/simulated_cases_and_serial_interval_and_weights.csv"
)

print("Trajectories shape:", trajectories.shape)
print("CSV written to:", csv_path)

# # Do the same but with a tempfile
# w = compute_serial_weights(mean=15.3, std=9.3, k_max=50, nquad=32, step=7.0)
# trajectories, csv_path = generate_batch(
#     N=500,
#     w=w,
#     max_weeks=50,
#     R_range=(0.5, 3.0),
#     initial_cases=[1],
#     extinction_window=14,      # e.g. 2 weeks of zeros → extinct
#     major_threshold=100,
#     seed=42,
# )

# print("Trajectories shape:", trajectories.shape)
# print("CSV written to:", csv_path)

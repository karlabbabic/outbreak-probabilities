**The purpose of the files in this folder is to extract relevant synthetic trajectories based on observed initial data and calculate the PMO based on a threshold placed on the final number of cases (might want to look at cumulative cases as well?)**

There should be two files: one that does the matching and calculation of the PMO, and another file that runs that:

```
# trajectory_matching/runner.py
results = match_and_compute_pmo(csv_path, observed, k=3, threshold=100)

# src/gui/
from trajectory_matching.runner import match_and_compute_pmo, save_matches_to_csv
```

1. trajectory.py

This will extract paths that match the observed data and, at the end, calculate the fraction with more than the threshold cases.

Functions: `match_trajectories(): `calculate_PMO() '. Input: synthetic data CSV file path. 
Output: a separate CSV, `matched_CSV`, containing trajectories with the specified initial condition. Also `PMO`, a float, which is the fraction of those matched trajectories to all.

2. runner.py

Runs the trajectory.py functions.

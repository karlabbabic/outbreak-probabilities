# src/outbreak-probabilities/trajectory_matching/trajectory.py

from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd


def trajectory_match_pmo(
    observed_weeks: Sequence[int],
    simulated_csv: str,
    header_rows: int = 3,
    week_prefix: str = "week_",
    return_matches_df: bool = False,
) -> Dict[str, Any]:
    """
    Match simulated trajectories where the first len(observed_weeks) weekly counts
    are identical to `observed_weeks`, and compute the fraction of those matches
    classified as major (PMO == 1).

    Parameters
    ----------
    observed_weeks :
        Sequence (list/tuple/1D array) of integers giving the observed weekly counts
        for weeks 1..k that should be matched exactly.
    simulated_csv :
        Path to the CSV file produced by generate_batch (it may include metadata rows).
    header_rows :
        Number of metadata rows before the header in the CSV. Default: 3 (w, R_range, master_seed).
    week_prefix :
        Prefix used for week columns in CSV (default "week_").
    return_matches_df :
        If True, include the DataFrame of matched rows under key "matches_df" in the return dict.

    Returns
    -------
    dict with keys:
      - "n_matches": int, number of matching trajectories
      - "n_major": int, number of matching trajectories with PMO == 1
      - "pmo_fraction": float or None, n_major / n_matches (None if n_matches == 0)
      - "matched_indices": list of integer row indices in the DataFrame corresponding to matches
      - optionally "matches_df": DataFrame with matched rows (if return_matches_df True)

    Notes
    -----
    - Matching is exact equality of integer weekly counts. If your observed data are floats
      or need rounding/tolerance, convert them to ints before calling this function.
    """
    simulated_path = Path(simulated_csv)
    if not simulated_path.exists():
        raise FileNotFoundError(f"Simulated CSV not found: {simulated_csv}")

    # Load CSV (skip metadata rows)
    df = pd.read_csv(simulated_path, header=header_rows)

    # Identify week columns in sorted order: week_1, week_2, ...
    week_cols = [c for c in df.columns if c.startswith(week_prefix)]
    if not week_cols:
        raise ValueError(f"No columns starting with '{week_prefix}' found in {simulated_csv}")

    # sort by numeric suffix
    try:
        week_cols = sorted(week_cols, key=lambda s: int(s.split("_")[1]))
    except Exception:
        # fallback: lexicographic sort if naming is unexpected
        week_cols = sorted(week_cols)

    k = len(observed_weeks)
    if k <= 0:
        raise ValueError("observed_weeks must contain at least one value")

    if k > len(week_cols):
        raise ValueError(f"observed_weeks length ({k}) > available simulated weeks ({len(week_cols)})")

    # Convert observed_weeks to integer numpy array for comparison
    obs_arr = np.asarray(observed_weeks, dtype=int)

    # Extract the first k week columns from simulated data as ints (coerce NaN -> -999 to avoid accidental match)
    sim_firstk = df[week_cols[:k]].fillna(-999).astype(int).to_numpy()

    # Compare rows to observed vector
    # We'll use vectorized comparison: (sim_firstk == obs_arr).all(axis=1)
    matches_mask = np.all(sim_firstk == obs_arr.reshape(1, -1), axis=1)

    matched_indices = list(np.nonzero(matches_mask)[0].tolist())
    n_matches = int(matches_mask.sum())

    # Default results
    n_major = 0
    pmo_fraction: Optional[float] = None
    matches_df = None

    if n_matches == 0:
        pmo_fraction = None
    else:
        # Ensure PMO column exists
        if "PMO" not in df.columns:
            raise ValueError("CSV does not contain 'PMO' column; cannot compute PMO fraction.")

        matches_df = df.loc[matches_mask].copy()
        # coerce PMO to integers (in case of float)
        try:
            pmo_vals = matches_df["PMO"].astype(int).to_numpy()
        except Exception:
            # fallback: treat truthy values as 1, else 0
            pmo_vals = matches_df["PMO"].apply(lambda x: 1 if x else 0).to_numpy()

        n_major = int((pmo_vals == 1).sum())
        pmo_fraction = float(n_major) / float(n_matches)

    out: Dict[str, Any] = {
        "n_matches": n_matches,
        "n_major": n_major,
        "pmo_fraction": pmo_fraction,
        "matched_indices": matched_indices,
    }

    if return_matches_df:
        out["matches_df"] = matches_df

    return out
# # Example small usage (not executed on import):
# if __name__ == "__main__":  # pragma: no cover
#     # example observed first 3 weekly counts
#     observed = [1,2,0]
#     csv = "data/test_simulations.csv"
#     res = trajectory_match_pmo(observed, csv, header_rows=3, return_matches_df=False)
#     print("Matches:", res["n_matches"], "Major among matches:", res["n_major"], "PMO fraction:", res["pmo_fraction"])

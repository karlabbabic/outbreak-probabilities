"""
Depreciated since changing trajectory simulation to cut off when the cumulative cases >= threshold

interpret_weekly_data.py

Creative interpretation & visualisation of weekly outbreak simulations.

Assumes CSV produced by simulate.generate_batch with columns:
    sim_id, R_draw, week_1, week_2, ..., week_K, cumulative_cases, status, PMO

Status is assumed to be one of:
    - "major"   (outbreak: cumulative >= threshold at some point)
    - "minor"   (dies out / extinct)
    - "ongoing" (neither major nor extinct within horizon)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _load_weeks(csv_path: str) -> Tuple[pd.DataFrame, list[str]]:
    """
    Load simulation CSV and return (DataFrame, ordered list of week_* columns).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    week_cols = [c for c in df.columns if c.startswith("week_")]
    week_cols = sorted(week_cols, key=lambda x: int(x.split("_")[1]))
    if not week_cols:
        raise ValueError("No 'week_*' columns found in CSV.")
    if "status" not in df.columns:
        raise KeyError("CSV missing 'status' column.")
    if "R_draw" not in df.columns:
        raise KeyError("CSV missing 'R_draw' column.")
    if "cumulative_cases" not in df.columns:
        raise KeyError("CSV missing 'cumulative_cases' column.")
    return df, week_cols


def _get_week_array(df: pd.DataFrame, week_cols: list[str]) -> np.ndarray:
    """Return weekly counts as a 2D numpy array."""
    return df[week_cols].to_numpy()


# -------------------------------------------------------------------
# 1. Final size histograms by outcome
# -------------------------------------------------------------------

def plot_final_size_by_status(csv_path: str, bins: int = 30) -> None:
    """
    Show how big outbreaks get in the end, separated by status.

    - Histogram of total cumulative_cases for:
        * major
        * minor
        * ongoing
    """
    df, _ = _load_weeks(csv_path)

    # Split by status
    major = df[df["status"] == "major"]["cumulative_cases"]
    minor = df[df["status"] == "minor"]["cumulative_cases"]
    ongoing = df[df["status"] == "ongoing"]["cumulative_cases"]

    plt.figure(figsize=(10, 5))

    # We overlay histograms with some transparency
    if not major.empty:
        plt.hist(
            major,
            bins=bins,
            alpha=0.5,
            label=f"Major (n={len(major)})",
        )
    if not minor.empty:
        plt.hist(
            minor,
            bins=bins,
            alpha=0.5,
            label=f"Minor (n={len(minor)})",
        )
    if not ongoing.empty:
        plt.hist(
            ongoing,
            bins=bins,
            alpha=0.5,
            label=f"Ongoing (n={len(ongoing)})",
        )

    plt.xlabel("Final cumulative cases")
    plt.ylabel("Number of simulations")
    plt.title("Final outbreak size by status")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 2. Time to outbreak (for major simulations)
# -------------------------------------------------------------------

def _time_to_threshold(weekly: np.ndarray, threshold: int) -> np.ndarray:
    """
    For each trajectory, compute the week index when cumulative first >= threshold.
    If never reached, return np.nan for that trajectory.
    """
    cumulative = weekly.cumsum(axis=1)  # shape (n_sims, n_weeks)
    reached = cumulative >= threshold

    n_sims, n_weeks = cumulative.shape
    times = np.full(n_sims, np.nan)

    for i in range(n_sims):
        idx = np.argmax(reached[i]) if reached[i].any() else None
        if idx is not None and reached[i, idx]:
            # week index is idx+1 in human terms
            times[i] = idx + 1
    return times


def plot_time_to_outbreak(csv_path: str, threshold: int = 100, bins: int = 15) -> None:
    """
    Among major outbreaks, show the distribution of 'time to outbreak'
    (week when cumulative first passes threshold).

    This helps answer: do most major outbreaks "pop" early or late?
    """
    df, week_cols = _load_weeks(csv_path)
    df_major = df[df["status"] == "major"]
    if df_major.empty:
        print("No major outbreaks found; nothing to plot.")
        return

    weekly = _get_week_array(df_major, week_cols)
    times = _time_to_threshold(weekly, threshold=threshold)
    times = times[~np.isnan(times)]

    plt.figure(figsize=(8, 4))
    plt.hist(times, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Week when cumulative first >= threshold")
    plt.ylabel("Number of major outbreaks")
    plt.title(f"Time to outbreak (threshold={threshold}) for major simulations")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 3. R distribution by outcome
# -------------------------------------------------------------------

def plot_R_by_status(csv_path: str, bins: int = 30) -> None:
    """
    Compare the distribution of R_draw between major and non-major simulations.

    This shows how much R separates outbreaks from non-outbreaks.
    """
    df, _ = _load_weeks(csv_path)

    major = df[df["status"] == "major"]["R_draw"]
    non_major = df[df["status"] != "major"]["R_draw"]

    plt.figure(figsize=(8, 4))

    if not major.empty:
        plt.hist(
            major,
            bins=bins,
            alpha=0.6,
            label=f"Outbreak (n={len(major)})",
            density=False,
        )
    if not non_major.empty:
        plt.hist(
            non_major,
            bins=bins,
            alpha=0.6,
            label=f"Non-outbreak (n={len(non_major)})",
            density=False,
        )

    plt.xlabel("R_draw")
    plt.ylabel("Abs. counts")
    plt.title("Distribution of R by outcome (n = 2000, R uniformly drawn from [0,5], threshold = 100 cases, 3 weeks of 0 cases = extinction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Distribution of R by outcome.jpg")

    plt.show()


# -------------------------------------------------------------------
# 4. Early growth vs outcome
# -------------------------------------------------------------------

def plot_early_growth_vs_status(
    csv_path: str,
    start_week: int = 1,
    end_week: int = 3,
) -> None:
    """
    Look at early growth and how it relates to outcome.

    For each simulation:
      - compute total cases in weeks [start_week..end_week]
      - plot distributions for major vs non-major

    This helps answer: do major outbreaks have much stronger early growth?
    """
    df, week_cols = _load_weeks(csv_path)
    weekly = _get_week_array(df, week_cols)  # (n_sims, n_weeks)
    statuses = df["status"].to_numpy()

    n_sims, n_weeks = weekly.shape
    if end_week > n_weeks:
        end_week = n_weeks

    # Convert to zero-based slice
    s = start_week - 1
    e = end_week

    early_sum = weekly[:, s:e].sum(axis=1)

    major_mask = statuses == "major"
    non_major_mask = statuses != "major"

    major_vals = early_sum[major_mask]
    non_major_vals = early_sum[non_major_mask]

    plt.figure(figsize=(8, 4))

    if major_vals.size > 0:
        plt.hist(
            major_vals,
            bins=20,
            alpha=0.6,
            label=f"Major (n={major_vals.size})",
        )
    if non_major_vals.size > 0:
        plt.hist(
            non_major_vals,
            bins=20,
            alpha=0.6,
            label=f"Non-major (n={non_major_vals.size})",
        )

    plt.xlabel(f"Cases in weeks {start_week}–{end_week}")
    plt.ylabel("Number of simulations")
    plt.title("Early growth vs outcome")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 5. A compact dashboard-like text summary
# -------------------------------------------------------------------

def print_summary_stats(csv_path: str) -> None:
    """
    Print a quick textual summary:
      - counts by status
      - PMO (fraction major)
      - R stats by status
      - median final size by status
    """
    df, _ = _load_weeks(csv_path)

    total = len(df)
    counts = df["status"].value_counts()
    pmo = counts.get("major", 0) / total if total > 0 else 0.0

    print("=== Simulation summary ===")
    print(f"Total simulations: {total}")
    print("Status counts:")
    for status, count in counts.items():
        print(f"  {status:7s}: {count:5d}  ({count/total:5.1%})")
    print(f"\nPMO estimate (fraction major): {pmo:.3f}")

    # R stats by status
    print("\nR_draw summary by status:")
    for status in ["major", "minor", "ongoing"]:
        subset = df[df["status"] == status]["R_draw"]
        if subset.empty:
            continue
        print(
            f"  {status:7s}: mean={subset.mean():.2f}, "
            f"median={subset.median():.2f}, "
            f"sd={subset.std(ddof=1):.2f}"
        )

    # Final size by status
    print("\nFinal size (cumulative_cases) by status (median [IQR]):")
    for status in ["major", "minor", "ongoing"]:
        subset = df[df["status"] == status]["cumulative_cases"]
        if subset.empty:
            continue
        q1 = subset.quantile(0.25)
        med = subset.median()
        q3 = subset.quantile(0.75)
        print(f"  {status:7s}: median={med:.1f}  [Q1={q1:.1f}, Q3={q3:.1f}]")

csv_path = "data/simulated_cases.csv"

# 1. Big-picture summary
print_summary_stats(csv_path)

# 2. How big do things get, by outcome?
plot_final_size_by_status(csv_path)

# 3. When do outbreaks "take off"?
plot_time_to_outbreak(csv_path, threshold=100)

# 4. How different is R for major vs non-major?
plot_R_by_status(csv_path)

# 5. Does early growth predict major outbreaks?
plot_early_growth_vs_status(csv_path, start_week=1, end_week=3)

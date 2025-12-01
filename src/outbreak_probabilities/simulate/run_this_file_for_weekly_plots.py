"""
plot_weekly_cases.py

Plot simulated outbreak trajectories BY WEEK using matplotlib.

Assumes CSVs produced by simulate.generate_batch() with columns:
    sim_id, R_draw, week_1, week_2, ..., week_K, cumulative_cases, status, PMO
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_weeks(csv_path: str):
    """
    Load simulation CSV and return (DataFrame, ordered list of week_* columns).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    week_cols = [c for c in df.columns if c.startswith("week_")]
    week_cols = sorted(week_cols, key=lambda x: int(x.split("_")[1]))
    if not week_cols:
        raise ValueError("No 'week_*' columns found in CSV.")
    return df, week_cols


def plot_weekly_cases(
    csv_path: str,
    max_weeks: Optional[int] = None,
    n_traj: Optional[int] = 200,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> None:
    """
    Plot weekly case trajectories colored by status, plus mean, median, and 10–90% spread.
    """
    df, week_cols = _load_weeks(csv_path)

    if "status" not in df.columns:
        raise KeyError("CSV missing 'status' column (expected 'minor'/'major'/'ongoing').")

    weekly = df[week_cols].to_numpy()  # shape (n_sims, n_weeks)
    statuses = df["status"].to_numpy()
    n_sims, n_weeks = weekly.shape

    # Optionally limit number of weeks
    if max_weeks is not None:
        n_weeks = min(n_weeks, max_weeks)
        weekly = weekly[:, :n_weeks]
        week_cols = week_cols[:n_weeks]

    weeks = np.arange(1, n_weeks + 1)

    # Decide which simulations to plot as individual lines
    if n_traj is None or n_traj >= n_sims:
        idx = np.arange(n_sims)
    else:
        idx = np.linspace(0, n_sims - 1, n_traj, dtype=int)

    plt.figure(figsize=(10, 5))

    # --- individual trajectories (colored by status) ---
    for i in idx:
        traj = weekly[i, :]
        status = statuses[i]

        if status == "major":
            color = "red"
        elif status == "minor":
            color = "blue"
        else:
            color = "grey"

        plt.plot(weeks, traj, color=color, linewidth=0.6, alpha=0.35)

    # --- summary stats: mean, median, 10–90% band ---
    mean_weekly = weekly.mean(axis=0)
    median_weekly = np.median(weekly, axis=0)
    q_low = np.quantile(weekly, lower_q, axis=0)
    q_high = np.quantile(weekly, upper_q, axis=0)

    # Shaded spread band
    plt.fill_between(
        weeks,
        q_low,
        q_high,
        alpha=0.25,
        label=f"{int(lower_q*100)}–{int(upper_q*100)}% range",
    )

    # Mean and median lines
    plt.plot(weeks, mean_weekly, color="black", linewidth=2.0, label="Mean weekly cases")
    plt.plot(
        weeks,
        median_weekly,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Median weekly cases",
    )

    # Legend with explicit color meaning for statuses
    legend_lines = [
        Line2D([0], [0], color="red", lw=2, label="Major (status='major')"),
        Line2D([0], [0], color="blue", lw=2, label="Minor / extinct (status='minor')"),
        Line2D([0], [0], color="grey", lw=2, label="Ongoing"),
        Line2D([0], [0], color="black", lw=2, label="Mean"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Median"),
    ]
    plt.legend(handles=legend_lines, loc="upper right")

    plt.xlabel("Week")
    plt.ylabel("Cases per week")
    plt.title("Weekly case trajectories (colored by status) with mean/median and 10–90% spread")
    plt.xticks(weeks)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# NEW: area + mean plots highlighting different groups
# ----------------------------------------------------------------------

def _summary_band(weekly: np.ndarray, weeks: np.ndarray, lower_q: float, upper_q: float):
    """
    Compute mean and quantile band for a weekly array.
    """
    mean_weekly = weekly.mean(axis=0)
    q_low = np.quantile(weekly, lower_q, axis=0)
    q_high = np.quantile(weekly, upper_q, axis=0)
    return mean_weekly, q_low, q_high


def plot_weekly_area_all(
    csv_path: str,
    max_weeks: Optional[int] = None,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> None:
    """
    Plot ONLY the area (10–90%) and mean weekly cases across ALL simulations.
    """
    df, week_cols = _load_weeks(csv_path)
    weekly = df[week_cols].to_numpy()
    n_sims, n_weeks = weekly.shape

    if max_weeks is not None:
        n_weeks = min(n_weeks, max_weeks)
        weekly = weekly[:, :n_weeks]

    weeks = np.arange(1, n_weeks + 1)
    mean_weekly, q_low, q_high = _summary_band(weekly, weeks, lower_q, upper_q)

    plt.figure(figsize=(10, 5))

    # Grey band for all simulations
    plt.fill_between(
        weeks,
        q_low,
        q_high,
        alpha=0.3,
        label=f"All sims: {int(lower_q*100)}–{int(upper_q*100)}% range",
    )

    # Thicker black mean line
    plt.plot(weeks, mean_weekly, color="black", linewidth=3.0, label="All sims: mean")

    plt.xlabel("Week")
    plt.ylabel("Cases per week")
    plt.title("Weekly cases (ALL simulations): area + mean")
    plt.xticks(weeks)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weekly_area_outbreaks(
    csv_path: str,
    max_weeks: Optional[int] = None,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> None:
    """
    Plot area + mean weekly cases for OUTBREAK trajectories only (status == 'major').
    """
    df, week_cols = _load_weeks(csv_path)

    if "status" not in df.columns:
        raise KeyError("CSV missing 'status' column.")

    df_major = df[df["status"] == "major"]
    if df_major.empty:
        print("No major outbreaks (status='major') found; nothing to plot.")
        return

    weekly = df_major[week_cols].to_numpy()
    n_sims, n_weeks = weekly.shape

    if max_weeks is not None:
        n_weeks = min(n_weeks, max_weeks)
        weekly = weekly[:, :n_weeks]

    weeks = np.arange(1, n_weeks + 1)
    mean_weekly, q_low, q_high = _summary_band(weekly, weeks, lower_q, upper_q)

    plt.figure(figsize=(10, 5))

    # Reddish band for major outbreaks
    plt.fill_between(
        weeks,
        q_low,
        q_high,
        alpha=0.3,
        label=f"Outbreaks: {int(lower_q*100)}–{int(upper_q*100)}% range",
    )

    # Bold red mean line
    plt.plot(weeks, mean_weekly, color="red", linewidth=3.0, label="Outbreaks: mean")

    plt.xlabel("Week")
    plt.ylabel("Cases per week")
    plt.title("Weekly cases (OUTBREAKS only): area + mean")
    plt.xticks(weeks)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weekly_area_non_outbreaks(
    csv_path: str,
    max_weeks: Optional[int] = None,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> None:
    """
    Plot area + mean weekly cases for NON-OUTBREAK trajectories.

    Here non-outbreak = status != 'major' (i.e. 'minor' and 'ongoing').
    """
    df, week_cols = _load_weeks(csv_path)

    if "status" not in df.columns:
        raise KeyError("CSV missing 'status' column.")

    df_non = df[df["status"] != "major"]
    if df_non.empty:
        print("No non-outbreak trajectories (status != 'major') found; nothing to plot.")
        return

    weekly = df_non[week_cols].to_numpy()
    n_sims, n_weeks = weekly.shape

    if max_weeks is not None:
        n_weeks = min(n_weeks, max_weeks)
        weekly = weekly[:, :n_weeks]

    weeks = np.arange(1, n_weeks + 1)
    mean_weekly, q_low, q_high = _summary_band(weekly, weeks, lower_q, upper_q)

    plt.figure(figsize=(10, 5))

    # Bluish band for non-outbreaks
    plt.fill_between(
        weeks,
        q_low,
        q_high,
        alpha=0.3,
        label=f"Non-outbreaks: {int(lower_q*100)}–{int(upper_q*100)}% range",
    )

    # Bold blue mean line
    plt.plot(weeks, mean_weekly, color="blue", linewidth=3.0, label="Non-outbreaks: mean")

    plt.xlabel("Week")
    plt.ylabel("Cases per week")
    plt.title("Weekly cases (NON-OUTBREAKS): area + mean")
    plt.xticks(weeks)
    plt.legend()
    plt.tight_layout()
    plt.show()



csv_path = "data/simulated_cases.csv"

# Original full plot: trajectories + everything
plot_weekly_cases(csv_path, max_weeks=10, n_traj=200)

# NEW: area + mean (all sims)
plot_weekly_area_all(csv_path, max_weeks=10)

# NEW: outbreaks only
plot_weekly_area_outbreaks(csv_path, max_weeks=10)

# NEW: non-outbreaks (status != 'major')
plot_weekly_area_non_outbreaks(csv_path, max_weeks=10)

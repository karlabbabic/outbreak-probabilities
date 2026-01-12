# src/outbreak-probabilities/trajectory_matching/plot_matches.py

from pathlib import Path
from typing import Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajectory import trajectory_match_pmo


def _week_columns(df: pd.DataFrame, prefix: str = "week_") -> list:
    """Return sorted week column names like ['week_1','week_2', ...]."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda s: int(s.split("_")[1])) if cols else []


def plot_matched_trajectories(
    observed_weeks: Sequence[int],
    simulated_csv: str,
    *,
    major_threshold: int = 100,
    header_rows: int = 3,
    week_prefix: str = "week_",
    out_png: Optional[str] = "figs/matched_trajectories.png",
    figsize: tuple = (9, 6),
    line_color: str = "tab:blue",
    alpha: float = 0.6,
    linewidth: float = 1.0,
    max_plot: Optional[int] = None,
    return_fig: bool = False,
) -> Dict[str, Any]:
    """
    Find trajectories that exactly match `observed_weeks` and plot them up to the cutoff.

    - Matched trajectories are plotted in `line_color`.
    - If a trajectory reaches `major_threshold` (cumulative), a red dot marks the cutoff week.
    - Returns summary dict with n_matches, n_major, pmo_fraction, matched_indices and optionally 'fig'.
    """
    # 1) find matches and get the matched dataframe
    res = trajectory_match_pmo(
        observed_weeks=observed_weeks,
        simulated_csv=simulated_csv,
        header_rows=header_rows,
        week_prefix=week_prefix,
        return_matches_df=True,
    )

    n_matches = res["n_matches"]
    if n_matches == 0:
        raise ValueError("No matching trajectories found.")

    matches_df = res.get("matches_df")
    matched_indices = res.get("matched_indices", [])

    # If matches_df wasn't returned for any reason, load from CSV and slice
    if matches_df is None:
        full = pd.read_csv(simulated_csv, header=header_rows)
        matches_df = full.iloc[matched_indices].copy()

    # 2) prepare week columns and numeric array
    week_cols = _week_columns(matches_df, prefix=week_prefix)
    if not week_cols:
        raise ValueError("No week_* columns found in simulated CSV / matched dataframe.")

    weeks = np.arange(1, len(week_cols) + 1)
    arr = matches_df[week_cols].to_numpy(dtype=float)  # shape (n_matches, n_weeks)

    # Optionally limit number plotted
    if max_plot is not None and max_plot < arr.shape[0]:
        arr = arr[:max_plot, :]
        matches_df = matches_df.iloc[:max_plot]
        plotted = arr.shape[0]
    else:
        plotted = arr.shape[0]

    # 3) compute cutoffs and prepare plotting segments
    cumul = np.cumsum(arr, axis=1)  # cumulative per trajectory
    # For each trajectory, find first index where cumulative >= threshold; if never, -1
    hits = np.argmax(cumul >= major_threshold, axis=1)
    reached_mask = (cumul >= major_threshold).any(axis=1)
    # For rows where no hit exists, np.argmax returned 0 but reached_mask is False; handle later.

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each trajectory up to its cutoff (or full length)
    for i, row in enumerate(arr):
        if reached_mask[i]:
            cutoff = int(hits[i])
            xs = weeks[: cutoff + 1]
            ys = row[: cutoff + 1]
        else:
            xs = weeks
            ys = row
        ax.plot(xs, ys, color=line_color, alpha=alpha, linewidth=linewidth)

    # Plot red dots for those that reached threshold (single scatter call)
    if reached_mask.any():
        cutoff_idxs = hits[reached_mask].astype(int)
        xs = weeks[cutoff_idxs]
        ys = arr[reached_mask, :][np.arange(cutoff_idxs.size), cutoff_idxs]
        ax.scatter(xs, ys, color="red", s=18, zorder=3, label="cutoff (≥ threshold)")

    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    ax.set_title(
        f"Matched trajectories (matches={n_matches}, plotted={plotted})\n"
        f"Cutoff: cumulative ≥ {major_threshold}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.ylim(-1,max(ys)+10)
    plt.tight_layout()
    ax.minorticks_on()
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # Minor ticks every 0.5 on X-axis

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # 4) save / return
    result = {
        "n_matches": n_matches,
        "n_major": res["n_major"],
        "pmo_fraction": res["pmo_fraction"],
        "matched_indices": matched_indices,
    }

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        result["out_png"] = out_png
        print(f"Saved matched trajectories plot to {out_png}")
        if return_fig:
            # If the user wants the fig object, we must recreate it (since we closed it).
            # Recreate quickly by calling the function again with return_fig=True and out_png=None
            # But to keep things simple, return without fig when saved.
            pass
    else:
        if return_fig:
            result["fig"] = fig

    return result


# Quick manual test when run as a script (not executed on import)
if __name__ == "__main__":
    observed = [1, 2, 0]
    csv = "data/test_simulations.csv"
    out = plot_matched_trajectories(
        observed_weeks=observed,
        simulated_csv=csv,
        major_threshold=100,
        header_rows=3,
        out_png="figs/matched_trajectories.png",
        max_plot=200,
    )
    print("Result:", out)

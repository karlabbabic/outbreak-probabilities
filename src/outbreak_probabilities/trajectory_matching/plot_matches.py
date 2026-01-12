# src/outbreak-probabilities/trajectory_matching/plot_matches.py
"""
Load matched trajectories and plot them.

Behavior preserved from original script:
- Matched trajectories with PMO==1 plotted in blue, PMO==0 in dark gray
- Red dot marks the first week where cumulative >= MAJOR_THRESHOLD
- X-axis shows integer ticks (data lives at weeks 1..n_weeks)
- Saves PNG: main (OUT_PNG)
- Adjust the configuration section below as needed and run the file.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# function that finds matches; assumed to live in trajectory.py
from trajectory import trajectory_match_pmo

# ----------------------
# Configuration (edit)
# ----------------------
SIM_CSV: str = "data/test_simulations.csv"
OBSERVED: List[int] = [1, 2, 0]            # observed prefix to match
HEADER_ROWS: int = 3                       # number of metadata rows in CSV
WEEK_PREFIX: str = "week_"
MAJOR_THRESHOLD: int = 100
OUT_PNG: str = "figs/matched_trajectories.png"
MAX_PLOT: Optional[int] = 200              # set an int to limit plotted matches for readability (e.g. 200)
FIGSIZE: Tuple[int, int] = (9, 6)
BLUE: str = "tab:blue"
GRAY: str = "dimgray"
ALPHA: float = 0.6
LINEWIDTH: float = 1.0
# ----------------------


def load_matches(sim_csv: str, observed: List[int], header_rows: int, week_prefix: str) -> Dict:
    """Call trajectory_match_pmo and return the result dictionary (expecting keys 'n_matches', 'pmo_fraction', 'matches_df')."""
    return trajectory_match_pmo(
        observed_weeks=observed,
        simulated_csv=sim_csv,
        header_rows=header_rows,
        week_prefix=week_prefix,
        return_matches_df=True,
    )

def get_week_columns(df: pd.DataFrame, week_prefix: str) -> List[str]:
    """Return sorted week_* columns (numerical order)."""
    week_cols = [c for c in df.columns if c.startswith(week_prefix)]
    # sort by integer after prefix, robust to different prefix forms like "week_1"
    week_cols = sorted(week_cols, key=lambda s: int(s.split(week_prefix)[1]))
    return week_cols

def prepare_plot_data(matches_df: pd.DataFrame, week_cols: List[str], max_plot: Optional[int], major_threshold: int):
    """Return arrays needed for plotting: arr, pmo_flags, cumul, reached, hit_idx, n_weeks."""
    if "PMO" not in matches_df.columns:
        raise SystemExit("Matched dataframe missing 'PMO' column.")

    pmo_flags = matches_df["PMO"].astype(int).to_numpy()
    arr = matches_df[week_cols].to_numpy(dtype=float)

    # apply MAX_PLOT (optional)
    if max_plot is not None and max_plot < arr.shape[0]:
        arr = arr[:max_plot, :]
        pmo_flags = pmo_flags[:max_plot]
        plotted = arr.shape[0]
    else:
        plotted = arr.shape[0]

    # cumulative & cutoff detection
    cumul = np.cumsum(arr, axis=1)
    reached = (cumul >= major_threshold).any(axis=1)
    # argmax returns index of first True; for rows with no True it returns 0 (we'll respect reached mask later)
    hit_idx = np.argmax(cumul >= major_threshold, axis=1)

    n_weeks = arr.shape[1]
    return arr, pmo_flags, cumul, reached, hit_idx, n_weeks, plotted

def plot_matches(
    arr: np.ndarray,
    pmo_flags: np.ndarray,
    cumul: np.ndarray,
    reached: np.ndarray,
    hit_idx: np.ndarray,
    n_weeks: int,
    plotted: int,
    n_matches: int,
    n_total: int,
    pmo_fraction: Optional[float],
    major_threshold: int,
    out_png: str,
    figsize: Tuple[int, int],
):
    weeks = np.arange(1, n_weeks + 1)  # data exist at weeks 1..n_weeks

    fig, ax = plt.subplots(figsize=figsize)

    # plot each trajectory, truncating lines at the cutoff if reached
    for i, weekly in enumerate(arr):
        color = BLUE if pmo_flags[i] == 1 else GRAY
        if reached[i]:
            k = int(hit_idx[i])
            ax.plot(weeks[: k + 1], weekly[: k + 1], color=color, alpha=ALPHA, linewidth=LINEWIDTH)
        else:
            ax.plot(weeks, weekly, color=color, alpha=ALPHA, linewidth=LINEWIDTH)

    # red dots at cutoff for those that reached threshold
    if reached.any():
        xs = weeks[hit_idx[reached]]
        ys = arr[reached, :][np.arange(xs.size), hit_idx[reached]]
        ax.scatter(xs, ys, color="red", s=20, zorder=3, label="cutoff")

    # legend: colored lines and PMO text
    pmo_label = f"PMO (matched) = {pmo_fraction:.3f}" if pmo_fraction is not None else "PMO (matched) = NA"
    # invisible artists to display labels for line colors & PMO text
    ax.plot([], [], color=BLUE, label="PMO = 1 (matched)")
    ax.plot([], [], color=GRAY, label="PMO = 0 (matched)")
    ax.plot([], [], " ", label=f"{pmo_label}")

    # x-axis ticks: integer ticks for each week (1..n_weeks)
    ax.set_xlim(1, n_weeks + 0.5)
    ax.set_xticks(np.arange(1, n_weeks + 1, 1))
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_ylim(0,max(ys)+1)
    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    ax.set_title(
        f"Matched trajectories (plotted {plotted} out of {n_matches}/{n_total} matches.)\n"
        f"Initial Cases: {OBSERVED} \nCutoff: cumulative ≥ {major_threshold}"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # keep the same spine visibility behavior as the original
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ensure formatter behavior matches original intent
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # set minor ticks/gridline cadence and thin minor gridlines
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor', linewidth=0.6)

    # save
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    # 1) find matches
    res = load_matches(sim_csv=SIM_CSV, observed=OBSERVED, header_rows=HEADER_ROWS, week_prefix=WEEK_PREFIX)

    n_matches = res.get("n_matches", 0)
    if n_matches == 0:
        raise SystemExit("No matching trajectories found.")

    pmo_fraction = res.get("pmo_fraction", None)
    matches_df = res.get("matches_df")
    if matches_df is None:
        raise SystemExit("trajectory_match_pmo did not return 'matches_df'.")

    # total number of simulated trajectories (for title)
    n_total = len(pd.read_csv(SIM_CSV, header=HEADER_ROWS))

    # 2) prepare week columns and numeric array
    week_cols = get_week_columns(matches_df, WEEK_PREFIX)
    if not week_cols:
        raise SystemExit("No week_* columns found in matched dataframe.")

    arr, pmo_flags, cumul, reached, hit_idx, n_weeks, plotted = prepare_plot_data(
        matches_df=matches_df,
        week_cols=week_cols,
        max_plot=MAX_PLOT,
        major_threshold=MAJOR_THRESHOLD,
    )

    # 3) plot and save
    plot_matches(
        arr=arr,
        pmo_flags=pmo_flags,
        cumul=cumul,
        reached=reached,
        hit_idx=hit_idx,
        n_weeks=n_weeks,
        plotted=plotted,
        n_matches=n_matches,
        n_total=n_total,
        pmo_fraction=pmo_fraction,
        major_threshold=MAJOR_THRESHOLD,
        out_png=OUT_PNG,
        figsize=FIGSIZE,
    )

    print(f"Saved matched trajectories plot to {OUT_PNG}")


if __name__ == "__main__":
    main()

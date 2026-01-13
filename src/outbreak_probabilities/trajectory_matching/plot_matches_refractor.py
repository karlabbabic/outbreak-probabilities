# src/outbreak_probabilities/trajectory_matching/plot_matches.py
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

# relative import for intra-package usage
from .trajectory import trajectory_match_pmo

# Defaults
SIM_CSV: str = "data/test_simulations.csv"
OBSERVED: List[int] = [1, 2, 0]
HEADER_ROWS: int = 3
WEEK_PREFIX: str = "week_"
MAJOR_THRESHOLD: int = 100
OUT_PNG: str = "figs/matched_trajectories.png"

SAMPLE_STRATEGY: str = "highest_peak"
SAMPLE_SIZE: Optional[int] = 200
MAX_PLOT: Optional[int] = 200
FIGSIZE: Tuple[int, int] = (9, 6)
BLUE: str = "tab:blue"
GRAY: str = "dimgray"
ALPHA: float = 0.6
LINEWIDTH: float = 1.0
# ----------------------


def load_matches(sim_csv: str, observed: List[int], header_rows: int, week_prefix: str) -> Dict:
    """Call trajectory_match_pmo and return the result dictionary."""
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
    week_cols = sorted(week_cols, key=lambda s: int(s.split(week_prefix)[1]))
    return week_cols


# ----- sampling helper (kept from your plot_traj) -----
def select_indices(df: pd.DataFrame, week_cols: List[str], strategy: str, sample_size: Optional[int],
                   hybrid_k: int = 25, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Return array of indices to plot according to strategy.
    Strategies:
      - random
      - highest_cumulative
      - highest_peak
      - highest_R
      - hybrid (includes extremes + random remainder)
    If sample_size is None or >= n, returns all indices.
    """
    n = len(df)
    rng = np.random.default_rng(random_seed)

    if sample_size is None or sample_size >= n:
        return np.arange(n)

    if strategy == "random":
        return rng.choice(n, size=sample_size, replace=False)

    cumulative = df[week_cols].sum(axis=1).values
    peak = df[week_cols].max(axis=1).values
    Rdraw = df["R_draw"].values if "R_draw" in df.columns else np.zeros(n)

    if strategy == "highest_cumulative":
        return np.argsort(-cumulative)[:sample_size]

    if strategy == "highest_peak":
        return np.argsort(-peak)[:sample_size]

    if strategy == "highest_R":
        return np.argsort(-Rdraw)[:sample_size]

    if strategy == "hybrid":
        k = min(hybrid_k, max(1, sample_size // 4))
        top_c = list(np.argsort(-cumulative)[:k])
        top_p = list(np.argsort(-peak)[:k])
        top_r = list(np.argsort(-Rdraw)[:k])
        idx_set = set(top_c + top_p + top_r)
        remaining = sample_size - len(idx_set)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), np.fromiter(idx_set, int))
            if remaining >= pool.size:
                choice = pool
            else:
                choice = rng.choice(pool, size=remaining, replace=False)
            idx_set.update(choice.tolist())
        return np.fromiter(sorted(idx_set), dtype=int)

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def prepare_plot_data(matches_df: pd.DataFrame, week_cols: List[str], max_plot: Optional[int], major_threshold: int):
    """Return arrays needed for plotting: arr, pmo_flags, cumul, reached, hit_idx, n_weeks, plotted."""
    if "PMO" not in matches_df.columns:
        raise SystemExit("Matched dataframe missing 'PMO' column.")

    pmo_flags = matches_df["PMO"].astype(int).to_numpy()
    arr = matches_df[week_cols].to_numpy(dtype=float)

    # apply MAX_PLOT (optional)
    if max_plot is not None and max_plot < arr.shape[0]:
        arr = arr[:max_plot, :].copy()
        pmo_flags = pmo_flags[:max_plot].copy()
        plotted = arr.shape[0]
    else:
        plotted = arr.shape[0]

    # cumulative & cutoff detection
    cumul = np.cumsum(arr, axis=1)
    reached = (cumul >= major_threshold).any(axis=1)
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
    observed: List[int],
    out_png: str,
    figsize: Tuple[int, int],
):
    """
    Draw and save the matched trajectories plot.
    `observed` is the observed initial-case prefix (used in the title).
    """
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
    ax.plot([], [], color=BLUE, label="PMO = 1 (matched)")
    ax.plot([], [], color=GRAY, label="PMO = 0 (matched)")
    ax.plot([], [], " ", label=f"{pmo_label}")

    # x-axis ticks: integer ticks for each week (1..n_weeks)
    ax.set_xlim(1, n_weeks + 0.5)
    ax.set_xticks(np.arange(1, n_weeks + 1, 1))
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)

    # set y-limit so last red dot is visible (preserve original logic)
    if reached.any():
        ax.set_ylim(0, max(arr[reached, :].max(), 1) + 1)
    else:
        ax.set_ylim(0, max(arr.max(), 1) + 1)

    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    # format observed list like "1,2,3"
    obs_str = ", ".join(str(x) for x in observed) if observed else "[]"
    ax.set_title(
        f"Matched trajectories (plotted {plotted} out of {n_matches}/{n_total} matches.)\n"
        f"Initial Cases: {obs_str}  |  Cutoff: cumulative ≥ {major_threshold}\n"
        f"Strategy used: {SAMPLE_STRATEGY}"
    )

    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # keep the same spine visibility behavior as the original
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ensure formatter behavior matches original intent
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    ax.grid(
        which="major",
        color="0.65",      # gray
        linewidth=1.2,
        alpha=0.6
    )

    # Minor ticks halfway between majors (but hidden)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Minor gridlines: lighter gray, thinner
    ax.grid(
        which="minor",
        color="0.85",      # lighter gray
        linewidth=0.6,
        alpha=0.8
    )

    # save
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run_plot_matches(
    sim_csv: str = SIM_CSV,
    observed: List[int] = OBSERVED,
    header_rows: int = HEADER_ROWS,
    week_prefix: str = WEEK_PREFIX,
    major_threshold: int = MAJOR_THRESHOLD,
    out_png: str = OUT_PNG,
    sample_strategy: str = SAMPLE_STRATEGY,
    sample_size: Optional[int] = SAMPLE_SIZE,
    max_plot: Optional[int] = MAX_PLOT,
    figsize: Tuple[int, int] = FIGSIZE,
):
    """
    Top-level function to be called from the runner.
    It returns the path to the saved PNG and also writes two auxiliary CSVs:
      - <out_stem>_match_indices.csv  (single column: match_index)
      - <out_stem>_sampled_matches.csv (full sampled dataframe with match_index)
    """
    # 1) find matches
    res = load_matches(sim_csv=sim_csv, observed=observed, header_rows=header_rows, week_prefix=week_prefix)

    n_matches = res.get("n_matches", 0)
    if n_matches == 0:
        raise SystemExit("No matching trajectories found.")

    pmo_fraction = res.get("pmo_fraction", None)
    matches_df = res.get("matches_df")
    if matches_df is None:
        raise SystemExit("trajectory_match_pmo did not return 'matches_df'.")

    # add a persistent identifier for each matched row so callers can reuse selections later
    matches_df = matches_df.copy()
    matches_df["match_index"] = matches_df.index.astype(int)

    # total number of simulated trajectories (for title)
    n_total = len(pd.read_csv(sim_csv, header=header_rows))

    # prepare week columns
    week_cols = get_week_columns(matches_df, week_prefix)
    if not week_cols:
        raise SystemExit("No week_* columns found in matched dataframe.")

    # sampling: decide which matched indices to keep for plotting
    n_matches_total = len(matches_df)
    effective_sample_size = sample_size if sample_size is not None else max_plot
    if effective_sample_size is None or effective_sample_size >= n_matches_total:
        # use all matches
        sampled_df = matches_df.copy()
        sel_idx = np.arange(n_matches_total)
        plotted_limit = sampled_df.shape[0]
    else:
        # select indices relative to matches_df
        sel_idx = select_indices(matches_df, week_cols, strategy=sample_strategy, sample_size=effective_sample_size, random_seed=42)
        sel_idx = np.unique(sel_idx)
        sampled_df = matches_df.iloc[sel_idx].reset_index(drop=True)
        plotted_limit = sampled_df.shape[0]

    # prepare numeric arrays from sampled_df
    arr, pmo_flags, cumul, reached, hit_idx, n_weeks, plotted = prepare_plot_data(
        matches_df=sampled_df,
        week_cols=week_cols,
        max_plot=None,
        major_threshold=major_threshold,
    )

    # plot and save
    plot_matches(
        arr=arr,
        pmo_flags=pmo_flags,
        cumul=cumul,
        reached=reached,
        hit_idx=hit_idx,
        n_weeks=n_weeks,
        plotted=plotted,
        n_matches=n_matches_total,
        n_total=n_total,
        pmo_fraction=pmo_fraction,
        major_threshold=major_threshold,
        observed=observed,
        out_png=out_png,
        figsize=figsize,
    )

    # Save indices + sampled dataframe for later reuse.
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    idx_path = out_path.parent / f"{stem}_match_indices.csv"
    sampled_matches_path = out_path.parent / f"{stem}_sampled_matches.csv"

    # Save the original match indices (one column). Use int dtype so it's easy to read back.
    pd.DataFrame({"match_index": sampled_df["match_index"].astype(int)}).to_csv(idx_path, index=False)

    # Save sampled rows (including the match_index column and any other metadata like PMO, R_draw, week_*)
    sampled_df.to_csv(sampled_matches_path, index=False)

    # Print where we saved them
    print(f"Saved sampled matches dataframe to: {sampled_matches_path}")
    print(f"Saved sampled match indices to: {idx_path}")

    # Return the main PNG path and the two auxiliary paths in case caller wants them
    return out_png, str(sampled_matches_path), str(idx_path)

def main():
    # uses module-level defaults unless overridden by editing this file
    out = run_plot_matches()
    print(f"Saved matched trajectories plot to {out}")


if __name__ == "__main__":
    main()

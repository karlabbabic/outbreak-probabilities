# src/outbreak_probabilities/trajectory_matching/pmo_vs_r_refractor.py
"""
Plot PMO as a function of the sampled matched trajectories (r = 1..R).

Behavior:
- Loads matched trajectories via trajectory_match_pmo (same contract as plot_matches).
- Samples matched trajectories according to a selection strategy (random, highest_peak, etc.).
- Optionally sorts the sampled trajectories by a sorting key (preserve sample order by default).
- Computes cumulative PMO fraction for r = 1..R:
      pmo_r[r] = (# of PMO==1 among first r sampled trajectories) / r
- Saves PNG with PMO fraction vs r (x-axis is trajectory index r).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# relative import for intra-package usage
from .trajectory import trajectory_match_pmo

# ----------------------
# Default configuration (can be overridden by run_pmo_vs_r)
# ----------------------
SIM_CSV: str = "data/test_simulations.csv"
OBSERVED: List[int] = [1, 2, 0]
HEADER_ROWS: int = 3
WEEK_PREFIX: str = "week_"
MAJOR_THRESHOLD: int = 100
OUT_PNG: str = "figs/pmo_vs_r.png"

SAMPLE_STRATEGY: str = "random"
SAMPLE_SIZE: Optional[int] = 200
MAX_PLOT: Optional[int] = None
FIGSIZE: Tuple[int, int] = (8, 5)

LINEWIDTH: float = 2.0
MARKERSIZE: float = 4.0
ALPHA: float = 0.85
BLUE: str = "tab:blue"
GRAY: str = "dimgray"
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


# Reuse selection helper from plot_matches to keep behavior identical
def select_indices(df: pd.DataFrame, week_cols: List[str], strategy: str, sample_size: Optional[int],
                   hybrid_k: int = 25, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Return array of indices to select according to strategy.
    Strategies supported:
      - random
      - highest_cumulative
      - highest_peak
      - highest_R
      - hybrid
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


def prepare_sample(matches_df: pd.DataFrame, week_cols: List[str], sample_strategy: str, sample_size: Optional[int],
                   random_seed: Optional[int] = None):
    """
    Sample matches_df according to strategy and return:
      sampled_df (pandas DataFrame of the sampled matches)
    """
    n_matches_total = len(matches_df)
    if sample_size is None or sample_size >= n_matches_total:
        sampled_df = matches_df.copy().reset_index(drop=True)
    else:
        sel_idx = select_indices(matches_df, week_cols, strategy=sample_strategy, sample_size=sample_size, random_seed=random_seed)
        # preserve selection order (important for random strategy)
        # if selection came from argsort-like methods, sel_idx will already be ordered by the metric
        sel_idx = np.array(sel_idx, dtype=int)
        sampled_df = matches_df.iloc[sel_idx].reset_index(drop=True)
    return sampled_df


def sort_sampled_df(sampled_df: pd.DataFrame, week_cols: List[str], sort_by: str = "sample_order") -> pd.DataFrame:
    """
    Sort the sampled dataframe. Options:
      - 'sample_order' : keep existing sampled order (default)
      - 'by_cumulative' : descending cumulative cases across all weeks
      - 'by_peak' : descending peak weekly cases
      - 'by_R' : descending R_draw (if exists)
      - 'by_PMO' : descending PMO (put PMO==1 first)
    Returns a new DataFrame (reset_index).
    """
    if sort_by == "sample_order":
        return sampled_df.reset_index(drop=True)

    if sort_by == "by_cumulative":
        key = sampled_df[week_cols].sum(axis=1)
        return sampled_df.assign(_key=key).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)

    if sort_by == "by_peak":
        key = sampled_df[week_cols].max(axis=1)
        return sampled_df.assign(_key=key).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)

    if sort_by == "by_R":
        if "R_draw" not in sampled_df.columns:
            return sampled_df.reset_index(drop=True)
        key = sampled_df["R_draw"]
        return sampled_df.assign(_key=key).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)

    if sort_by == "by_PMO":
        # place PMO==1 first
        return sampled_df.assign(_key=sampled_df["PMO"].astype(int)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)

    raise ValueError(f"Unknown sort_by value: {sort_by}")


def compute_pmo_vs_r(sampled_df: pd.DataFrame) -> np.ndarray:
    """
    Given sampled_df (ordered), compute cumulative PMO fraction for r=1..R:
      pmo_r[r-1] = (# PMO==1 among first r rows) / r
    Returns numpy array length R with floats in [0,1].
    """
    pmo_flags = sampled_df["PMO"].astype(int).to_numpy()
    # cumulative sum of PMO ones
    cumsum = np.cumsum(pmo_flags, dtype=float)
    r = np.arange(1, pmo_flags.size + 1, dtype=float)
    return cumsum / r


def plot_pmo_vs_r(
    pmo_r: np.ndarray,
    out_png: str,
    figsize: Tuple[int, int],
    observed: List[int],
    sample_strategy: str,
    sample_size: Optional[int],
    sort_by: str,
):
    """Plot and save PMO(r) vs r."""
    R = pmo_r.size
    rs = np.arange(1, R + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rs, pmo_r, marker="o", linewidth=LINEWIDTH, markersize=MARKERSIZE, alpha=ALPHA, label="PMO(r)")
    # plot horizontal line at overall PMO among sampled set:
    overall = pmo_r[-1] if R > 0 else np.nan
    if not np.isnan(overall):
        ax.axhline(overall, linestyle="--", linewidth=1.0, label=f"Overall PMO (R={R}) = {overall:.3f}")

    ax.set_xlabel("Sampled matched trajectory index r")
    ax.set_ylabel("Cumulative PMO fraction (first r trajectories)")
    obs_str = ", ".join(str(x) for x in observed) if observed else "[]"
    ax.set_title(
        f"PMO vs r (cumulative) — Initial Cases: {obs_str}\nSampling: {sample_strategy} (size={sample_size})  |  sort_by: {sort_by}"
    )
    ax.set_xlim(1, max(1, R))
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.linspace(1, max(1, R), min(10, R)) if R > 1 else [1])
    ax.grid(alpha=0.6)
    ax.spines['top'].set_visible(False)     # Hide top spine
    ax.spines['right'].set_visible(False)   # Hide right spine
    ax.spines['bottom'].set_visible(False)     # Hide top spine
    ax.spines['left'].set_visible(False)   # Hide right spine
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run_pmo_vs_r_refractor(
    sim_csv: str = SIM_CSV,
    observed: List[int] = OBSERVED,
    header_rows: int = HEADER_ROWS,
    week_prefix: str = WEEK_PREFIX,
    out_png: str = OUT_PNG,
    sample_strategy: str = SAMPLE_STRATEGY,
    sample_size: Optional[int] = SAMPLE_SIZE,
    sort_by: str = "sample_order",
    figsize: Tuple[int, int] = FIGSIZE,
    random_seed: Optional[int] = 42,
):
    """
    Top-level function to create and save the PMO vs r plot.
    Returns the path to the saved PNG.
    """
    res = load_matches(sim_csv=sim_csv, observed=observed, header_rows=header_rows, week_prefix=week_prefix)
    n_matches = res.get("n_matches", 0)
    if n_matches == 0:
        raise SystemExit("No matching trajectories found.")

    matches_df = res.get("matches_df")
    if matches_df is None:
        raise SystemExit("trajectory_match_pmo did not return 'matches_df'.")

    week_cols = get_week_columns(matches_df, week_prefix)
    if not week_cols:
        raise SystemExit("No week_* columns found in matched dataframe.")

    # 1) sample according to strategy
    sampled_df = prepare_sample(matches_df=matches_df, week_cols=week_cols,
                                sample_strategy=sample_strategy, sample_size=sample_size,
                                random_seed=random_seed)

    # 2) optionally apply MAX_PLOT trimming (keeps first MAX_PLOT in current order)
    if MAX_PLOT is not None and MAX_PLOT < len(sampled_df):
        sampled_df = sampled_df.iloc[:MAX_PLOT].reset_index(drop=True)

    # 3) sort according to user's requested ordering
    sampled_sorted = sort_sampled_df(sampled_df=sampled_df, week_cols=week_cols, sort_by=sort_by)

    # 4) compute pmo vs r
    pmo_r = compute_pmo_vs_r(sampled_sorted)

    # 5) plot and save
    plot_pmo_vs_r(
        pmo_r=pmo_r,
        out_png=out_png,
        figsize=figsize,
        observed=observed,
        sample_strategy=sample_strategy,
        sample_size=sample_size,
        sort_by=sort_by,
    )

    return out_png


def main():
    out = run_pmo_vs_r_refractor()
    print(f"Saved PMO vs r plot to {out}")


if __name__ == "__main__":
    main()

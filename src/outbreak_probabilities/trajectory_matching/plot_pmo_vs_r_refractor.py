#!/usr/bin/env python3
# src/outbreak_probabilities/trajectory_matching/pmo_vs_r_refractor.py
"""
Plot PMO as a function of sampled matched trajectories (r = 1..R)
or as a function of the full simulation index (1..N_total) with updates at matched indices.

Main function:
  run_pmo_vs_r_refractor(..., full_index=False)

- full_index=False: same as original (x-axis r = 1..R)
- full_index=True: x-axis spans 1..N_total; PMO fraction steps only at matched sim IDs
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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# import analytic helper to compute integrated PMO from initial cases
from ..analytic.analytical_refractor import compute_pmo_from_string, DEFAULT_R_MIN, DEFAULT_R_MAX

# defaults
SIM_CSV: str = "data/test_simulations.csv"
OBSERVED: List[int] = [1, 2, 0]
HEADER_ROWS: int = 3
WEEK_PREFIX: str = "week_"
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
    """Obtain matches via trajectory_match_pmo (same contract used elsewhere)."""
    return trajectory_match_pmo(
        observed_weeks=observed,
        simulated_csv=sim_csv,
        header_rows=header_rows,
        week_prefix=week_prefix,
        return_matches_df=True,
    )


def get_week_columns(df: pd.DataFrame, week_prefix: str) -> List[str]:
    week_cols = [c for c in df.columns if c.startswith(week_prefix)]
    return sorted(week_cols, key=lambda s: int(s.split(week_prefix)[1]))


def select_indices(df: pd.DataFrame, week_cols: List[str], strategy: str, sample_size: Optional[int],
                   hybrid_k: int = 25, random_seed: Optional[int] = None) -> np.ndarray:
    """Select indices in the matches dataframe according to a strategy (same logic as before)."""
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
                   random_seed: Optional[int] = None) -> pd.DataFrame:
    """Sample matches_df and preserve returned order (important for random)."""
    n_total = len(matches_df)
    if sample_size is None or sample_size >= n_total:
        return matches_df.copy().reset_index(drop=True)
    sel_idx = select_indices(matches_df, week_cols, strategy=sample_strategy, sample_size=sample_size, random_seed=random_seed)
    sel_idx = np.array(sel_idx, dtype=int)
    return matches_df.iloc[sel_idx].reset_index(drop=True)


def compute_pmo_r_from_ordered(sampled_df: pd.DataFrame) -> np.ndarray:
    pmo_flags = sampled_df["PMO"].astype(int).to_numpy()
    if pmo_flags.size == 0:
        return np.array([], dtype=float)
    cumsum = np.cumsum(pmo_flags, dtype=float)
    r = np.arange(1, pmo_flags.size + 1, dtype=float)
    return cumsum / r


def plot_pmo_vs_r(pmo_r: np.ndarray, out_png: str, figsize: Tuple[int, int], observed: List[int],
                  sample_strategy: str, sample_size: Optional[int], sort_by: str,
                  analytic_pmo: Optional[float] = None) -> str:
    """Plot PMO(r) vs r (r = 1..R). Returns saved path.

    If analytic_pmo is provided it will be shown as a single horizontal line
    computed from the observed initial cases and the simulation R-range.
    """

    R = pmo_r.size

    rs = np.arange(1, R + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rs, pmo_r, marker="o", linewidth=LINEWIDTH, markersize=MARKERSIZE, alpha=ALPHA, label="Empirical PMO(r)")
    overall = pmo_r[-1] if R > 0 else np.nan
    if not np.isnan(overall):
        ax.axhline(overall, linestyle="--", linewidth=1.0, label=f"Overall empirical PMO = {overall:.5f}")

    # analytic horizontal line (single value) if provided
    if analytic_pmo is not None and np.isfinite(analytic_pmo):
        ax.axhline(analytic_pmo, linestyle="-.", linewidth=1.25, color="xkcd:forest green",
                   label=f"Analytic PMO (observed,R-range) = {analytic_pmo:.5f}")

    ax.set_xlabel("Sampled matched trajectory index r")
    ax.set_ylabel("Cumulative PMO fraction (first r trajectories)")
    obs_str = ", ".join(str(x) for x in observed) if observed else "[]"
    ax.set_title(
        f"Cumulative PMO across matched outbreaks only \n{R} outbreaks with initial cases {obs_str}\nSampling: {sample_strategy} (size={sample_size})  |  sort_by: {sort_by}"
    )
    ax.set_xlim(1, max(1, R))
    ax.set_ylim(0, 1.0)
    if R > 1:
        ax.set_xticks(np.linspace(1, max(1, R), min(10, R)))
    else:
        ax.set_xticks([1])
    ax.grid(alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def plot_pmo_over_full_index(
    sel_sim_ids: np.ndarray,
    sel_pmo: np.ndarray,
    N_total: int,
    out_png: str,
    figsize: Tuple[int, int],
    analytic_single: Optional[float] = None,
    draw_verticals: bool = True,
    smooth: bool = True,
    max_points: int = 5000,
) -> (str, pd.DataFrame):
    """
    Plot cumulative PMO on full sim index 1..N_total.

    If analytic_single is provided (a scalar), draw it as a horizontal line for comparison.
    """
    # 1) build events dataframe and exact cumulative fractions
    if sel_sim_ids.size == 0:
        events_df = pd.DataFrame(columns=["sim_id", "PMO", "event_order", "cum_pmo"])
        # trivial flat curve
        dense_x = np.array([1, N_total], dtype=float)
        dense_y = np.zeros_like(dense_x, dtype=float)
        final_pmo = 0.0
    else:
        cumulative_counts = np.cumsum(sel_pmo)
        event_no = np.arange(1, sel_pmo.size + 1)
        cum_pmo_fraction = cumulative_counts / event_no

        events_df = pd.DataFrame({
            "sim_id": sel_sim_ids.astype(int),
            "PMO": sel_pmo.astype(int),
            "event_order": event_no.astype(int),
            "cum_pmo": cum_pmo_fraction,
        })

        # 2) create dense x grid for plotting but cap points for memory
        if N_total <= max_points:
            dense_x = np.arange(1, N_total + 1, dtype=float)
        else:
            dense_x = np.linspace(1, float(N_total), num=max_points, dtype=float)

        # 3) interpolate exact cumulative values defined at sel_sim_ids onto dense_x
        first_event = float(sel_sim_ids[0])
        last_event = float(sel_sim_ids[-1])
        final_pmo = float(cum_pmo_fraction[-1])

        event_x = sel_sim_ids.astype(float)
        event_y = cum_pmo_fraction.astype(float)

        dense_y = np.interp(dense_x, event_x, event_y, left=0.0, right=final_pmo)

    # plotting: smoothed/interpolated curve + exact event markers
    fig, ax = plt.subplots(figsize=figsize)

    # plot the (smoothed/interpolated) running PMO curve
    ax.plot(dense_x, dense_y, linewidth=LINEWIDTH, alpha=0.8, color="grey", label="Running PMO (empirical)")

    # dashed horizontal line at the FINAL (true) PMO value
    ax.axhline(
        final_pmo,
        linestyle="--",
        linewidth=1.2,
        color="xkcd:bright orange",
        alpha=0.7,
        label=f"Final empirical PMO = {final_pmo:.3f}",
    )

    # overlay exact event scatter (cum_pmo_fraction at sel_sim_ids)
    if sel_sim_ids.size:
        ax.scatter(events_df["sim_id"].values, events_df["cum_pmo"].values,
                   s=35,
                   c=[BLUE if v == 1 else GRAY for v in events_df["PMO"].values],
                   edgecolors="none", linewidths=0.1, zorder=1, label="Matched events (empirical)",
                   alpha=[0.1 if v == 1 else 0.05 for v in events_df["PMO"].values])

    # analytic horizontal line if provided
    if analytic_single is not None and np.isfinite(analytic_single):
        ax.axhline(analytic_single, linestyle="-.", linewidth=1.25, color="xkcd:forest green",
                   alpha=0.9, label=f"Analytic PMO = {analytic_single:.3f}")

    ax.set_xlim(-0.5, float(N_total) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Simulation ID (1..N_total)")
    ax.set_ylabel("Cumulative PMO fraction (running)")
    ax.set_title(f"Cumulative PMO across full simulation index\n({sel_sim_ids.size}/{N_total} matches)")

    
    # hide extra spines like other plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Minor ticks halfway between majors (but hidden)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Minor gridlines: lighter gray, thinner
    ax.grid(
        which="minor",
        color="0.85",      # lighter gray
        linewidth=0.6,
        alpha=0.9
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    out_png_full = Path(out_png).with_name(
        Path(out_png).stem + "_full_index.png")

    Path(out_png_full).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out_png, events_df


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
    full_index: bool = False,
):
    """
    Top-level function.

    - If full_index is False: behaves like original (x-axis r=1..R).
      returns (out_png_path, None)
    - If full_index is True: x-axis is 1..N_total and updates only at matched sim IDs.
      returns (out_png_path, events_df) where events_df lists sim_id, PMO, event_order, cum_pmo
    """
    res = load_matches(sim_csv=sim_csv, observed=observed, header_rows=header_rows, week_prefix=week_prefix)
    n_matches = int(res.get("n_matches", 0))
    if n_matches == 0:
        raise SystemExit("No matching trajectories found.")

    matches_df = res.get("matches_df")
    if matches_df is None:
        raise SystemExit("trajectory_match_pmo did not return 'matches_df'.")

    week_cols = get_week_columns(matches_df, week_prefix)
    if not week_cols:
        raise SystemExit("No week_* columns found in matched dataframe.")

    # ensure persistent sim ids are available (match_index) -- assume original row index if missing
    matches_df = matches_df.copy()
    if "match_index" not in matches_df.columns:
        matches_df["match_index"] = matches_df.index.astype(int)

    # 1) sample according to strategy
    sampled_df = prepare_sample(matches_df=matches_df, week_cols=week_cols,
                                sample_strategy=sample_strategy, sample_size=sample_size,
                                random_seed=random_seed)

    # 2) optional trimming
    if MAX_PLOT is not None and MAX_PLOT < len(sampled_df):
        sampled_df = sampled_df.iloc[:MAX_PLOT].reset_index(drop=True)

    # 3) apply sorting
    if sort_by != "sample_order":
        # small inline sort helper to avoid duplication
        if sort_by == "by_cumulative":
            sampled_df = sampled_df.assign(_key=sampled_df[week_cols].sum(axis=1)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_peak":
            sampled_df = sampled_df.assign(_key=sampled_df[week_cols].max(axis=1)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_R" and "R_draw" in sampled_df.columns:
            sampled_df = sampled_df.assign(_key=sampled_df["R_draw"]).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_PMO":
            sampled_df = sampled_df.assign(_key=sampled_df["PMO"].astype(int)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        else:
            # unknown sort_by falls back to sample_order
            sampled_df = sampled_df.reset_index(drop=True)
    else:
        sampled_df = sampled_df.reset_index(drop=True)

    # --- Read the full simulation CSV so we can fetch the R range if present ---
    total_df = pd.read_csv(sim_csv, header=header_rows)

    # infer R_min/R_max from the simulation file if possible (prefer common column names)
    r_col_candidates = [c for c in ("R_draw", "R", "r_draw", "r") if c in total_df.columns]
    if r_col_candidates:
        r_col = r_col_candidates[0]
        try:
            R_min_val = float(total_df[r_col].min())
            R_max_val = float(total_df[r_col].max())
            if not np.isfinite(R_min_val):
                R_min_val = DEFAULT_R_MIN
            if not np.isfinite(R_max_val):
                R_max_val = DEFAULT_R_MAX
            if R_min_val > R_max_val:
                R_min_val, R_max_val = min(R_min_val, R_max_val), max(R_min_val, R_max_val)
        except Exception:
            R_min_val, R_max_val = DEFAULT_R_MIN, DEFAULT_R_MAX
    else:
        R_min_val, R_max_val = DEFAULT_R_MIN, DEFAULT_R_MAX

    # Single-line analytic PMO: computed from observed initial cases and inferred R range.
    initial_cases_str = ",".join(str(int(x)) for x in observed) if observed else ""
    try:
        analytic_pmo = float(compute_pmo_from_string(initial_cases_str, nR=2001, R_min=R_min_val, R_max=R_max_val).get("PMO", float("nan")))
    except Exception:
        analytic_pmo = float("nan")

    # Branch: full_index mode vs sampled-index mode
    if not full_index:
        pmo_r = compute_pmo_r_from_ordered(sampled_df)
        out_path = plot_pmo_vs_r(
            pmo_r=pmo_r,
            out_png=out_png,
            figsize=figsize,
            observed=observed,
            sample_strategy=sample_strategy,
            sample_size=sample_size,
            sort_by=sort_by,
            analytic_pmo=analytic_pmo,
        )
        return out_path, None

    # full_index mode: we need N_total and the sim ids of the sampled rows
    N_total = len(total_df)

    # selected sampled rows' original sim IDs (assume match_index is 0-based row id in original sim CSV)
    sim_ids_zero_based = sampled_df["match_index"].astype(int).to_numpy()
    sim_ids_one_based = sim_ids_zero_based + 1

    # sort events by sim_id ascending
    order_by_simid = np.argsort(sim_ids_one_based)
    sel_sim_ids = sim_ids_one_based[order_by_simid]
    sel_pmo = sampled_df["PMO"].astype(int).to_numpy()[order_by_simid]

    # analytic_single is the same across events (we computed above)
    analytic_single = analytic_pmo

    out_path, events_df = plot_pmo_over_full_index(
        sel_sim_ids=sel_sim_ids,
        sel_pmo=sel_pmo,
        N_total=N_total,
        out_png=out_png,
        figsize=figsize,
        analytic_single=analytic_single,
    )

    # also save events CSV next to PNG for reproducibility
    events_csv = Path(out_path).with_name(Path(out_path).stem + "_events.csv")
    events_df.to_csv(events_csv, index=False)
    print(f"Saved events CSV to: {events_csv}")

    return out_path, events_df


def main():
    out, meta = run_pmo_vs_r_refractor()
    print(f"Saved PMO plot to {out}")
    if meta is not None:
        print("Events metadata returned (first 10 rows):")
        print(meta.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

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

def compute_running_ci(
    pmo_flags: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.90,
    random_seed: Optional[int] = None,
) -> (np.ndarray, np.ndarray):
    """
    Compute pointwise CI band for the running PMO by repeatedly shuffling the
    order of pmo_flags and computing the running cumulative fraction.

    Returns (lower, upper) arrays of length R where lower is the (alpha/2) percentile
    and upper is the (1 - alpha/2) percentile with alpha = 1 - ci.
    """
    if pmo_flags is None or pmo_flags.size == 0:
        return np.array([]), np.array([])

    R = pmo_flags.size
    rng = np.random.default_rng(random_seed)
    n_boot = max(1, int(n_boot))

    # container for boot runs (n_boot, R)
    runs = np.empty((n_boot, R), dtype=float)

    for i in range(n_boot):
        perm = rng.permutation(R)
        perm_flags = pmo_flags[perm]
        csum = np.cumsum(perm_flags, dtype=float)
        r = np.arange(1, R + 1, dtype=float)
        runs[i, :] = csum / r

    alpha = 1.0 - float(ci)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = np.percentile(runs, lower_pct, axis=0)
    upper = np.percentile(runs, upper_pct, axis=0)
    return lower, upper


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

def plot_pmo_vs_r(
    pmo_r: np.ndarray,
    out_png: str,
    figsize: Tuple[int, int],
    observed: List[int],
    sample_strategy: str,
    sample_size: Optional[int],
    sort_by: str,
    analytic_pmo: Optional[float] = None,
    pmo_flags: Optional[np.ndarray] = None,
    show_final_pmo: bool = False,
    show_ci: bool = True,
    ci: float = 0.90,
    n_boot: int = 500,
    ci_random_seed: Optional[int] = 42,
) -> str:

    """
    Plot PMO(r) vs r (r = 1..R) with the same styling as the full-index plot:
    - running empirical PMO: blue line
    - individual points: very light blue dots
    - final empirical PMO: black dashed line
    - analytic PMO: orange dashed-dotted line
    """
    BLUE = "xkcd:azure"
    ORANGE = "xkcd:bright orange"
    LIGHT_POINT_ALPHA = 0.50
    LINE_ALPHA = 0.95

    R = int(pmo_r.size)
    rs = np.arange(1, R + 1)

    # optional: compute CI band (pointwise) via shuffling bootstrap of pmo_flags
    lower = upper = None
    if show_ci and (pmo_flags is not None) and (pmo_flags.size == R) and R > 0:
        try:
            lower, upper = compute_running_ci(pmo_flags=pmo_flags, n_boot=n_boot, ci=ci, random_seed=ci_random_seed)
        except Exception:
            lower, upper = None, None

    fig, ax = plt.subplots(figsize=figsize)

    # final empirical PMO (horizontal dashed black)
    overall = float(pmo_r[-1]) if R > 0 else float("nan")

        # shaded CI band (if available)
    if lower is not None and upper is not None and lower.size == R and upper.size == R:
        ax.fill_between(
            rs,
            lower,
            upper,
            step=None,
            alpha=0.18,
            color=BLUE,
            zorder=1,
            label=f"{int(ci*100)}% band (random shuffling)",
        )

    # very light individual points (muted)
    if R > 0:
        # select up to 50 equidistant indices
        n_pts = min(50, R)
        idx = np.linspace(0, R - 1, n_pts, dtype=int)

        ax.scatter(
            rs[idx],
            pmo_r[idx],
            s=30,
            c=[BLUE] * n_pts,
            edgecolors="none",
            alpha=LIGHT_POINT_ALPHA,
            zorder=2,
            label="_nolegend_",
        )

    # running empirical PMO: blue line (solid)
    if R > 0:
        ax.plot(
            rs,
            pmo_r,
            linewidth=LINEWIDTH if "LINEWIDTH" in globals() else 2.0,
            alpha=LINE_ALPHA,
            color=BLUE,
            label=f"Running PMO = {overall:.5f}",
            zorder=3,
        )

    # if R > 0 and np.isfinite(overall) and show_final_pmo == True:
    #     print("show_final_pmo")
    #     ax.axhline(overall, linestyle="-", linewidth=2.4, color="black", alpha=0.8,
    #                label=f"Final empirical PMO = {overall:.5f}", zorder=4)
    # elif R > 0 and np.isfinite(overall) and show_final_pmo == False:
    #     ax.axhline(overall, linestyle="-", linewidth=2.4, color="black", alpha=0,
    #                label=f"Final empirical PMO = {overall:.5f}", zorder=4)

    # analytic PMO (orange dash-dot)
    if analytic_pmo is not None and np.isfinite(analytic_pmo):
        ax.axhline(analytic_pmo, linestyle="--", linewidth=1.4, color=ORANGE,
                   alpha=1.0, label=f"Analytic PMO = {analytic_pmo:.5f}", zorder=5)

    # axes, title, ticks
    ax.set_xlabel("Sampled matched trajectory index")
    ax.set_ylabel("Cumulative PMO fraction")
    obs_str = ", ".join(str(x) for x in observed) if observed else "[]"

    # guard sample_size display
    disp_sample_size = sample_size if (sample_size is None or sample_size <= R) else R

    ax.set_title(
        f"Cumulative PMO across matched outbreaks\n{R} outbreaks with initial cases {obs_str}\n"
        f"Sampling: {sample_strategy} | sort_by: {sort_by}"
    )

    ax.set_xlim(1, max(1, R))
    ax.set_ylim(-0.01, 1.02)

    # Integer, evenly spaced ticks using linspace (safe version)
    if R > 1:
        n_ticks = min(10, R)
        ticks = np.linspace(1, R, n_ticks)
        ticks = np.unique(np.round(ticks).astype(int))
        ticks[0] = 1
        ticks[-1] = R
        ax.set_xticks(ticks)
    else:
        ax.set_xticks([1])


    # styling consistent with full-index plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, which="major", linestyle="--")

    # compact legend
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
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
    # NEW params used for title consistency
    observed: Optional[List[int]] = None,
    sample_strategy: str = "unknown",
    sort_by: str = "sample_order",
    n_matches: Optional[int] = None,
    # CI options
    show_ci: bool = True,
    ci: float = 0.90,
    n_boot: int = 500,
    ci_random_seed: Optional[int] = 42,
) -> (str, pd.DataFrame):
    """
    Plot cumulative PMO on full sim index 1..N_total.

    - The running PMO is interpolated (step-like) so it updates at matched sim IDs.
    - A pointwise CI band (by random shuffling of matched PMO flags) is computed at
      event-order positions and interpolated to the dense full-index x-grid.
    - Title mirrors the sampled-index plot format.
    """
    BLUE = "xkcd:azure"
    ORANGE = "xkcd:bright orange"
    LIGHT_POINT_ALPHA = 0.10
    LINE_ALPHA = 0.95

    # build events dataframe and exact cumulative fractions at event order (1..m)
    if sel_sim_ids.size == 0:
        events_df = pd.DataFrame(columns=["sim_id", "PMO", "event_order", "cum_pmo"])
        dense_x = np.array([1, N_total], dtype=float)
        dense_y = np.zeros_like(dense_x, dtype=float)
        final_pmo = 0.0
        dense_lower = dense_upper = np.zeros_like(dense_x, dtype=float)
    else:
        m = sel_pmo.size
        cumulative_counts = np.cumsum(sel_pmo)
        event_no = np.arange(1, m + 1)
        cum_pmo_fraction = cumulative_counts / event_no

        events_df = pd.DataFrame({
            "sim_id": sel_sim_ids.astype(int),
            "PMO": sel_pmo.astype(int),
            "event_order": event_no.astype(int),
            "cum_pmo": cum_pmo_fraction,
        })

        # dense x grid (full simulation index) - integer positions where possible
        if N_total <= max_points:
            dense_x = np.arange(1, N_total + 1, dtype=float)
        else:
            dense_x = np.linspace(1.0, float(N_total), num=max_points, dtype=float)

        # final PMO at last event
        final_pmo = float(cum_pmo_fraction[-1])

        # event_x are the sim IDs (1-based) and event_y are cum_pmo_fraction
        event_x = sel_sim_ids.astype(float)
        event_y = cum_pmo_fraction.astype(float)

        # interpolate the step-like curve: between events the cumulative PMO stays at the last event value
        # we produce a dense_y by interpolation but ensure left=0 and right=final_pmo
        dense_y = np.interp(dense_x, event_x, event_y, left=0.0, right=final_pmo)

        # ----- compute CI at event-order positions (if requested) -----
        dense_lower = dense_upper = None
        if show_ci and m > 0:
            try:
                # compute CI at event order length m using same helper as sampled-index
                lower_evt, upper_evt = compute_running_ci(pmo_flags=sel_pmo.astype(int), n_boot=n_boot, ci=ci, random_seed=ci_random_seed)
                # lower_evt/upper_evt are length m arrays aligned to event order 1..m
                # interpolate them to dense_x (same approach as for dense_y)
                # when event_x has duplicates or not strictly increasing, np.interp still works because sel_sim_ids are sorted
                dense_lower = np.interp(dense_x, event_x, lower_evt, left=0.0, right=final_pmo)
                dense_upper = np.interp(dense_x, event_x, upper_evt, left=0.0, right=final_pmo)
            except Exception:
                dense_lower = dense_upper = None

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    # CI band (interpolated) plotted first so line sits on top
    if (dense_lower is not None) and (dense_upper is not None):
        ax.fill_between(dense_x, dense_lower, dense_upper, step=None, alpha=0.18, color=BLUE, zorder=1,
                        label=f"{int(ci*100)}% band (random shuffling)")


        # overlay exact event scatter: very light filled dots, colored by PMO but muted
    if sel_sim_ids.size:
        mask_pos = events_df["PMO"].astype(int) == 1
        mask_neg = ~mask_pos

        # positive PMO (blue) — label "A"
        if mask_pos.any():
            ax.scatter(
                events_df.loc[mask_pos, "sim_id"].values,
                events_df.loc[mask_pos, "cum_pmo"].values,
                s=36,
                c=BLUE,
                edgecolors="none",
                alpha=0,
                zorder=4,
                label="Major outbreak",
            )

        # negative PMO (light grey) — label "B"
        if mask_neg.any():
            ax.scatter(
                events_df.loc[mask_neg, "sim_id"].values,
                events_df.loc[mask_neg, "cum_pmo"].values,
                s=36,
                c="lightgrey",
                edgecolors="none",
                alpha=0,
                zorder=4,
                label="Outbreak extinction",
            )

    # running empirical PMO: blue line (smoothed/interpolated)
    ax.plot(dense_x, dense_y, linewidth=LINEWIDTH, alpha=LINE_ALPHA, color=BLUE, label=f"Running PMO = {final_pmo:.5f}", zorder=2)


    # analytic horizontal line if provided: orange
    if analytic_single is not None and np.isfinite(analytic_single):
        ax.axhline(analytic_single, linestyle="--", linewidth=1.5, color=ORANGE, alpha=1.0,
                   label=f"Analytic PMO = {analytic_single:.3f}", zorder=5)

    ax.set_xlim(1, float(N_total) + 0.5)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("Simulation ID (1..N_total)")
    ax.set_ylabel("Cumulative PMO fraction")

    # --- Title: mirror the sampled-index title style ---
    obs_str = ", ".join(str(x) for x in (observed or [])) if observed is not None else "[]"
    n_matches_local = int(n_matches) if (n_matches is not None) else int(sel_sim_ids.size)
    ax.set_title(
        f"Cumulative PMO across matched outbreaks over full simulation\n{n_matches_local} outbreaks with initial cases {obs_str}\n")

    # style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, which="major", linestyle="--")

    # show legend (only for main lines)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    out_png_full = Path(out_png).with_name(Path(out_png).stem + "_full_index.png")
    Path(out_png_full).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_full, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(out_png_full), events_df



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
    show_final_pmo: bool = False, # disable horizontal line of running pmo
    show_ci: bool = True,        # enable band
    ci: int = 0.90,             # 90% band
    n_boot: int = 500,          # reduce/increase as needed (trade-off speed vs smoothness)
    ci_random_seed: int = 42,
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
        pmo_flags = sampled_df["PMO"].astype(int).to_numpy()
        out_path = plot_pmo_vs_r(
            pmo_r=pmo_r,
            out_png=out_png,
            figsize=figsize,
            observed=observed,
            sample_strategy=sample_strategy,
            sample_size=sample_size,
            sort_by=sort_by,
            analytic_pmo=analytic_pmo,
            pmo_flags=pmo_flags,
            show_final_pmo=show_final_pmo, # disable horizontal line of running pmo
            show_ci=show_ci,        # enable band
            ci=ci,             # 90% band
            n_boot=n_boot,          # reduce/increase as needed (trade-off speed vs smoothness)
            ci_random_seed=ci_random_seed,
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
        observed=observed,
        sample_strategy=sample_strategy,
        sort_by=sort_by,
        n_matches=sel_sim_ids.size,
        show_ci=show_ci,
        ci=ci,
        n_boot=n_boot,
        ci_random_seed=ci_random_seed,
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

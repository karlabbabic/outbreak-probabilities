#!/usr/bin/env python3
"""
Fast combined plotting for large N.

- PMO vs R (binned)
- Trajectories with cutoff markers, but sampled smartly for speed:
    strategies: random, highest_cumulative, highest_peak, highest_R, hybrid
- Uses LineCollection for fast rendering of many lines
- Adds options to overlay mean and 10-90% band
- Non-interactive backend, saves PNGs

- Show extreme only
python plot_traj.py --sample-strategy highest_cumulative --sample-size 100

- Hybrid sampling with reproducibility
python plot_traj.py --sample-size 300 --random-seed 123

- Overlay mean, quartile 
python plot_traj.py --overlay-mean --overlay-quantiles

- Change input file 
python plot_traj.py --csv data/simulated_cases_bench_4.csv
"""

from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------- IO and helpers ----------

def load_sim_csv(path: str, header_rows: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, header=header_rows)
    week_cols = [c for c in df.columns if c.startswith("week_")]
    week_cols = sorted(week_cols, key=lambda s: int(s.split("_")[1])) if week_cols else []
    return df, week_cols

def compute_binned_pmo(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    if "R_draw" not in df.columns or "PMO" not in df.columns:
        raise ValueError("Dataframe must contain 'R_draw' and 'PMO' columns")
    r_min, r_max = df["R_draw"].min(), df["R_draw"].max()
    bins = np.linspace(r_min, r_max, n_bins + 1)
    dfc = df.copy()
    dfc["R_bin"] = pd.cut(dfc["R_draw"], bins=bins, include_lowest=True)
    summary = (
        dfc.groupby("R_bin", observed=True)
           .agg(PMO_mean=("PMO", "mean"), count=("PMO", "size"))
           .reset_index()
    )
    summary["R_mid"] = summary["R_bin"].apply(lambda b: 0.5 * (b.left + b.right))
    return summary

# ---------- fast trajectory sampling & plotting ----------

def select_indices(df: pd.DataFrame, week_cols: List[str], strategy: str, sample_size: int, hybrid_k: int = 25, random_seed: Optional[int] = None):
    """
    Return array of indices to plot according to strategy.
    hybrid_k: how many top-per-extreme to always include for 'hybrid'
    """
    n = len(df)
    rng = np.random.default_rng(random_seed)

    if sample_size is None or sample_size >= n:
        return np.arange(n)

    if strategy == "random":
        return rng.choice(n, size=sample_size, replace=False)

    # compute extremes
    cumulative = df[[c for c in week_cols]].sum(axis=1).values
    peak = df[[c for c in week_cols]].max(axis=1).values
    Rdraw = df["R_draw"].values if "R_draw" in df.columns else np.zeros(n)

    idx = set()

    if strategy == "highest_cumulative":
        idx = set(np.argsort(-cumulative)[:sample_size])
        return np.fromiter(idx, dtype=int)

    if strategy == "highest_peak":
        idx = set(np.argsort(-peak)[:sample_size])
        return np.fromiter(idx, dtype=int)

    if strategy == "highest_R":
        idx = set(np.argsort(-Rdraw)[:sample_size])
        return np.fromiter(idx, dtype=int)

    if strategy == "hybrid":
        # always include top hybrid_k of extremes
        k = min(hybrid_k, sample_size//4 or 1)
        top_c = list(np.argsort(-cumulative)[:k])
        top_p = list(np.argsort(-peak)[:k])
        top_r = list(np.argsort(-Rdraw)[:k])
        idx_set = set(top_c + top_p + top_r)
        remaining = sample_size - len(idx_set)
        if remaining > 0:
            # choose remaining at random from the rest
            pool = np.setdiff1d(np.arange(n), np.fromiter(idx_set, int))
            choice = rng.choice(pool, size=remaining, replace=False)
            idx_set.update(choice.tolist())
        return np.fromiter(idx_set, dtype=int)

    raise ValueError(f"Unknown sampling strategy: {strategy}")

def mask_after_threshold_array(weeks_arr: np.ndarray, threshold: int) -> np.ndarray:
    weeks = weeks_arr.astype(float)
    cumulative = np.cumsum(weeks, axis=1)
    mask = np.zeros_like(weeks, dtype=bool)
    n_sim, n_weeks = weeks.shape
    for i in range(n_sim):
        reached = np.where(cumulative[i, :] >= threshold)[0]
        if reached.size > 0:
            first_idx = reached[0]
            if first_idx + 1 < n_weeks:
                mask[i, first_idx + 1 :] = True
    weeks[mask] = np.nan
    return weeks

def fast_plot_trajectories(
    df: pd.DataFrame,
    week_cols: List[str],
    save_path: str = "figs/weekly_trajectories.png",
    sample_strategy: str = "hybrid",
    sample_size: int = 200,
    major_threshold: int = 100,
    overlay_mean: bool = True,
    overlay_quantiles: Tuple[float, float] = (0.10, 0.90),
    random_seed: Optional[int] = 42,
    figsize: Tuple[int,int] = (10,6),
):
    """
    Fast plotting of many trajectories:
    - selects indices by strategy
    - uses LineCollection to draw many lines efficiently
    - marks cutoff points with red dots for the plotted subset
    - optionally overlays mean and quantile ribbon computed across all trajectories (masked after threshold)
    """
    dfc = df.copy()
    n_total = len(dfc)
    if not week_cols:
        raise ValueError("No week columns provided")

    weeks = np.arange(1, len(week_cols) + 1)
    arr = dfc[week_cols].to_numpy(dtype=float)

    # masked array for mean/quantiles: censor after threshold
    arr_masked = mask_after_threshold_array(arr, major_threshold)

    # compute overlay statistics (over all sims, using masked array)
    if overlay_mean:
        mean_per_week = np.nanmean(arr_masked, axis=0)
    else:
        mean_per_week = None
    if overlay_quantiles:
        q_lo = np.nanpercentile(arr_masked, overlay_quantiles[0]*100, axis=0)
        q_hi = np.nanpercentile(arr_masked, overlay_quantiles[1]*100, axis=0)
    else:
        q_lo = q_hi = None

    # select indices to plot
    sel_idx = select_indices(dfc, week_cols, strategy=sample_strategy, sample_size=sample_size, random_seed=random_seed)
    sel_idx = np.unique(sel_idx)  # ensure ordering/uniqueness
    sel_arr = arr[sel_idx, :]

    # For plotting, truncate each selected trajectory at cutoff (so lines end where they should)
    segs = []
    cutoff_points_x = []
    cutoff_points_y = []
    for row in sel_arr:
        cumulative = np.cumsum(row)
        hit = np.where(cumulative >= major_threshold)[0]
        if hit.size > 0:
            cutoff = int(hit[0])
            xs = weeks[: cutoff + 1]
            ys = row[: cutoff + 1]
            segs.append(np.column_stack([xs, ys]))
            cutoff_points_x.append(xs[-1])
            cutoff_points_y.append(ys[-1])
        else:
            xs = weeks
            ys = row
            segs.append(np.column_stack([xs, ys]))

    # Create LineCollection for speed
    lc = LineCollection(segs, linewidths=0.9, colors=(0.5, 0.5, 0.5, 0.35), zorder=1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.autoscale()

    # red dots for cutoffs
    if cutoff_points_x:
        ax.scatter(cutoff_points_x, cutoff_points_y, color="red", s=20, zorder=3, label="cutoff")

    # overlay mean and quantile ribbon
    if overlay_mean and mean_per_week is not None:
        ax.plot(weeks, mean_per_week, color="#1f77b4", linewidth=2, label="mean")

    if q_lo is not None and q_hi is not None:
        ax.fill_between(weeks, q_lo, q_hi, color="#7f8fa6", alpha=0.25, label=f"{int(overlay_quantiles[0]*100)}-{int(overlay_quantiles[1]*100)}%")

    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    ax.set_title(f"Weekly trajectories (cutoff ≥ {major_threshold}) — plotted {len(sel_idx)} of {n_total}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fast trajectories plot to {save_path}")

# ---------- PMO plot (same as before) ----------

def plot_pmo_vs_R(df: pd.DataFrame, n_bins: int = 20, save_path: str = "figs/pmo_vs_r.png"):
    binned = compute_binned_pmo(df, n_bins=n_bins)
    plt.figure(figsize=(8,5))
    jitter = (np.random.rand(len(df)) - 0.5) * 0.02
    plt.scatter(df["R_draw"], df["PMO"] + jitter, alpha=0.25, s=18, label="raw outcomes")
    plt.plot(binned["R_mid"], binned["PMO_mean"], marker="o", linewidth=2, label="binned PMO (mean)")
    plt.xlabel("R")
    plt.ylabel("Probability of Major Outbreak (PMO)")
    plt.title("PMO vs R")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PMO plot to {save_path}")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Fast plotting for simulation outputs")
    p.add_argument("--csv", default="data/test_simulations.csv")
    p.add_argument("--header-rows", type=int, default=3)
    p.add_argument("--out-pmo", default="figs/pmo_vs_r.png")
    p.add_argument("--out-traj", default="figs/weekly_trajectories.png")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--major-threshold", type=int, default=100)
    p.add_argument("--sample-strategy", choices=["random","highest_cumulative","highest_peak","highest_R","hybrid"], default="hybrid")
    p.add_argument("--sample-size", type=int, default=200, help="How many trajectories to plot (hybrid will ensure extremes included)")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--overlay-mean", action="store_true", help="Overlay mean on trajectories")
    p.add_argument("--overlay-quantiles", action="store_true", help="Overlay 10-90 percentile quantile band")
    return p.parse_args()

def main():
    args = parse_args()
    df, week_cols = load_sim_csv(args.csv, header_rows=args.header_rows)
    print(f"Loaded {args.csv}: {len(df)} sims, weeks found: {len(week_cols)}")

    # PMO plot
    try:
        plot_pmo_vs_R(df, n_bins=args.bins, save_path=args.out_pmo)
    except Exception as e:
        print("PMO plot failed:", e)

    # fast trajectories plot
    try:
        overlay_q = (0.1, 0.9) if args.overlay_quantiles else None
        fast_plot_trajectories(
            df,
            week_cols,
            save_path=args.out_traj,
            sample_strategy=args.sample_strategy,
            sample_size=args.sample_size,
            major_threshold=args.major_threshold,
            overlay_mean=args.overlay_mean,
            overlay_quantiles=overlay_q,
            random_seed=args.random_seed,
        )
    except Exception as e:
        print("Trajectories plot failed:", e)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# src/outbreak_probabilities/trajectory_matching/pmo_error_grid.py
"""
Compute and plot the RELATIVE error ((empirical - analytic) / analytic) for 3-week initial
conditions where week1 is fixed to 1 and week2/week3 vary from 0..max_digit.

Additionally records how many matched samples (r) are required for the running
relative error to become smaller than a threshold (CLI: --rel-error-threshold).
If analytic PMO is (near) zero, we fall back to absolute error.

Outputs:
  * CSV with rows: w2, w3, analytic_pmo, empirical_pmo, rel_error_final, n_matches,
                   time_to_threshold_r, time_to_threshold_frac
  * PNG heatmap of relative error over the (w2,w3) grid
  * PNG heatmap of time-to-threshold (in number of matched samples)
  * JSON summary

Usage (example):
    PYTHONPATH=src python -m outbreak_probabilities.trajectory_matching.pmo_error_grid \
        --sim-csv data/test_simulations.csv \
        --out-dir figs/pmo_error_grid \
        --max-digit 9 \
        --sample-size 500 \
        --nR 2001 \
        --rel-error-threshold 0.01
"""
from pathlib import Path
from typing import Tuple, Optional, List
import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd
import matplotlib

# If running headless (CI), use Agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# try to use tqdm for nicer progress bars, otherwise fallback to prints
try:
    from tqdm import tqdm
    TQDM = tqdm
except Exception:  # pragma: no cover - tqdm optional
    TQDM = lambda x, **k: x

# relative imports from package
from .plot_pmo_vs_r_refractor import get_week_columns, prepare_sample, compute_pmo_r_from_ordered, load_matches
from ..analytic.analytical_refractor import compute_pmo_from_string, DEFAULT_R_MIN, DEFAULT_R_MAX

# defaults
DEFAULT_SIM_CSV = "data/test_simulations.csv"
DEFAULT_OUT_DIR = "figs/pmo_error_grid"
DEFAULT_MAX_DIGIT = 9
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_NR = 2001  # resolution for analytic integration over R grid
DEFAULT_REL_ERROR_THRESHOLD = 0.01  # 1% relative error threshold


# helpers -------------------------------------------------------------

def compute_analytic_pmo_for_obs(obs: List[int], nR: int = DEFAULT_NR,
                                 R_min: float = DEFAULT_R_MIN, R_max: float = DEFAULT_R_MAX) -> float:
    """
    Compute analytic PMO by calling compute_pmo_from_string (wrapper in analytic_refractor).
    Returns float PMO or nan if failure.
    """
    if not obs:
        return float("nan")
    obs_str = ",".join(str(int(x)) for x in obs)
    try:
        res = compute_pmo_from_string(obs_str, nR=nR, R_min=R_min, R_max=R_max)
        pmo = float(res.get("PMO", float("nan")))
        return pmo
    except Exception:
        return float("nan")


def compute_empirical_pmo_for_obs(
    sim_csv: str,
    observed: List[int],
    header_rows: int = 3,
    week_prefix: str = "week_",
    sample_strategy: str = "random",
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = 1234,
) -> Tuple[np.ndarray, int]:
    """
    Use trajectory_match_pmo via load_matches and compute the empirical running PMO.

    Returns (pmo_r, n_matches)
      * pmo_r is a numpy array with running PMO values in sampling/selection order.
        If there are no matches returns empty array.
      * n_matches is number of matching simulated trajectories
    """
    # get matches (trajectory_match_pmo wrapper)
    try:
        res = load_matches(sim_csv=sim_csv, observed=observed, header_rows=header_rows, week_prefix=week_prefix)
    except Exception:
        return np.array([]), 0

    n_matches = int(res.get("n_matches", 0))
    if n_matches == 0:
        return np.array([]), 0

    matches_df = res.get("matches_df")
    if matches_df is None or matches_df.shape[0] == 0:
        return np.array([]), 0

    # sample (keeps order; prepare_sample from pmo_vs_r_refractor)
    week_cols = get_week_columns(matches_df, week_prefix)
    sampled_df = prepare_sample(matches_df=matches_df, week_cols=week_cols,
                                sample_strategy=sample_strategy, sample_size=sample_size,
                                random_seed=random_seed)

    # compute empirical running PMO
    pmo_r = compute_pmo_r_from_ordered(sampled_df)  # numpy array, length = sampled rows
    return pmo_r, n_matches


def compute_relative_error_array(pmo_r: np.ndarray, analytic_pmo: float) -> np.ndarray:
    """
    Given running empirical pmo array and scalar analytic_pmo, compute
    relative_error_array = (pmo_r - analytic_pmo) / analytic_pmo
    If analytic_pmo is zero or near-zero, fall back to absolute error:
      relative_error_array = pmo_r - analytic_pmo
    Returns numpy array of same shape as pmo_r (may be empty).
    """
    if pmo_r.size == 0:
        return np.array([])
    if not math.isfinite(analytic_pmo):
        return np.full_like(pmo_r, np.nan, dtype=float)
    # threshold for treating analytic as zero
    if abs(analytic_pmo) < 1e-12:
        # fallback to absolute error
        return pmo_r - analytic_pmo
    return (pmo_r - analytic_pmo) / analytic_pmo


def first_index_below_threshold(arr: np.ndarray, threshold: float) -> Optional[int]:
    """
    Return first 1-based index r where abs(arr[r-1]) <= threshold.
    If not found, return None.
    If arr empty, return None.
    """
    if arr.size == 0:
        return None
    abs_arr = np.abs(arr)
    idx = np.argmax(abs_arr <= threshold)  # returns 0 if first element satisfies, or 0 when none satisfy
    if abs_arr[idx] <= threshold:
        return int(idx + 1)  # 1-based
    return None


# main analysis routine ----------------------------------------------

def build_grid_and_evaluate(
    sim_csv: str,
    out_dir: str,
    max_digit: int = DEFAULT_MAX_DIGIT,
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    header_rows: int = 3,
    week_prefix: str = "week_",
    random_seed: Optional[int] = 1234,
    nR: int = DEFAULT_NR,
    R_min: Optional[float] = None,
    R_max: Optional[float] = None,
    rel_error_threshold: float = DEFAULT_REL_ERROR_THRESHOLD,
) -> Path:
    """
    Evaluate analytic vs empirical PMO on grid (w2=0..max_digit, w3=0..max_digit),
    where observed = [1, w2, w3].

    Writes CSV results and two heatmap PNGs to out_dir; returns path to CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_csv = str(sim_csv)

    # Infer R bounds from analytic defaults if not provided
    R_min = DEFAULT_R_MIN if R_min is None else float(R_min)
    R_max = DEFAULT_R_MAX if R_max is None else float(R_max)

    rows = []
    grid_w2 = list(range(0, max_digit + 1))
    grid_w3 = list(range(0, max_digit + 1))

    total_iters = len(grid_w2) * len(grid_w3)

    iterator = TQDM([(w2, w3) for w2 in grid_w2 for w3 in grid_w3],
                    desc="Evaluating initial conditions", total=total_iters)

    for w2, w3 in iterator:
        observed = [1, int(w2), int(w3)]

        # analytic
        analytic_pmo = compute_analytic_pmo_for_obs(observed, nR=nR, R_min=R_min, R_max=R_max)

        # empirical running series and count
        pmo_r, n_matches = compute_empirical_pmo_for_obs(
            sim_csv=sim_csv,
            observed=observed,
            header_rows=header_rows,
            week_prefix=week_prefix,
            sample_strategy="random",
            sample_size=sample_size,
            random_seed=random_seed,
        )

        # final empirical (last running value) if present
        empirical_final = float(pmo_r[-1]) if (pmo_r.size > 0 and math.isfinite(pmo_r[-1])) else float("nan")

        # compute relative error array and final relative error
        rel_error_arr = compute_relative_error_array(pmo_r, analytic_pmo)
        if rel_error_arr.size > 0 and math.isfinite(rel_error_arr[-1]):
            rel_error_final = float(rel_error_arr[-1])
        else:
            rel_error_final = float("nan")

        # find first r where abs(relative error) <= threshold
        r_first = first_index_below_threshold(rel_error_arr, rel_error_threshold)
        if r_first is None:
            time_to_threshold_r = float("nan")
            time_to_threshold_frac = float("nan")
        else:
            time_to_threshold_r = int(r_first)
            # if n_matches==0, avoid division by zero
            time_to_threshold_frac = float(time_to_threshold_r) / float(min(n_matches, max(1, int(np.maximum(1, n_matches)))))

        rows.append({
            "w2": int(w2),
            "w3": int(w3),
            "observed_str": f"{observed[0]},{observed[1]},{observed[2]}",
            "analytic_pmo": float(analytic_pmo) if math.isfinite(analytic_pmo) else float("nan"),
            "empirical_pmo": empirical_final,
            "rel_error_final": rel_error_final,
            "n_matches": int(n_matches),
            "time_to_threshold_r": time_to_threshold_r,
            "time_to_threshold_frac": time_to_threshold_frac,
        })

        # progress bar update text (if tqdm available)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"last_obs": f"{observed}", "n_matches": n_matches,
                                  "rel_err": f"{rel_error_final:.3g}" if math.isfinite(rel_error_final) else "nan"})

    # Build DataFrame and write CSV
    df = pd.DataFrame(rows)
    csv_out = out_dir / f"pmo_error_grid_w2_0-{max_digit}_w3_0-{max_digit}.csv"
    df.to_csv(csv_out, index=False)

    # Pivot for relative error heatmap (w2 rows, w3 columns)
    pivot_rel = df.pivot(index="w2", columns="w3", values="rel_error_final")
    counts = df.pivot(index="w2", columns="w3", values="n_matches")
    pivot_time = df.pivot(index="w2", columns="w3", values="time_to_threshold_r")

    # --- Heatmap 1: relative error ---
    fig, ax = plt.subplots(figsize=(8, 6))
    # use diverging colormap centered at 0
    # compute vlim ignoring NaNs
    rel_vals = pivot_rel.values if pivot_rel.size else np.array([[0.0]])
    if pivot_rel.size:
        # choose symmetric vlim around 0 using percentile to avoid extreme outliers dominating
        finite = np.abs(rel_vals[np.isfinite(rel_vals)])
        if finite.size:
            vlim = np.percentile(finite, 98)
            vlim = max(vlim, 1e-6)
        else:
            vlim = 1.0
    else:
        vlim = 1.0
    im = ax.imshow(pivot_rel.values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)

    # annotate cells with n_matches if small grid; otherwise only show ticks
    annotate = (max_digit <= 12)
    if annotate:
        for i, w2 in enumerate(sorted(pivot_rel.index)):
            for j, w3 in enumerate(sorted(pivot_rel.columns)):
                val = pivot_rel.loc[w2, w3]
                cnt = counts.loc[w2, w3]
                if not (pd.isna(val) or not np.isfinite(val)):
                    ax.text(j, i, f"{val:.3f}\n({int(cnt)})", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, f"n={int(cnt)}", ha="center", va="center", fontsize=7, color="gray")

    ax.set_xticks(np.arange(len(pivot_rel.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot_rel.columns])
    ax.set_yticks(np.arange(len(pivot_rel.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot_rel.index])
    ax.set_xlabel("week 3 initial cases (w3)")
    ax.set_ylabel("week 2 initial cases (w2)")
    ax.set_title(f"Relative error: (empirical - analytic) / analytic\nObserved start: [1, w2, w3]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative error (fraction)")

    png_rel_out = csv_out.with_suffix(".rel_error_heatmap.png")
    fig.tight_layout()
    fig.savefig(png_rel_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Heatmap 2: time to threshold (r) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # use a sequential colormap; mask NaNs
    time_vals = pivot_time.values if pivot_time.size else np.array([[np.nan]])
    # choose vmax as maybe 99th percentile to avoid long tail; but ensure at least 1
    finite_times = time_vals[np.isfinite(time_vals)]
    if finite_times.size:
        vmax_time = int(max(1, int(np.percentile(finite_times, 98))))
    else:
        vmax_time = 1

    # For visualization, set NaNs to a sentinel (we will mask them)
    mask = ~np.isfinite(time_vals)
    # show masked array with imshow by using ma.masked_invalid
    im2 = ax2.imshow(np.ma.masked_invalid(time_vals), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax_time)

    # annotate small grids
    if annotate:
        for i, w2 in enumerate(sorted(pivot_time.index)):
            for j, w3 in enumerate(sorted(pivot_time.columns)):
                val = pivot_time.loc[w2, w3]
                if pd.isna(val) or not np.isfinite(val):
                    ax2.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                else:
                    ax2.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=8)

    ax2.set_xticks(np.arange(len(pivot_time.columns)))
    ax2.set_xticklabels([str(int(x)) for x in pivot_time.columns])
    ax2.set_yticks(np.arange(len(pivot_time.index)))
    ax2.set_yticklabels([str(int(x)) for x in pivot_time.index])
    ax2.set_xlabel("week 3 initial cases (w3)")
    ax2.set_ylabel("week 2 initial cases (w2)")
    ax2.set_title(f"Time to reach |relative error| <= {rel_error_threshold} (in matched samples r)\nNaN => threshold not reached")

    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Samples (r) to threshold (98th pct cap shown)")

    png_time_out = csv_out.with_suffix(".time_to_threshold_heatmap.png")
    fig2.tight_layout()
    fig2.savefig(png_time_out, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Also save JSON summary (min/max stats)
    summary = {
        "rel_error_vlim": float(vlim),
        "time_vmax_98pct": int(vmax_time),
        "max_abs_rel_error": float(np.nanmax(np.abs(pivot_rel.values))) if pivot_rel.size else None,
        "mean_abs_rel_error": float(np.nanmean(np.abs(pivot_rel.values))) if pivot_rel.size else None,
        "n_cells": int(pivot_rel.size) if pivot_rel.size else 0,
        "grid_shape": (len(pivot_rel.index), len(pivot_rel.columns)),
        "csv": str(csv_out),
        "rel_error_heatmap": str(png_rel_out),
        "time_to_threshold_heatmap": str(png_time_out),
        "rel_error_threshold": float(rel_error_threshold),
    }
    (out_dir / "pmo_error_grid_summary.json").write_text(json.dumps(summary, indent=2))

    return csv_out


# CLI ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid comparison: analytic vs empirical PMO for [1,w2,w3].")
    p.add_argument("--sim-csv", type=str, default=DEFAULT_SIM_CSV, help="Path to simulated CSV (matching data)")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to save CSV/PNG outputs")
    p.add_argument("--max-digit", type=int, default=DEFAULT_MAX_DIGIT, help="Max value for w2,w3 grid (inclusive)")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size used to compute empirical PMO (<=0 => use all matches)")
    p.add_argument("--nR", type=int, default=DEFAULT_NR, help="Number of R grid points for analytic integration")
    p.add_argument("--header-rows", type=int, default=3, help="Number of metadata header rows in simulation CSV")
    p.add_argument("--week-prefix", type=str, default="week_", help="Prefix for week columns in CSV")
    p.add_argument("--random-seed", type=int, default=1234, help="Seed for sampling matched trajectories")
    p.add_argument("--rel-error-threshold", type=float, default=DEFAULT_REL_ERROR_THRESHOLD, help="Relative error threshold (fraction) to measure time-to-threshold")
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Running pmo error grid evaluation with parameters:")
    print(vars(args))

    csv_out = build_grid_and_evaluate(
        sim_csv=args.sim_csv,
        out_dir=args.out_dir,
        max_digit=args.max_digit,
        sample_size=(None if args.sample_size <= 0 else args.sample_size),
        header_rows=args.header_rows,
        week_prefix=args.week_prefix,
        random_seed=args.random_seed,
        nR=args.nR,
        rel_error_threshold=args.rel_error_threshold,
    )

    print(f"Finished. Results written to: {csv_out}")
    print(f"Relative error heatmap saved to: {Path(csv_out).with_suffix('.rel_error_heatmap.png')}")
    print(f"Time-to-threshold heatmap saved to: {Path(csv_out).with_suffix('.time_to_threshold_heatmap.png')}")
    print(f"Summary JSON: {Path(args.out_dir) / 'pmo_error_grid_summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# src/outbreak_probabilities/trajectory_matching/pmo_error_grid.py
"""
Compute and plot the error (empirical PMO - analytic PMO) for 3-week initial
conditions where week1 is fixed to 1 and week2/week3 vary from 0..max_digit.

- For each initial condition [1, w2, w3] the script:
    * computes analytic PMO via compute_pmo_from_string(...)
    * finds matched simulated trajectories via trajectory_match_pmo(...)
    * computes the empirical PMO (final empirical fraction among matched sims,
      using a sampled subset if requested)
    * records the error = empirical - analytic

- Outputs:
    * CSV with rows: w2, w3, analytic_pmo, empirical_pmo, error, n_matches
    * PNG heatmap of error over the (w2,w3) grid, saved next to the CSV.

Usage (example):
    PYTHONPATH=src python -m outbreak_probabilities.trajectory_matching.pmo_error_grid \
        --sim-csv data/test_simulations.csv \
        --out-dir figs/pmo_error_grid \
        --max-digit 9 \
        --sample-size 500 \
        --nR 2001
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
) -> Tuple[float, int]:
    """
    Use trajectory_match_pmo via load_matches and compute the final empirical PMO.

    Returns (empirical_pmo, n_matches)
      * empirical_pmo is the final cumulative PMO fraction (or nan if no matches)
      * n_matches is number of matching simulated trajectories
    """
    # get matches (trajectory_match_pmo wrapper)
    try:
        res = load_matches(sim_csv=sim_csv, observed=observed, header_rows=header_rows, week_prefix=week_prefix)
    except Exception:
        return float("nan"), 0

    n_matches = int(res.get("n_matches", 0))
    if n_matches == 0:
        return float("nan"), 0

    matches_df = res.get("matches_df")
    if matches_df is None or matches_df.shape[0] == 0:
        return float("nan"), 0

    # sample (keeps order; prepare_sample from pmo_vs_r_refractor)
    week_cols = get_week_columns(matches_df, week_prefix)
    sampled_df = prepare_sample(matches_df=matches_df, week_cols=week_cols,
                                sample_strategy=sample_strategy, sample_size=sample_size,
                                random_seed=random_seed)

    # compute empirical running PMO; take the final value
    pmo_r = compute_pmo_r_from_ordered(sampled_df)
    if pmo_r.size == 0:
        return float("nan"), n_matches
    empirical_final = float(pmo_r[-1])
    return empirical_final, n_matches


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
) -> Path:
    """
    Evaluate analytic vs empirical PMO on grid (w2=0..max_digit, w3=0..max_digit),
    where observed = [1, w2, w3].

    Writes CSV results and a heatmap PNG to out_dir; returns path to CSV.
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

        # empirical
        empirical_pmo, n_matches = compute_empirical_pmo_for_obs(
            sim_csv=sim_csv,
            observed=observed,
            header_rows=header_rows,
            week_prefix=week_prefix,
            sample_strategy="random",
            sample_size=sample_size,
            random_seed=random_seed,
        )

        # error (empirical - analytic)
        if math.isfinite(empirical_pmo) and math.isfinite(analytic_pmo):
            error = float(empirical_pmo - analytic_pmo)
        elif math.isfinite(empirical_pmo) and not math.isfinite(analytic_pmo):
            error = float("nan")
        else:
            error = float("nan")

        rows.append({
            "w2": int(w2),
            "w3": int(w3),
            "observed_str": f"{observed[0]},{observed[1]},{observed[2]}",
            "analytic_pmo": float(analytic_pmo) if math.isfinite(analytic_pmo) else float("nan"),
            "empirical_pmo": float(empirical_pmo) if math.isfinite(empirical_pmo) else float("nan"),
            "error": error,
            "n_matches": int(n_matches),
        })

        # progress bar update text (if tqdm available)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"last_obs": f"{observed}", "n_matches": n_matches})

    # Build DataFrame and write CSV
    df = pd.DataFrame(rows)
    csv_out = out_dir / f"pmo_error_grid_w2_0-{max_digit}_w3_0-{max_digit}.csv"
    df.to_csv(csv_out, index=False)

    # Create heatmap of error (w2 rows, w3 columns)
    # Mask invalid cells where n_matches==0 or error is nan
    pivot = df.pivot(index="w2", columns="w3", values="error")
    counts = df.pivot(index="w2", columns="w3", values="n_matches")

    # set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # use diverging colormap centered at 0
    vlim = np.nanmax(np.abs(pivot.values)) if pivot.size else 1.0
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)

    # annotate cells with n_matches if small grid; otherwise only show ticks
    annotate = (max_digit <= 12)
    if annotate:
        for i, w2 in enumerate(sorted(pivot.index)):
            for j, w3 in enumerate(sorted(pivot.columns)):
                val = pivot.loc[w2, w3]
                cnt = counts.loc[w2, w3]
                if not (pd.isna(val) or not np.isfinite(val)):
                    ax.text(j, i, f"{val:.3f}\n({int(cnt)})", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, f"n={int(cnt)}", ha="center", va="center", fontsize=7, color="gray")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("week 3 initial cases (w3)")
    ax.set_ylabel("week 2 initial cases (w2)")
    ax.set_title("Empirical PMO - Analytic PMO (error)\nObserved start: [1, w2, w3]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Empirical - Analytic PMO")

    png_out = csv_out.with_suffix(".error_heatmap.png")
    fig.tight_layout()
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also save JSON summary (min/max stats)
    summary = {
        "max_abs_error": float(np.nanmax(np.abs(pivot.values))) if pivot.size else None,
        "mean_abs_error": float(np.nanmean(np.abs(pivot.values))) if pivot.size else None,
        "n_cells": int(pivot.size) if pivot.size else 0,
        "grid_shape": (len(pivot.index), len(pivot.columns)),
        "csv": str(csv_out),
        "heatmap": str(png_out),
    }
    (out_dir / "pmo_error_grid_summary.json").write_text(json.dumps(summary, indent=2))

    return csv_out


# CLI ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid comparison: analytic vs empirical PMO for [1,w2,w3].")
    p.add_argument("--sim-csv", type=str, default=DEFAULT_SIM_CSV, help="Path to simulated CSV (matching data)")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to save CSV/PNG outputs")
    p.add_argument("--max-digit", type=int, default=DEFAULT_MAX_DIGIT, help="Max value for w2,w3 grid (inclusive)")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size used to compute empirical PMO (None => use all matches)")
    p.add_argument("--nR", type=int, default=DEFAULT_NR, help="Number of R grid points for analytic integration")
    p.add_argument("--header-rows", type=int, default=3, help="Number of metadata header rows in simulation CSV")
    p.add_argument("--week-prefix", type=str, default="week_", help="Prefix for week columns in CSV")
    p.add_argument("--random-seed", type=int, default=1234, help="Seed for sampling matched trajectories")
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
    )

    print(f"Finished. Results written to: {csv_out}")
    print(f"Heatmap saved to: {Path(csv_out).with_suffix('.error_heatmap.png')}")
    print(f"Summary JSON: {Path(args.out_dir) / 'pmo_error_grid_summary.json'}")


if __name__ == "__main__":
    main()

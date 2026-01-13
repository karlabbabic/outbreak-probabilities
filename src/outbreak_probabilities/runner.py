#!/usr/bin/env python3
# src/outbreak_probabilities/runner.py — concise runner (with commented CLI options)

import argparse
import re
import time
from typing import List, Optional, Tuple

# Make sure imports are correct for running from source
from .simulate import simulate_paths_refractor as sim
from .simulate import plot_traj_refractor as plot
from .trajectory_matching import plot_matches_refractor as tm
from .trajectory_matching import plot_pmo_vs_r_refractor as pmo_r 

# Parser for initial cases like 1,2,3
def parse_int_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
    return [int(x) for x in re.split(r"[,\s;]+", s.strip()) if x]

# Parser for figsize like 800,400
def parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    w, h = [float(x) for x in s.split(",")]
    return (w, h)

def main():
    p = argparse.ArgumentParser(description="Runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------- simulate ----------
    sim_p = sub.add_parser("simulate", help="Generate simulated outbreak trajectories")
    sim_p.add_argument("-N", "--num", dest="N", type=int, default=1000,
                    metavar="N",
                    help="Number of outbreaks to create (default: 1000)")
    sim_p.add_argument("--seed", type=int, default=42,
                    metavar="SEED",
                    help="RNG seed for reproducibility (default: 42)")
    sim_p.add_argument("--out", default="data/test_simulations.csv",
                    metavar="PATH",
                    help="Output CSV path (default: data/test_simulations.csv)")
    sim_p.add_argument("--initial-cases", type=str, default="1",
                    metavar="LIST",
                    help="Initial cases starting week 1 (comma/space separated, default: '1')")
    sim_p.add_argument("--max-weeks", type=int, default=50,
                    metavar="WEEKS",
                    help="Simulation length in weeks (default: 50)")
    sim_p.add_argument("--write-weeks", type=int, default=5,
                    metavar="WEEKS",
                    help="Number of weeks to write to CSV (default: 5)")
    sim_p.add_argument("--major-threshold", type=int, default=100,
                    metavar="THRESH",
                    help="Cumulative cases considered a major outbreak (default: 100)")
    sim_p.add_argument("--r-min", type=float, default=0.0, metavar="R_MIN",
                    help="Minimum R value (default: 0.0)")
    sim_p.add_argument("--r-max", type=float, default=10.0, metavar="R_MAX",
                    help="Maximum R value (default: 10.0)")
    sim_p.add_argument("--generate-full", action="store_true",
                    help="Store full weekly cases to CSV (may conflict with --write-weeks)")
    # sim_p.add_argument("--r-dist", default="uniform", metavar="DIST", help="R distribution (default: uniform)")
    # sim_p.add_argument("--use-tempfile", action="store_true", help="Write to tempfile instead of out path")

    # ---------- plot ----------
    plot_p = sub.add_parser("plot")

    plot_p.add_argument("--out", default="data/test_simulations.csv",
                    metavar="PATH",
                    help="Input CSV path (default: data/test_simulations.csv)")
    plot_p.add_argument("--sample-size", type=int, default=200,
                    metavar="SIZE",
                    help="Number of outbreaks to plot (default: 200)")
    plot_p.add_argument("--sample-strategy", type=str, default="hybrid",
                    metavar="STRATEGY",
                    help="How to pick outbreaks: random, highest_cumulative, highest_R, hybrid (extremes + random)")
    plot_p.add_argument("--overlay-mean", action="store_true")
    plot_p.add_argument("--overlay-quantiles", action="store_true")

    plot_p.add_argument("--bins", type=int, default=20)
    plot_p.add_argument("--major-threshold", type=int, default=100)
    plot_p.add_argument("--random-seed", type=int, default=42)

    # plot_p.add_argument("--header-rows", type=int, default=3)
    # plot_p.add_argument("--out-pmo", default="figs/pmo_vs_r.png")
    # plot_p.add_argument("--out-traj", default="figs/weekly_trajectories.png")

    # ---------- match ----------
    match_p = sub.add_parser("match")
    match_p.add_argument("--sim-csv", default="data/test_simulations.csv")
    match_p.add_argument("--out", default="figs/matched_trajectories.png")
    match_p.add_argument("--initial-cases", type=str, default="1,2,3")
    match_p.add_argument("--sample-size", type=int, default=200)
    match_p.add_argument("--max-plot", type=int, default=200)
    match_p.add_argument("--figsize", type=str, default=None)
    # match_p.add_argument("--header-rows", type=int, default=3)
    # match_p.add_argument("--week-prefix", default="week_")
    match_p.add_argument("--major-threshold", type=int, default=100)
    match_p.add_argument("--sample-strategy", default="highest_peak")
    match_p.add_argument("--random-seed", type=int, default=42)

    # ---------- pmo_vs_r (new) ----------
    pmo_p = sub.add_parser("pmo_vs_r", help="Plot PMO fraction as a function of sampled matched trajectories (r=1..R)")
    pmo_p.add_argument("--sim-csv", default="data/test_simulations.csv", help="Simulations CSV used for matching")
    pmo_p.add_argument("--out", default="figs/pmo_vs_r.png", help="Output PNG path")
    pmo_p.add_argument("--initial-cases", type=str, default="1,2,3", help="Observed initial cases (comma or space separated)")
    pmo_p.add_argument("--sample-size", type=int, default=200)
    pmo_p.add_argument("--sample-strategy", default="random", help="Sampling strategy (random, highest_peak, highest_cumulative, highest_R, hybrid)")
    pmo_p.add_argument("--sort-by", default="sample_order", help="Sorting of sampled matches before computing PMO(r): sample_order, by_cumulative, by_peak, by_R, by_PMO")
    pmo_p.add_argument("--figsize", type=str, default=None, help="Figure size 'width,height' (optional)")
    pmo_p.add_argument("--header-rows", type=int, default=3, help="CSV header rows for simulate output")
    pmo_p.add_argument("--week-prefix", type=str, default="week_", help="Week column prefix in CSV")
    pmo_p.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    pmo_p.add_argument("--max-plot", type=int, default=None)  

    args = p.parse_args()
    t0 = time.perf_counter()

    if args.cmd == "simulate":
        init = parse_int_list(args.initial_cases)
        cfg = sim.SimConfig(
            N=args.N,
            seed=args.seed,
            out_path=args.out,
            initial_cases=init,
            max_weeks=args.max_weeks,
            major_threshold=args.major_threshold,
            R_range=(args.R_min, args.R_max),
            generate_full=args.generate_full,
            write_weeks=args.write_weeks
            # R_dist=args.R_dist,
        )
        sim.simulate_batch(cfg)
        print("Simulation done ->", args.out)

    elif args.cmd == "plot":
        overlay_q = (0.1, 0.9) if args.overlay_quantiles else None
        plot.run_plotting(
            csv=args.csv,
            sample_strategy=args.sample_strategy,
            sample_size=args.sample_size,
            overlay_mean=args.overlay_mean,
            overlay_quantiles=overlay_q,
            # header_rows=args.header_rows,
            # out_pmo=args.out_pmo,
            # out_traj=args.out_traj,
            bins=args.bins,
            major_threshold=args.major_threshold,
            random_seed=args.random_seed,
        )
        print("Plots written")

    elif args.cmd == "match":
        init = parse_int_list(args.initial_cases)
        figsize = parse_figsize(args.figsize)
        tm.run_plot_matches(
            sim_csv=args.sim_csv,
            observed=init,
            sample_size=args.sample_size,
            max_plot=args.max_plot,
            out_png=args.out,
            figsize=figsize,
            # header_rows=args.header_rows,
            # week_prefix=args.week_prefix,
            major_threshold=args.major_threshold,
            sample_strategy=args.sample_strategy,
        )
        print("Matched plot ->", args.out)

    elif args.cmd == "pmo_vs_r":
        init = parse_int_list(args.initial_cases)
        figsize = parse_figsize(args.figsize)
        # call the new function
        pmo_r.run_pmo_vs_r_refractor(
            sim_csv=args.sim_csv,
            observed=init,
            header_rows=args.header_rows,
            week_prefix=args.week_prefix,
            out_png=args.out,
            sample_strategy=args.sample_strategy,
            sample_size=args.sample_size,
            sort_by=args.sort_by,
            figsize=figsize if figsize is not None else None,
            random_seed=args.random_seed,
        )
        print("PMO vs r plot ->", args.out)

    print(f"Done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

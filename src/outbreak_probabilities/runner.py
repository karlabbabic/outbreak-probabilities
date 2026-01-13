#!/usr/bin/env python3
# src/outbreak_probabilities/runner.py
import argparse
import time

# adjust names if your modules are spelled differently
from .simulate import simulate_paths_refractor as sim
from .simulate import plot_traj_refractor as plot

"""
1. Simulate
PYTHONPATH=src python -m outbreak_probabilities.runner simulate --N 100000 --seed 42 --out data/test_simulations_2.csv

2. Plot
PYTHONPATH=src python -m outbreak_probabilities.runner plot --csv data/test_simulations_2.csv --sample-strategy random

3. Match trajectories
PYTHONPATH=src python -m outbreak_probabilities.runner match --initial-cases 1,2,3,4
"""
#!/usr/bin/env python3
# src/outbreak_probabilities/runner.py — concise runner (with commented CLI options)

import argparse, re, time
from typing import List, Optional, Tuple

from .simulate import simulate_paths_refractor as sim
from .simulate import plot_traj_refractor as plot
from .trajectory_matching import plot_matches_refractor as tm

def parse_int_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
    return [int(x) for x in re.split(r"[,\s;]+", s.strip()) if x]


def parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    w, h = [float(x) for x in s.split(",")]
    return (w, h)


def main():
    p = argparse.ArgumentParser(description="Tiny runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------- simulate ----------
    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("--N", type=int, default=1000)
    sim_p.add_argument("--seed", type=int, default=42)
    sim_p.add_argument("--out", default="data/test_simulations.csv")
    sim_p.add_argument("--initial-cases", type=str, default="1")
    # sim_p.add_argument("--max-weeks", type=int, default=15)
    # sim_p.add_argument("--major-threshold", type=int, default=100)
    # sim_p.add_argument("--R-min", type=float, default=0.0)
    # sim_p.add_argument("--R-max", type=float, default=10.0)
    # sim_p.add_argument("--R-dist", default="uniform")
    # sim_p.add_argument("--use-tempfile", action="store_true")
    # sim_p.add_argument("--workers", type=int)
    # sim_p.add_argument("--chunk-size", type=int)

    # ---------- plot ----------
    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--csv", default="data/test_simulations.csv")
    plot_p.add_argument("--sample-size", type=int, default=200)
    plot_p.add_argument("--sample-strategy", default="hybrid")
    plot_p.add_argument("--overlay-mean", action="store_true")
    plot_p.add_argument("--overlay-quantiles", action="store_true")
    # plot_p.add_argument("--header-rows", type=int, default=3)
    # plot_p.add_argument("--out-pmo", default="figs/pmo_vs_r.png")
    # plot_p.add_argument("--out-traj", default="figs/weekly_trajectories.png")
    # plot_p.add_argument("--bins", type=int, default=20)
    # plot_p.add_argument("--major-threshold", type=int, default=100)
    # plot_p.add_argument("--random-seed", type=int, default=42)

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
    # match_p.add_argument("--major-threshold", type=int, default=100)
    match_p.add_argument("--sample-strategy", default="highest_peak")
    # match_p.add_argument("--random-seed", type=int, default=42)

    args = p.parse_args()
    t0 = time.perf_counter()

    if args.cmd == "simulate":
        init = parse_int_list(args.initial_cases)
        cfg = sim.SimConfig(
            N=args.N,
            seed=args.seed,
            out_path=args.out,
            initial_cases=init,
            # max_weeks=args.max_weeks,
            # major_threshold=args.major_threshold,
            # R_range=(args.R_min, args.R_max),
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
            # bins=args.bins,
            # major_threshold=args.major_threshold,
            # random_seed=args.random_seed,
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
            # major_threshold=args.major_threshold,
            # sample_strategy=args.sample_strategy,
        )
        print("Matched plot ->", args.out)

    print(f"Done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

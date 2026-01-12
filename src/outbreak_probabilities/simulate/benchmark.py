"""
Benchmark script for simulate/

Run this to:
- simulate trajectories under a wider range of parameter settings
- keep problem sizes modest but non-trivial
- print basic stats (shape, PMO, runtime)

It expects:
    - compute_serial_weights in calculate_serial_weights.py
    - generate_batch in batch_processing.py
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from calculate_serial_weights import compute_serial_weights
from batch_processing import generate_batch

def run_benchmarks():
    # Serial weights for Ebola are calculated once 
    # Could also benchmark this, although I expect it to be quite fast, and it only runs once anyways
    w = compute_serial_weights(mean=15.3, std=9.3, k_max=60, nquad=32, step=7.0)

    # Define a wider set of scenarios to benchmark.
    scenarios = [
        {
            "label": "Subcritical R=0.8, N=20000, 10 weeks",
            "N": 20000,
            "max_weeks": 10,
            "R_range": (0.8, 0.8),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,5, N=200000, 10 weeks",
            "N": 200000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,5, N=1000000, 10 weeks",
            "N": 1000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,5, N=2000000, 10 weeks",
            "N": 2000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,5, N=3000000, 10 weeks",
            "N": 3000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,5, N=5000000, 10 weeks",
            "N": 5000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,10, N=200000, 10 weeks",
            "N": 200000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,10, N=1000000, 10 weeks",
            "N": 1000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,10, N=2000000, 10 weeks",
            "N": 2000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,10, N=3000000, 10 weeks",
            "N": 3000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        },
        {
            "label": "R between 0,10, N=5000000, 10 weeks",
            "N": 5000000,
            "max_weeks": 10,
            "R_range": (0,5),
            "extinction_window": 4,
        }
    ]

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(scenarios, start=1):
        print("\n" + "=" * 80)
        print(f"Scenario {i}: {cfg['label']}")
        print(
            f"  N={cfg['N']}, max_weeks={cfg['max_weeks']}, "
            f"R_range={cfg['R_range']}, extinction_window={cfg['extinction_window']}"
        )

        # Give each scenario its own CSV so nothing gets overwritten
        csv_name = f"simulated_cases_bench_{i}.csv"
        out_path = out_dir / csv_name

        t0 = time.perf_counter()
        trajectories, csv_path = generate_batch(
            N=cfg["N"],
            w=w,
            max_weeks=cfg["max_weeks"],
            R_range=cfg["R_range"],
            initial_cases=[1],
            extinction_window=cfg["extinction_window"],
            major_threshold=100,
            seed=42,
            out_path=str(out_path),
            use_tempfile=False,
        )
        t1 = time.perf_counter()
        runtime = t1 - t0

        print(f"  Trajectories shape: {trajectories.shape}")
        print(f"  CSV written to:     {csv_path}")
        print(f"  Runtime:            {runtime:.3f} seconds")

        # Compute PMO (fraction of trajectories classified as major) + some R summary
        pmo_estimate = np.nan
        mean_R = np.nan
        median_R = np.nan

        try:
            # The CSV now includes two metadata rows at the top:
            # row 0 = weights (w), row 1 = R_range, row 2 = header
            # Use header=2 so pandas uses the third row as the column names.
            df = pd.read_csv(csv_path, header=2)

            # If that doesn't produce the expected columns (e.g. older CSV without metadata),
            # fall back to the default behavior and try header=0.
            if "PMO" not in df.columns or "R_draw" not in df.columns:
                df_alt = pd.read_csv(csv_path, header=0)
                # prefer the header=2 read if it has more of the expected columns, otherwise use the fallback
                if ("PMO" in df_alt.columns and "R_draw" in df_alt.columns) and ("PMO" not in df.columns or "R_draw" not in df.columns):
                    df = df_alt

            if "PMO" in df.columns:
                pmo_estimate = df["PMO"].mean()
                n_major = int(df["PMO"].sum())
                print(f"  PMO estimate:       {pmo_estimate:.3f} ({n_major} major out of {cfg['N']})")
            else:
                print("  Warning: CSV has no 'PMO' column; cannot compute PMO.")

            if "R_draw" in df.columns:
                mean_R = df["R_draw"].mean()
                median_R = df["R_draw"].median()
                print(f"  R_draw mean/median: {mean_R:.3f} / {median_R:.3f}")
        except Exception as e:
            print(f"  Warning: could not read CSV to compute PMO/R summary ({e})")

        results.append(
            {
                "scenario": cfg["label"],
                "N": cfg["N"],
                "max_weeks": cfg["max_weeks"],
                "R_min": cfg["R_range"][0],
                "R_max": cfg["R_range"][1],
                "runtime_sec": runtime,
                "PMO": pmo_estimate,
                "mean_R": mean_R,
                "median_R": median_R,
                "csv_path": str(csv_path),
            }
        )

    # Optional: print a compact summary table at the end
    print("\n" + "=" * 80)
    print("Benchmark summary:")
    for r in results:
        print(
            f"- {r['scenario']}\n"
            f"    N={r['N']}, max_weeks={r['max_weeks']}, R in [{r['R_min']}, {r['R_max']}]\n"
            f"    runtime={r['runtime_sec']:.3f}s, PMO={r['PMO']:.3f}, "
            f"mean_R={r['mean_R']:.2f}, median_R={r['median_R']:.2f}"
        )


if __name__ == "__main__":
    run_benchmarks()

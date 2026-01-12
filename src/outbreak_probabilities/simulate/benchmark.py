"""
Benchmark script for simulate/

- Each scenario uses a deterministic per-scenario master seed (1000 + i).
- generate_batch writes three metadata rows: w, R_range, master_seed, then the header row.
- We therefore read CSVs with header=3 to get the actual data columns (sim_id, sim_seed, R_draw, week_..., cumulative_cases, status, PMO).
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from calculate_serial_weights import compute_serial_weights
from batch_processing import generate_batch

def run_benchmarks():
    # Serial weights for Ebola are calculated once
    w = compute_serial_weights(mean=15.3, std=9.3, k_max=60, nquad=32, step=7.0)

    # Scenarios: adjust R_range and optionally R_dist / R_dist_params per scenario
    scenarios = [
        {"label": "Subcritical R=0.8, N=200, 10 weeks", "N": 200, "max_weeks": 10, "R_range": (0.8, 0.8), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 5.0, N=200, 10 weeks",  "N": 200,  "max_weeks": 10, "R_range": (0.0, 5.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 5.0, N=1000, 10 weeks", "N": 1000, "max_weeks": 10, "R_range": (0.0, 5.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 5.0, N=2000, 10 weeks", "N": 2000, "max_weeks": 10, "R_range": (0.0, 5.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 5.0, N=3000, 10 weeks", "N": 3000, "max_weeks": 10, "R_range": (0.0, 5.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 5.0, N=5000, 10 weeks", "N": 5000, "max_weeks": 10, "R_range": (0.0, 5.0), "extinction_window": 4, "R_dist": "uniform"},
        # 0..10 ranges (examples)
        {"label": "R between 0.0 and 10.0, N=2000, 10 weeks", "N": 2000, "max_weeks": 10, "R_range": (0.0, 10.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 10.0, N=1000, 10 weeks", "N": 1000, "max_weeks": 10, "R_range": (0.0, 10.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 10.0, N=2000, 10 weeks", "N": 2000, "max_weeks": 10, "R_range": (0.0, 10.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 10.0, N=3000, 10 weeks", "N": 3000, "max_weeks": 10, "R_range": (0.0, 10.0), "extinction_window": 4, "R_dist": "uniform"},
        {"label": "R between 0.0 and 10.0, N=5000, 10 weeks", "N": 5000, "max_weeks": 10, "R_range": (0.0, 10.0), "extinction_window": 4, "R_dist": "uniform"},
    ]

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(scenarios, start=1):
        print("\n" + "=" * 80)
        print(f"Scenario {i}: {cfg['label']}")
        print(f"  N={cfg['N']}, max_weeks={cfg['max_weeks']}, R_range={cfg['R_range']}, extinction_window={cfg['extinction_window']}")

        csv_name = f"simulated_cases_bench_{i}.csv"
        out_path = out_dir / csv_name

        # deterministic per-scenario master seed
        seed = 1000 + i
        print(f"  Using master seed: {seed}")

        t0 = time.perf_counter()
        trajectories, csv_path = generate_batch(
            N=cfg["N"],
            w=w,
            max_weeks=cfg["max_weeks"],
            R_range=cfg["R_range"],
            initial_cases=[1],
            extinction_window=cfg["extinction_window"],
            major_threshold=100,
            seed=seed,
            out_path=str(out_path),
            use_tempfile=False,
            R_dist=cfg.get("R_dist", "uniform"),
            R_dist_params=cfg.get("R_dist_params", None),
        )
        runtime = time.perf_counter() - t0

        print(f"  Trajectories shape: {trajectories.shape}")
        print(f"  CSV written to:     {csv_path}")
        print(f"  Runtime:            {runtime:.3f} seconds")

        pmo_estimate = np.nan
        mean_R = np.nan
        median_R = np.nan

        try:
            # read CSV: generate_batch writes three metadata rows (w, R_range, master_seed) then the header
            df = pd.read_csv(csv_path, header=3)

            # fallback if different header format
            if "PMO" not in df.columns or "R_draw" not in df.columns:
                df_alt = pd.read_csv(csv_path, header=0)
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
                # quick PMO-by-R bin check (optional)
                try:
                    bins = pd.cut(df["R_draw"], bins=10)
                    print("  PMO by R bins (mean):")
                    print(df.groupby(bins)["PMO"].mean().round(3))
                except Exception:
                    pass
        except Exception as e:
            print(f"  Warning: could not read CSV to compute PMO/R summary ({e})")

        results.append({
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
        })

if __name__ == "__main__":
    run_benchmarks()

#!/usr/bin/env python3
"""
compare_combined.py

Combine ML convergence (GB & RF) with deterministic trajectory-mapping PMO
that is recomputed from the full sim CSV for each initial_sample.

Behavior:
 - ALWAYS attempts to read the full simulation CSV provided by `sim_csv`.
   (tries header=3 then header=0 to be robust to the metadata lines).
 - Performs exact matching on the first k week_* columns where k = len(initial_sample).
 - Stores matched sim indices in-memory (matched_index_cache) for reuse during the process.
 - Produces a plot mapping the trajectory-matching PMO (computed from matched rows only)
   onto the ML training-size x axis as a step + per-size values.
 - Writes only the PNG and metadata JSON. DOES NOT WRITE matched CSVs.

Usage:
    PYTHONPATH=src python -m outbreak_probabilities.compare.compare_combined

Author: adapted for your repo and requirements.
"""
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
import traceback

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- User-editable defaults ----
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"
OUT_DIR = BASE_DIR / "src" / "outbreak_probabilities" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# training sizes (x axis)
DATA_SIZES = [500 * i for i in range(1, 70)]

MODEL_NAMES = ["GB", "RF"]
COL_GB = "xkcd:asparagus"
COL_RF = "royalblue"
COL_PMO = "xkcd:darkblue"
COL_ANALYTIC = "red"

# CI / bootstrap options
CI_LEVEL = 0.90
CI_BOOT = 500

# ------------------ internal cache ------------------
# Store matched sim_id lists keyed by initial_sample tuple so repeated calls in the same process reuse them.
matched_index_cache: Dict[Tuple[int, ...], np.ndarray] = {}

# ---------------- helpers ----------------
def compute_running_ci(pmo_flags: np.ndarray, n_boot: int = 500, ci: float = 0.90, random_seed: Optional[int] = None):
    if pmo_flags is None or pmo_flags.size == 0:
        return np.array([]), np.array([])
    R = pmo_flags.size
    rng = np.random.default_rng(random_seed)
    n_boot = max(1, int(n_boot))
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

def get_week_columns(df: pd.DataFrame, week_prefix: str = "week_") -> List[str]:
    week_cols = [c for c in df.columns if c.startswith(week_prefix)]
    # ensure they sort numerically by suffix
    return sorted(week_cols, key=lambda s: int(s.split(week_prefix)[1]) if s.split(week_prefix)[1].isdigit() else s)

def load_full_sim_csv(path: str) -> pd.DataFrame:
    """
    Try to robustly load the full sim CSV. Many of your CSVs have 3 metadata rows
    before the CSV header, so try header=3 first, then header=0.
    Return a DataFrame with cleaned column names.
    """
    last_exc = None
    for header in (3, 0):
        try:
            df = pd.read_csv(path, header=header)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_exc = e
    # If both attempts fail, re-raise
    raise last_exc

def find_matches_in_full_sim_df(full_df: pd.DataFrame, initial_sample: Tuple[int, ...], week_prefix: str = "week_") -> pd.DataFrame:
    """
    Exact-match the first k weeks of full_df to the initial_sample tuple.
    Returns a DataFrame of matched rows (preserving original sim order).
    Assumes full_df contains week_* columns.
    """
    week_cols = get_week_columns(full_df, week_prefix)
    k = len(initial_sample)
    if k == 0:
        raise ValueError("initial_sample empty")
    if k > len(week_cols):
        raise ValueError(f"initial_sample length {k} > available week columns {len(week_cols)}")
    mask = np.ones(len(full_df), dtype=bool)
    # compare each week column robustly
    for j in range(k):
        col = week_cols[j]
        target = initial_sample[j]
        # try numeric compare first
        try:
            col_num = pd.to_numeric(full_df[col], errors="coerce").to_numpy()
            # Use equality for integers but allow small tolerance; also treat NaN as mismatch
            mask &= np.isfinite(col_num) & (np.isclose(col_num, float(target), atol=1e-9))
        except Exception:
            # fallback string compare
            mask &= (full_df[col].astype(str).str.strip() == str(target))
    matched = full_df.loc[mask].copy().reset_index(drop=True)
    return matched

def derive_pmo_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a PMO column exists (0/1). If present, coerce to int; otherwise try common fallbacks:
    - 'status' == 'major' or 'major' substring
    - 'final_size' or 'cumulative_cases' > 0 => major
    """
    df = df.copy()
    if "PMO" in df.columns:
        try:
            df["PMO"] = df["PMO"].astype(int)
            return df
        except Exception:
            # try mapping textual '1'/'0' or 'True'/'False'
            df["PMO"] = df["PMO"].map(lambda v: 1 if str(v).strip() in ("1", "True", "true", "major") else 0).astype(int)
            return df
    # look for status-like column
    if "status" in df.columns:
        df["PMO"] = df["status"].astype(str).str.contains("major", case=False, na=False).astype(int)
        return df
    if "final_size" in df.columns:
        try:
            df["PMO"] = (pd.to_numeric(df["final_size"], errors="coerce").fillna(0) > 0).astype(int)
            return df
        except Exception:
            pass
    if "cumulative_cases" in df.columns:
        try:
            df["PMO"] = (pd.to_numeric(df["cumulative_cases"], errors="coerce").fillna(0) > 0).astype(int)
            return df
        except Exception:
            pass
    # give up
    raise RuntimeError("Could not derive PMO column from full sim CSV; provide a column named 'PMO' or 'status'/'final_size'/'cumulative_cases'.")

# ---------------- ML loader (unchanged, from your file) ----------------
def load_ml_curves_for_sample(sample_tuple: Tuple[int, ...], model_dir: Path, data_sizes: List[int], model_names: List[str]) -> Dict[str, List[Optional[float]]]:
    results = {m: [] for m in model_names}
    arr = np.array(sample_tuple).reshape(1, -1)
    for size in data_sizes:
        for m in model_names:
            stem = f"ML_SIM_{size}_{m}"
            model_path = model_dir / f"{stem}.pkl"
            scaler_path = model_dir / f"{stem}_scaler.pkl"
            try:
                mdl = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                scaled = scaler.transform(arr)
                if hasattr(mdl, "predict_proba"):
                    pred = float(mdl.predict_proba(scaled)[0][1])
                else:
                    pred = float(mdl.predict(scaled)[0])
                results[m].append(pred)
            except Exception:
                results[m].append(None)
    return results

def make_combined_plot(
    initial_sample: Tuple[int, ...] = (1, 2, 0),
    sim_csv: Optional[str] = None,   # path to full sim CSV (required)
    model_dir: Path = MODEL_DIR,
    data_sizes: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    out_png: Optional[str] = None,
    sample_size: Optional[int] = None,  # deterministic truncation per S (earliest sim ids)
):
    """
    Build combined ML vs trajectory-mapping plot for the given initial_sample.

    Requirements / behaviour:
      - sim_csv must point to the full simulation CSV (tries header=3 then header=0).
      - exact-matches the first k week_* columns where k = len(initial_sample).
      - computes analytic PMO for the given initial_sample (infers R_min/R_max from sim_csv if present).
      - computes trajectory-mapping PMO using only matched rows (matched sim_ids <= S for each S).
      - returns (out_png, meta)
    """
    if data_sizes is None:
        data_sizes = DATA_SIZES
    if model_names is None:
        model_names = MODEL_NAMES
    if out_png is None:
        out_png = str(OUT_DIR / f"comparison_initial_{'_'.join(map(str, initial_sample))}.png")

    if sim_csv is None:
        raise RuntimeError("sim_csv must be provided (full simulation CSV path).")

    data_sizes = np.array(data_sizes, dtype=int)
    x_min = float(np.min(data_sizes))
    x_max = float(np.max(data_sizes))

    # 1) ML predictions
    ml_results = load_ml_curves_for_sample(initial_sample, model_dir, data_sizes.tolist(), model_names)

    # 2) Build matches for the current initial_sample by reading the full simulation CSV
    key = tuple(int(x) for x in initial_sample)
    matched_df = None
    matched_sim_ids = None
    full_df = None

    # If cached, reuse
    if key in matched_index_cache and f"df_{key}" in matched_index_cache:
        matched_sim_ids = matched_index_cache[key]
        matched_df = matched_index_cache[f"df_{key}"].copy()
        # attempt to keep full_df if cached (not required)
        full_df = matched_index_cache.get(f"full_{key}", None)
    else:
        # load full sim csv (try robustly)
        full_df = load_full_sim_csv(sim_csv)  # may raise
        week_cols = get_week_columns(full_df, "week_")
        if not week_cols:
            raise RuntimeError("Provided sim_csv does not contain week_ columns; can't re-match initial_sample.")
        # find exact matches on first k weeks
        matched_df = find_matches_in_full_sim_df(full_df, initial_sample, week_prefix="week_")
        if matched_df.empty:
            raise RuntimeError(f"No matches found in full sim CSV for initial_sample={initial_sample}")
        # ensure PMO column
        matched_df = derive_pmo_column(matched_df)
        # derive sim_id if missing (preserve original order)
        if "sim_id" not in matched_df.columns:
            if "match_index" in matched_df.columns:
                matched_df["sim_id"] = matched_df["match_index"].astype(int) + 1
            else:
                # recompute mask to obtain original indices
                mask = np.ones(len(full_df), dtype=bool)
                k = len(initial_sample)
                week_cols = get_week_columns(full_df, "week_")
                for j in range(k):
                    col = week_cols[j]
                    target = initial_sample[j]
                    col_num = pd.to_numeric(full_df[col], errors="coerce").to_numpy()
                    mask &= np.isfinite(col_num) & (np.isclose(col_num, float(target), atol=1e-9))
                original_indices = np.nonzero(mask)[0]
                if len(original_indices) >= len(matched_df):
                    sim_ids = (original_indices[: len(matched_df)] + 1).astype(int)
                    matched_df = matched_df.reset_index(drop=True)
                    matched_df["sim_id"] = sim_ids
                else:
                    matched_df = matched_df.reset_index().rename(columns={"index": "sim_id"})
                    matched_df["sim_id"] = matched_df["sim_id"].astype(int) + 1

        matched_df = matched_df.sort_values("sim_id").reset_index(drop=True)
        matched_sim_ids = matched_df["sim_id"].astype(int).to_numpy()
        # cache
        matched_index_cache[key] = matched_sim_ids
        matched_index_cache[f"df_{key}"] = matched_df.copy()
        matched_index_cache[f"full_{key}"] = full_df.copy()

    # Limit matched rows to the maximum S we will map to
    max_S = int(np.max(data_sizes))
    matched_df = matched_df[matched_df["sim_id"].astype(int) <= max_S].copy()
    matched_df = matched_df.sort_values("sim_id").reset_index(drop=True)

    # 3) For each training size S compute PMO using all matches with sim_id <= S
    ml_x = data_sizes.astype(float)
    ml_pmo_by_size = np.full(ml_x.shape, np.nan, dtype=float)
    ml_lower = np.full(ml_x.shape, np.nan, dtype=float)
    ml_upper = np.full(ml_x.shape, np.nan, dtype=float)

    for i, S in enumerate(data_sizes):
        sel = matched_df[matched_df["sim_id"].astype(int) <= int(S)].copy()
        if sel.shape[0] == 0:
            ml_pmo_by_size[i] = np.nan
            ml_lower[i] = np.nan
            ml_upper[i] = np.nan
            continue
        # deterministic truncation to earliest sim ids if sample_size set
        sel = sel.sort_values("sim_id").reset_index(drop=True)
        if sample_size is not None and sample_size < sel.shape[0]:
            sel = sel.iloc[:sample_size]
        flags = sel["PMO"].astype(int).to_numpy()
        ml_pmo_by_size[i] = float(flags.mean())
        try:
            lo_arr, hi_arr = compute_running_ci(pmo_flags=flags, n_boot=CI_BOOT, ci=CI_LEVEL, random_seed=42)
            if lo_arr.size and hi_arr.size:
                ml_lower[i] = float(lo_arr[-1])
                ml_upper[i] = float(hi_arr[-1])
        except Exception:
            ml_lower[i] = np.nan
            ml_upper[i] = np.nan

        # ---------------- Robust analytic PMO computation (uses the current initial_sample) ----------------
    analytic_val = float("nan")
    try:
        # Import analytic helper
        from outbreak_probabilities.analytic.analytical_refractor import compute_pmo_from_string  # type: ignore

        # ensure full_df available
        if full_df is None:
            full_df = load_full_sim_csv(sim_csv)

        # infer R_min / R_max from full_df when possible
        Rmin = None
        Rmax = None
        for col in ("R_draw", "R", "r_draw", "r"):
            if col in full_df.columns:
                numeric = pd.to_numeric(full_df[col], errors="coerce")
                if numeric.notna().any():
                    Rmin = int(numeric.min())
                    Rmax = int(numeric.max())
                    break

        # try package defaults if inference failed
        if Rmin is None or Rmax is None:
            try:
                from outbreak_probabilities.analytic.analytical_refractor import DEFAULT_R_MIN, DEFAULT_R_MAX  # type: ignore
                Rmin = int(DEFAULT_R_MIN)
                Rmax = int(DEFAULT_R_MAX)
            except Exception:
                Rmin, Rmax = 1, 10

        # build initial string expected by compute_pmo_from_string
        initial_str = ",".join(str(int(x)) for x in initial_sample) if initial_sample else ""

        # call analytic helper and robustly extract PMO
        res = compute_pmo_from_string(initial_str, nR=2001, R_min=Rmin, R_max=Rmax)

        # handle multiple possible return shapes:
        # - a float
        # - dict with 'PMO' or 'pmo' key
        # - dict with nested 'result' or similar
        analytic_val_candidate = float("nan")
        if isinstance(res, (float, int)):
            analytic_val_candidate = float(res)
        elif isinstance(res, dict):
            # try common keys
            for k in ("PMO", "pmo", "Pmo", "pm0"):
                if k in res:
                    try:
                        analytic_val_candidate = float(res[k])
                        break
                    except Exception:
                        pass
            # try nested structures
            if np.isnan(analytic_val_candidate):
                # flatten shallow nested dicts
                for v in res.values():
                    if isinstance(v, (float, int)):
                        analytic_val_candidate = float(v)
                        break
                    if isinstance(v, dict):
                        for kk in ("PMO", "pmo"):
                            if kk in v:
                                try:
                                    analytic_val_candidate = float(v[kk])
                                    break
                                except Exception:
                                    pass
                        if not np.isnan(analytic_val_candidate):
                            break
        # final assign if finite
        if np.isfinite(analytic_val_candidate):
            analytic_val = float(analytic_val_candidate)
        else:
            analytic_val = float("nan")
    except Exception as e:
        # print diagnostic so user can see why analytic wasn't computed
        print("Analytic PMO computation failed:", str(e))
        # optional: print traceback for debugging
        import traceback as _tb
        _tb.print_exc()
        analytic_val = float("nan")

    # Build a fine-grained step visualization mapped to ML x-space for plotting the "true" timeline
    if matched_df.shape[0] == 0:
        xs_step = np.array([x_min, x_max], dtype=float)
        ys_step = np.array([0.0, 0.0], dtype=float)
        mapped_event_x = np.array([], dtype=float)
        event_cum = np.array([], dtype=float)
    else:
        event_sim_ids = matched_df["sim_id"].astype(float).to_numpy()
        event_cum = np.cumsum(matched_df["PMO"].astype(int).to_numpy()).astype(float) / (np.arange(1, len(matched_df) + 1).astype(float))
        mapped_event_x = np.interp(event_sim_ids, [1.0, float(max_S)], [x_min, x_max])
        xs_list = [x_min]
        ys_list = [0.0]
        for mx, nv in zip(mapped_event_x, event_cum):
            xs_list.append(mx); ys_list.append(ys_list[-1])
            xs_list.append(mx); ys_list.append(nv)
        xs_list.append(x_max); ys_list.append(float(event_cum[-1]))
        xs_step = np.array(xs_list, dtype=float)
        ys_step = np.array(ys_list, dtype=float)

    # ---------------- PLOTTING ----------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # ML curves
    for m, col in zip(model_names, (COL_GB, COL_RF)):
        y = np.array([v if v is not None else np.nan for v in ml_results[m]])
        ax.plot(ml_x, y, label=f"{m} predicted PMO = {y[-1]:.5f}", color=col, linewidth=2.5)

    # analytic (horizontal)
    if np.isfinite(analytic_val):
        ax.axhline(analytic_val, color=COL_ANALYTIC, linestyle="--", linewidth=2.0, label=f"Analytic PMO = {analytic_val:.5f}")
        band_low = max(0.0, analytic_val - 0.05)
        band_high = min(1.0, analytic_val + 0.05)
        ax.fill_between(ml_x, band_low, band_high, color=COL_ANALYTIC, alpha=0.08)
        # ax.fill_between(ml_x, band_low, band_high, color=COL_ANALYTIC, alpha=0.08, label="Analytic ±5%")


    # trajectory-mapping step and mapped points
    ax.plot(xs_step[2:], ys_step[2:], color=COL_PMO, linewidth=1.8, drawstyle="steps-post",
            label=f"Trajectory-mapping PMO = {ys_step[-1]:.5f}", zorder=3, alpha=0.8)
    # ax.plot(ml_x, ml_pmo_by_size, color=COL_PMO, linewidth=1.1, linestyle="--", alpha=0.9, label="PMO (mapped to ML x)", zorder=2)

    # # CI shading if available (pointwise at each ML size)
    # if np.isfinite(ml_lower).any() and np.isfinite(ml_upper).any():
    #     ax.fill_between(ml_x, ml_lower, ml_upper, alpha=0.18, color=COL_PMO, label=f"{int(CI_LEVEL*100)}% CI (shuffle)")

    # scatter event points (visible)
    if mapped_event_x.size:
        ax.scatter(mapped_event_x, event_cum, s=5, color=COL_PMO, edgecolors="none", zorder=1, alpha=0.2)

    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("PMO")
    ax.set_title(f"PMO comparison of machine learning (RF, GB) and trajectory matching\n Initial cases {tuple(initial_sample)}")
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, which="major", linestyle="--")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save metadata JSON next to the PNG
    meta = {
        "initial": tuple(initial_sample),
        "data_sizes": data_sizes.tolist(),
        "model_dir": str(model_dir),
        "ml_models": model_names,
        "analytic_value_used": None if np.isnan(analytic_val) else float(analytic_val),
        "n_matches_total": int(len(matched_df)),
        "max_sim_id_used": int(max_S),
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("Saved plot:", out_png)
    print("Saved metadata:", meta_path)
    return str(out_png), meta

# ---------------- quick-run if called as script ----------------
if __name__ == "__main__":
    # Edit these two as needed: path to FULL sim CSV and initials to iterate
    SIM_CSV_PATH = str(Path(BASE_DIR) / "data" / "test_simulations.csv")  # <-- set to your full sim CSV
    INITIALS = [(1,4,0)]

    for INITIAL in INITIALS:
        OUT = str(OUT_DIR / f"comparison_initial_{'_'.join(map(str, INITIAL))}.png")
        try:
            out, meta = make_combined_plot(
                initial_sample=INITIAL,
                sim_csv=SIM_CSV_PATH,
                model_dir=MODEL_DIR,
                data_sizes=DATA_SIZES,
                model_names=MODEL_NAMES,
                out_png=OUT,
                sample_size=None,  # leave None to include all matches up to S
            )
            print("Done:", out)
        except Exception as exc:
            print("Failed for initial", INITIAL, ":", exc)
            traceback.print_exc()
            raise

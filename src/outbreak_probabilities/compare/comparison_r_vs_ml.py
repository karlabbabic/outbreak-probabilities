#!/usr/bin/env python3
"""
comparison_r_vs_ml.py

Plot GB & RF ML curves vs trajectory-matching PMO plotted as a STEP function
that updates only at matched sim IDs (all matches with sim_id <= max(data_sizes)).

Usage:
    PYTHONPATH=src python -m outbreak_probabilities.compare.comparison_r_vs_ml
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Try to import helpers from trajectory module; fallback to local copies (as before)
_USE_REMOTE = False
_import_trace = None
try:
    from outbreak_probabilities.trajectory_matching.plot_pmo_vs_r_refractor import (
        compute_running_ci,
        prepare_sample,
        get_week_columns,
        load_matches,
        DEFAULT_R_MIN,
        DEFAULT_R_MAX,
    )
    _USE_REMOTE = True
except Exception:
    try:
        from ..trajectory_matching.plot_pmo_vs_r_refractor import (  # type: ignore
            compute_running_ci,
            prepare_sample,
            get_week_columns,
            load_matches,
            DEFAULT_R_MIN,
            DEFAULT_R_MAX,
        )
        _USE_REMOTE = True
    except Exception as e:
        _import_trace = traceback.format_exc()
        _USE_REMOTE = False

if not _USE_REMOTE:
    # Local fallback implementations (copied to match your refactor)
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

    def get_week_columns(df: pd.DataFrame, week_prefix: str) -> List[str]:
        week_cols = [c for c in df.columns if c.startswith(week_prefix)]
        return sorted(week_cols, key=lambda s: int(s.split(week_prefix)[1]))

    def select_indices(df: pd.DataFrame, week_cols: List[str], strategy: str, sample_size: Optional[int],
                       hybrid_k: int = 25, random_seed: Optional[int] = None):
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
        n_total = len(matches_df)
        if sample_size is None or sample_size >= n_total:
            return matches_df.copy().reset_index(drop=True)
        sel_idx = select_indices(matches_df, week_cols, strategy=sample_strategy, sample_size=sample_size, random_seed=random_seed)
        sel_idx = np.array(sel_idx, dtype=int)
        return matches_df.iloc[sel_idx].reset_index(drop=True)

    def load_matches(sim_csv: str, observed: List[int], header_rows: int, week_prefix: str) -> Dict:
        try:
            from .trajectory import trajectory_match_pmo  # type: ignore
            return trajectory_match_pmo(observed_weeks=observed, simulated_csv=sim_csv, header_rows=header_rows, week_prefix=week_prefix, return_matches_df=True)
        except Exception:
            return {}
    DEFAULT_R_MIN = 1
    DEFAULT_R_MAX = 10
    print("WARNING: Using local fallback PMO helpers (couldn't import plot_pmo_vs_r_refractor).")
    if _import_trace:
        print(_import_trace)
else:
    print("Using remote PMO helpers from plot_pmo_vs_r_refractor (preferred).")

# analytic helper
try:
    from outbreak_probabilities.analytic.analytical_refractor import compute_pmo_from_string
except Exception:
    try:
        from ..analytic.analytical_refractor import compute_pmo_from_string  # type: ignore
    except Exception:
        compute_pmo_from_string = None

# Configs
PACKAGE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_DIR / "machine_learning" / "Model_SIM"
OUT_DIR = Path(__file__).resolve().parent

DATA_SIZES = [500 * i for i in range(1, 70)]
MODEL_NAMES = ["GB", "RF"]

COL_GB = "darkorange"
COL_RF = "royalblue"
COL_PMO = "xkcd:azure"
COL_ANALYTIC = "red"

DEFAULT_MATCHED_CSV = "figs/pmo_vs_r_matched_trajectories_full.csv"

# ML loader (same as your existing)
def load_ml_curves(sample_tuple: Tuple[int, ...], model_dir: Path, data_sizes: List[int], model_names: List[str]) -> Dict[str, List[Optional[float]]]:
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


def compute_analytic_for_initial(initial_sample: Tuple[int, ...], sim_csv_for_R: Optional[str] = None) -> float:
    if compute_pmo_from_string is None:
        return float("nan")
    try:
        R_min_val = DEFAULT_R_MIN if DEFAULT_R_MIN is not None else 1
        R_max_val = DEFAULT_R_MAX if DEFAULT_R_MAX is not None else 10
        if sim_csv_for_R:
            try:
                total_df = pd.read_csv(sim_csv_for_R, header=3)
                r_col_candidates = [c for c in ("R_draw", "R", "r_draw", "r") if c in total_df.columns]
                if r_col_candidates:
                    r_col = r_col_candidates[0]
                    R_min_val = int(total_df[r_col].min())
                    R_max_val = int(total_df[r_col].max())
            except Exception:
                R_min_val, R_max_val = DEFAULT_R_MIN, DEFAULT_R_MAX
        initial_cases_str = ",".join(str(int(x)) for x in initial_sample) if initial_sample else ""
        res = compute_pmo_from_string(initial_cases_str, nR=2001, R_min=R_min_val, R_max=R_max_val)
        return float(res.get("PMO", float("nan")))
    except Exception:
        return float("nan")

# Main: plot ML curves and PMO step function
def make_comparison_plot(
    initial_sample: Tuple[int, ...] = (1, 2, 0),
    matched_csv: str = DEFAULT_MATCHED_CSV,
    model_dir: Path = MODEL_DIR,
    data_sizes: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    out_png: str = None,
    ci_n_boot: int = 500,
    ci_level: float = 0.90,
    sim_csv_for_inference: Optional[str] = None,
    header_rows: int = 3,
    sample_strategy: str = "random",
    sample_size: Optional[int] = 200,
    sort_by: str = "sample_order",
):
    if data_sizes is None:
        data_sizes = DATA_SIZES
    if model_names is None:
        model_names = MODEL_NAMES
    if out_png is None:
        out_png = str(OUT_DIR / f"comparison_initial_{'_'.join(map(str, initial_sample))}.png")

    data_sizes = np.array(data_sizes, dtype=int)
    x_min = float(np.min(data_sizes))
    x_max = float(np.max(data_sizes))

    # ML
    ml_results = load_ml_curves(initial_sample, model_dir, data_sizes.tolist(), model_names)

    # load matches: prefer trajectory matcher if available, otherwise attempt to build matches
    matches_df = None
    try:
        matches_res = load_matches(sim_csv=sim_csv_for_inference if sim_csv_for_inference else "", observed=list(initial_sample), header_rows=header_rows, week_prefix="week_")
        if isinstance(matches_res, dict) and matches_res.get("matches_df") is not None:
            matches_df = matches_res.get("matches_df")
    except Exception:
        matches_df = None

    # If load_matches failed, attempt to build matches directly from sim_csv_for_inference (exact matching)
    if matches_df is None and sim_csv_for_inference:
        try:
            total_df = pd.read_csv(sim_csv_for_inference, header=header_rows)
            week_cols = get_week_columns(total_df, "week_")
            # only try exact match on the prefix length of observed sample
            k = len(initial_sample)
            wanted_cols = week_cols[:k]
            if len(wanted_cols) == k:
                mask = np.ones(len(total_df), dtype=bool)
                for col, val in zip(wanted_cols, initial_sample):
                    mask = mask & (total_df[col].astype(int) == int(val))
                cand = total_df.loc[mask].copy()
                if not cand.empty:
                    # ensure sim_id 1-based
                    cand = cand.reset_index()
                    cand = cand.rename(columns={"index": "sim_row_index"})
                    cand["sim_id"] = cand["sim_row_index"].astype(int) + 1
                    # If PMO column exists in sim CSV, use it; otherwise try common fallbacks
                    if "PMO" in cand.columns:
                        cand["PMO"] = cand["PMO"].astype(int)
                    elif "major_outbreak" in cand.columns:
                        cand["PMO"] = cand["major_outbreak"].astype(int)
                    elif "final_size" in cand.columns:
                        # heuristic: final_size > 0 -> major outbreak (user can override)
                        cand["PMO"] = (cand["final_size"].astype(float) > 0.0).astype(int)
                    else:
                        # We need PMO to be present; can't proceed without it
                        cand = cand.drop(columns=[c for c in cand.columns if c.startswith("week_")], errors="ignore")
                        raise RuntimeError("sim_csv_for_inference provided but no PMO/major_outbreak/final_size column found to infer outbreak label.")
                    matches_df = cand.copy()
        except Exception:
            matches_df = None

    # final fallback: read provided matched_csv path
    if matches_df is None:
        try:
            matches_df = pd.read_csv(matched_csv)
        except Exception as e:
            raise RuntimeError(f"Could not obtain matched dataframe (tried load_matches, sim csv, and matched_csv). Last error: {e}")

    # --- SAVE per-initial matched CSV so comparisons are reproducible for this initial_sample ---
    matched_save_name = f"matched_initial_{'_'.join(map(str, initial_sample))}.csv"
    matched_save_path = Path(out_png).with_name(matched_save_name) if out_png is not None else (OUT_DIR / matched_save_name)
    Path(matched_save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        matches_df.to_csv(matched_save_path, index=False)
    except Exception:
        matches_df.reset_index().to_csv(matched_save_path, index=False)
    saved_matched_csv = str(matched_save_path)

    # ensure sim_id 1-based and PMO present
    if "sim_id" not in matches_df.columns:
        if "match_index" in matches_df.columns:
            matches_df["sim_id"] = matches_df["match_index"].astype(int) + 1
        elif "sim_row_index" in matches_df.columns:
            matches_df["sim_id"] = matches_df["sim_row_index"].astype(int) + 1
        else:
            matches_df = matches_df.reset_index().rename(columns={"index": "sim_id"})
            matches_df["sim_id"] = matches_df["sim_id"].astype(int) + 1

    if "PMO" not in matches_df.columns:
        raise RuntimeError("matched dataframe must contain a 'PMO' column (0/1 flags).")

    # keep only matches with sim_id <= max(data_sizes) (user request)
    max_S = int(np.max(data_sizes))
    matches_df = matches_df[matches_df["sim_id"] <= max_S].copy()
    matches_df = matches_df.sort_values("sim_id").reset_index(drop=True)

    # prepare sample (same ordering/selection logic)
    week_cols = get_week_columns(matches_df, "week_") if "week_0" in " ".join(matches_df.columns) or any(c.startswith("week_") for c in matches_df.columns) else []
    sampled_df = prepare_sample(matches_df=matches_df, week_cols=week_cols, sample_strategy=sample_strategy, sample_size=sample_size, random_seed=42)
    if sort_by != "sample_order":
        if sort_by == "by_cumulative" and week_cols:
            sampled_df = sampled_df.assign(_key=sampled_df[week_cols].sum(axis=1)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_peak" and week_cols:
            sampled_df = sampled_df.assign(_key=sampled_df[week_cols].max(axis=1)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_R" and "R_draw" in sampled_df.columns:
            sampled_df = sampled_df.assign(_key=sampled_df["R_draw"]).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        elif sort_by == "by_PMO":
            sampled_df = sampled_df.assign(_key=sampled_df["PMO"].astype(int)).sort_values("_key", ascending=False).drop(columns=["_key"]).reset_index(drop=True)
        else:
            sampled_df = sampled_df.reset_index(drop=True)
    else:
        sampled_df = sampled_df.reset_index(drop=True)

    # Build events: sim_id (1-based), PMO, event_order, cum_pmo
    if sampled_df.shape[0] == 0:
        events_df = pd.DataFrame(columns=["sim_id", "PMO", "event_order", "cum_pmo"])
    else:
        sel_sim_ids = sampled_df["sim_id"].astype(int).to_numpy()
        sel_pmo = sampled_df["PMO"].astype(int).to_numpy()
        event_order = np.arange(1, sel_pmo.size + 1)
        cum_pmo = np.cumsum(sel_pmo).astype(float) / event_order.astype(float)
        events_df = pd.DataFrame({
            "sim_id": sel_sim_ids,
            "PMO": sel_pmo,
            "event_order": event_order,
            "cum_pmo": cum_pmo,
        })

    # analytic
    analytic_val = compute_analytic_for_initial(initial_sample, sim_csv_for_R=sim_csv_for_inference) if compute_pmo_from_string is not None else float("nan")

    # ---------------------------
    # NEW: Build per-training-size PMO (correct mapping)
    # For each training data size S (500,1000,...), compute empirical PMO among
    # all matches with sim_id <= S (i.e. ML trained on the first S sims).
    # This yields a step-like curve that updates only when new matches fall inside S.
    # ---------------------------

    # Ensure matches_df is sorted by sim_id and limited to <= max_S
    matches_sorted = matches_df.sort_values("sim_id").reset_index(drop=True)
    max_S = int(np.max(data_sizes))

    # events_df: full set of matches used for step visualization (all sim_id <= max_S)
    events_df = matches_sorted[matches_sorted["sim_id"] <= max_S].copy()
    if "PMO" in events_df.columns:
        events_df["PMO"] = events_df["PMO"].astype(int)
    else:
        raise RuntimeError("matched dataframe must contain a 'PMO' column (0/1 flags).")

    # Preallocate arrays for ML x axis (one value per data_size)
    ml_x = data_sizes.astype(float)
    n_sizes = ml_x.size
    ml_pmo_on_ml_x = np.full(n_sizes, np.nan, dtype=float)
    ml_lower = np.full(n_sizes, np.nan, dtype=float)
    ml_upper = np.full(n_sizes, np.nan, dtype=float)

    # For each training size S, compute PMO among matches with sim_id <= S
    for i, S in enumerate(data_sizes):
        sel = events_df[events_df["sim_id"] <= int(S)]
        if sel.shape[0] == 0:
            # no matches yet --> treat as NaN so plotting shows empty until first match
            ml_pmo_on_ml_x[i] = np.nan
            ml_lower[i] = np.nan
            ml_upper[i] = np.nan
            continue
        flags = sel["PMO"].astype(int).to_numpy()
        ml_pmo_on_ml_x[i] = float(flags.mean())
        # compute shuffle CI (pointwise) on the flags used up to this S
        try:
            lower_i, upper_i = compute_running_ci(pmo_flags=flags, n_boot=ci_n_boot, ci=ci_level, random_seed=42)
            # compute_running_ci returns arrays length = R (R = len(flags)) representing
            # cumulative running PMO distribution for event-order 1..R. We want the final cumulative PMO
            # (i.e., final event), so take the last element from those arrays.
            if lower_i.size > 0 and upper_i.size > 0:
                ml_lower[i] = float(lower_i[-1])
                ml_upper[i] = float(upper_i[-1])
            else:
                ml_lower[i] = np.nan
                ml_upper[i] = np.nan
        except Exception:
            ml_lower[i] = np.nan
            ml_upper[i] = np.nan

    # Build a continuous step visualization for the sim_id timeline (xs_step, ys_step)
    # so we can show the "true" step function aligned to sim-id space (helpful for diagnostics).
    if events_df.shape[0] == 0:
        xs_step = np.array([x_min, x_max], dtype=float)
        ys_step = np.array([0.0, 0.0], dtype=float)
    else:
        # Map sim_id domain [1..max_S] to ML x domain [x_min..x_max] for plotting the step in ML x-space
        event_sim_ids = events_df["sim_id"].astype(float).to_numpy()
        event_cum = np.cumsum(events_df["PMO"].astype(int).to_numpy()).astype(float) / (np.arange(1, len(events_df) + 1).astype(float))
        mapped_event_x = np.interp(event_sim_ids, [1.0, float(max_S)], [x_min, x_max])
        # create step arrays (hold previous value until event, then jump)
        xs_list = [x_min]
        ys_list = [0.0]
        for mx, newval in zip(mapped_event_x, event_cum):
            # hold previous until event mx, then step to newval at mx
            xs_list.append(mx)
            ys_list.append(ys_list[-1])
            xs_list.append(mx)
            ys_list.append(newval)
        # end at x_max with final value
        xs_list.append(x_max)
        ys_list.append(float(event_cum[-1]))
        xs_step = np.array(xs_list, dtype=float)
        ys_step = np.array(ys_list, dtype=float)

    # At this point:
    # - ml_x: ML x positions (training sizes)
    # - ml_pmo_on_ml_x: empirical PMO value for each ML training size (nan before first match)
    # - ml_lower / ml_upper: CI for each ML training size (nan when unavailable)
    # - xs_step / ys_step: fine-grained step coordinates mapped to ML x-space for plotting the step


        # ----------------
    # Plotting + save + metadata + return
    # ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    # ML curves
    for m, col in zip(model_names, (COL_GB, COL_RF)):
        y = np.array([v if v is not None else np.nan for v in ml_results[m]])
        ax.plot(ml_x, y, label=f"{m} predicted PMO", color=col, linewidth=2.5)

    # analytic
    if analytic_val is not None and not np.isnan(analytic_val):
        ax.axhline(analytic_val, color=COL_ANALYTIC, linestyle="--", linewidth=2.0, label=f"Analytic = {analytic_val:.5f}")
        ax.fill_between(ml_x, analytic_val - 0.05, analytic_val + 0.05, color=COL_ANALYTIC, alpha=0.08, label="Analytic ±5%")

    # PMO step function (xs_step, ys_step) and mapped points at ML x
    ax.plot(xs_step, ys_step, color=COL_PMO, linewidth=1.8, drawstyle="steps-post",
            label=f"Trajectory-matching step PMO (events={len(events_df)})", zorder=3)
    ax.plot(ml_x, ml_pmo_on_ml_x, color=COL_PMO, linewidth=0.9, alpha=0.8, linestyle="--",
            label="Mapped to ML x (interpolated)", zorder=2)

    # scatter event points (if any)
    if len(events_df):
        # mapped_event_x and event_cum were built earlier
        try:
            ax.scatter(mapped_event_x, event_cum, s=20, color=COL_PMO, edgecolors="none", zorder=4)
        except NameError:
            # fallback: use mapped positions from xs_step/ys_step by taking the pairs we created earlier
            jump_idxs = np.where(np.diff(xs_step) == 0)[0] + 1
            ax.scatter(xs_step[jump_idxs], ys_step[jump_idxs], s=20, color=COL_PMO, edgecolors="none", zorder=4)

    ax.set_xlabel("Training data size (number of simulations)")
    ax.set_ylabel("P(major outbreak) / PMO")
    ax.set_title(f"ML predictions vs Trajectory-matching PMO (step by sim_id) — initial {tuple(initial_sample)}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xscale('log')
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save events CSV and metadata JSON next to PNG
    events_csv = Path(out_png).with_name(Path(out_png).stem + "_events.csv")
    try:
        events_df.to_csv(events_csv, index=False)
    except Exception:
        pd.DataFrame(columns=["sim_id", "PMO", "event_order", "cum_pmo"]).to_csv(events_csv, index=False)

    meta = {
        "initial": tuple(initial_sample),
        "data_sizes": data_sizes.tolist(),
        "model_dir": str(model_dir),
        "ml_models": model_names,
        # point at the saved per-initial matched CSV
        "matched_csv": saved_matched_csv if 'saved_matched_csv' in locals() else str(matched_csv),
        "analytic_value_used": None if analytic_val is None or np.isnan(analytic_val) else float(analytic_val),
        "n_events": int(len(events_df)) if "events_df" in locals() else 0,
        "max_sim_id_used": int(max_S),
    }


    meta_path = Path(out_png).with_suffix(".json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved plot: {out_png}")
    print(f"Saved events CSV: {events_csv}")
    print(f"Saved metadata JSON: {meta_path}")

    # return final paths/meta as expected by caller
    return str(out_png), meta


def main():
    INITIAL = (1, 2, 0)
    MATCHED_CSV = "figs/pmo_vs_r_matched_trajectories_full.csv"
    SIM_CSV_FOR_INFERENCE = None
    OUT_PNG = str(OUT_DIR / f"comparison_initial_{'_'.join(map(str, INITIAL))}.png")

    try:
        result = make_comparison_plot(
            initial_sample=INITIAL,
            matched_csv=MATCHED_CSV,
            model_dir=MODEL_DIR,
            data_sizes=DATA_SIZES,
            model_names=MODEL_NAMES,
            out_png=OUT_PNG,
            ci_n_boot=500,
            ci_level=0.90,
            sim_csv_for_inference=SIM_CSV_FOR_INFERENCE,
            header_rows=3,
            sample_strategy="random",
            sample_size=200,
            sort_by="sample_order",
        )
        # Defensive check
        if result is None:
            raise RuntimeError("make_comparison_plot returned None — check for swallowed exceptions inside the function.")
        out, meta = result
        print("Saved:", out)
        print("Meta:", json.dumps(meta, indent=2))
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        print("make_comparison_plot failed. See traceback above.")
        raise


if __name__ == "__main__":
    main()

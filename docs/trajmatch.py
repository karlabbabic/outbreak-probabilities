import base64
import io
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy.random import default_rng
from numpy.polynomial.legendre import leggauss
from scipy.stats import gamma as scipy_gamma

from pyscript import document


MEAN_SI_DAYS = 15.3
SD_SI_DAYS = 9.3
BLUE = "tab:blue"
GRAY = "dimgray"
ALPHA = 0.6
LINEWIDTH = 1.0

# Browser-side app state.
SIM_TRAJECTORIES = None       # shape: (N, max_weeks)
SIM_WEEK_TABLE = None         # shape: (N, write_weeks)
SIM_R_DRAWS = None            # shape: (N,)
SIM_PMO_FLAGS = None          # shape: (N,)
SIM_STATUS = None             # shape: (N,)
SIM_IDS = None                # shape: (N,)
SIM_CONFIG = None
LATEST_MATCH_MASK = None
LATEST_OBSERVED = None


def by_id(element_id):
    return document.getElementById(element_id)


def set_text(element_id, text):
    el = by_id(element_id)
    if el is not None:
        el.textContent = str(text)


def value_of(element_id):
    el = by_id(element_id)
    if el is None:
        raise ValueError(f"Missing HTML element: {element_id}")
    return el.value


def int_value(element_id, default=None):
    raw = str(value_of(element_id)).strip()
    if raw == "" and default is not None:
        return int(default)
    return int(float(raw))


def float_value(element_id, default=None):
    raw = str(value_of(element_id)).strip()
    if raw == "" and default is not None:
        return float(default)
    return float(raw)


def set_button_busy(button_id, busy, busy_text=None, ready_text=None):
    button = by_id(button_id)
    if button is not None:
        button.disabled = bool(busy)
        if busy and busy_text is not None:
            button.textContent = busy_text
        if (not busy) and ready_text is not None:
            button.textContent = ready_text


def clear_image(image_id, placeholder_id, placeholder_text):
    image = by_id(image_id)
    placeholder = by_id(placeholder_id)
    if image is not None:
        image.removeAttribute("src")
        image.style.display = "none"
    if placeholder is not None:
        placeholder.textContent = placeholder_text
        placeholder.style.display = "block"


def show_png(image_id, placeholder_id, png_bytes):
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    image = by_id(image_id)
    placeholder = by_id(placeholder_id)
    if image is not None:
        image.src = f"data:image/png;base64,{encoded}"
        image.style.display = "block"
    if placeholder is not None:
        placeholder.style.display = "none"


def fig_to_png(fig, dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@lru_cache(maxsize=64)
def compute_serial_weights(mean, std, k_max, nquad=64, step=7.0):
    """Weekly serial-interval weights using a triangular kernel and Gauss-Legendre quadrature."""
    k_max = int(k_max)
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if mean <= 0.0 or std <= 0.0:
        raise ValueError("mean and std must be positive")

    var = std ** 2
    shape = (mean / std) ** 2
    scale = var / mean
    g = scipy_gamma(a=shape, scale=scale)

    if k_max == 1:
        return np.array([1.0], dtype=float)

    nodes, quad_weights = leggauss(int(nquad))
    w = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        center = step * k
        left = step * (k - 1)
        right = step * (k + 1)
        half_width = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)
        u = half_width * nodes + midpoint

        tri = 1.0 - np.abs(u - center) / step
        tri[tri < 0.0] = 0.0
        w[k - 1] = half_width * np.sum(quad_weights * tri * g.pdf(u))

    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Serial-interval weights sum to a non-positive or non-finite value")

    w[0] = max(0.0, 1.0 - w[1:].sum())
    w /= w.sum()
    return w


def calculate_R(R_range, rng=None):
    if rng is None:
        rng = default_rng()
    rmin, rmax = float(R_range[0]), float(R_range[1])
    if rmax < rmin:
        raise ValueError("R maximum must be greater than or equal to R minimum")
    if rmin == rmax:
        return float(rmin)
    return float(rng.uniform(rmin, rmax))


def is_extinct_window(trajectory_prefix, window):
    if not window or window <= 0:
        return False
    if trajectory_prefix.size < window:
        return False
    return bool(np.all(trajectory_prefix[-window:] == 0))


def simulate_trajectory(w, max_weeks=50, R=None, initial_cases=None,
                        rng=None, extinction_window=10, major_threshold=100,
                        stop_on_major=True):
    if rng is None:
        rng = default_rng()
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1:
        raise ValueError("w must be one-dimensional")

    R = float(R)
    initial = [1] if initial_cases is None else list(initial_cases)
    trajectory = np.zeros(int(max_weeks), dtype=int)
    L = min(len(initial), int(max_weeks))
    trajectory[:L] = np.asarray(initial[:L], dtype=int)
    cumulative = int(trajectory[:L].sum())

    major_flag = cumulative >= major_threshold
    extinct_flag = False
    t_end = L

    for t in range(L, int(max_weeks)):
        max_lag = min(w_arr.size, t)
        if max_lag == 0:
            lam_base = 0.0
        else:
            past = trajectory[t - max_lag:t]
            ws = w_arr[:max_lag]
            lam_base = float(np.dot(ws, past[::-1]))

        lam = R * lam_base
        new_cases = int(rng.poisson(lam)) if lam > 0.0 else 0
        trajectory[t] = new_cases
        cumulative += new_cases
        t_end = t + 1

        if cumulative >= major_threshold:
            major_flag = True
            if stop_on_major:
                break

        if extinction_window is not None and is_extinct_window(trajectory[:t + 1], extinction_window):
            extinct_flag = True
            break

    if major_flag:
        status, pmo = "major", 1
    elif extinct_flag:
        status, pmo = "minor", 0
    else:
        status, pmo = "ongoing", 0

    return trajectory, R, cumulative, status, pmo, t_end


@dataclass
class SimConfig:
    N: int = 5000
    max_weeks: int = 50
    mean_serial: float = MEAN_SI_DAYS
    std_serial: float = SD_SI_DAYS
    k_max: int = 50
    nquad: int = 32
    step: float = 7.0
    R_min: float = 0.0
    R_max: float = 10.0
    initial_cases: tuple = (1,)
    extinction_window: int = 10
    major_threshold: int = 100
    seed: int = 42
    write_weeks: int = 5


def simulate_paths(cfg):
    master_rng = default_rng(cfg.seed)
    w = compute_serial_weights(cfg.mean_serial, cfg.std_serial, cfg.k_max, cfg.nquad, cfg.step)
    if len(w) < cfg.max_weeks:
        raise RuntimeError(f"weights length {len(w)} < max_weeks {cfg.max_weeks}; increase k_max")

    n_write = min(max(1, int(cfg.write_weeks)), int(cfg.max_weeks))
    N = int(cfg.N)
    max_weeks = int(cfg.max_weeks)
    trajectories = np.zeros((N, max_weeks), dtype=int)
    week_table = np.zeros((N, n_write), dtype=int)
    r_draws = np.zeros(N, dtype=float)
    pmo_flags = np.zeros(N, dtype=int)
    status = np.empty(N, dtype=object)
    sim_ids = np.arange(1, N + 1, dtype=int)

    for sim_idx in range(N):
        sim_seed = int(master_rng.integers(low=0, high=2**63 - 1, dtype=np.int64))
        child_rng = default_rng(sim_seed)
        R = calculate_R((cfg.R_min, cfg.R_max), rng=child_rng)
        traj, R, cumulative, stat, pmo, t_end = simulate_trajectory(
            w=w,
            max_weeks=max_weeks,
            R=R,
            initial_cases=cfg.initial_cases,
            rng=child_rng,
            extinction_window=cfg.extinction_window,
            major_threshold=cfg.major_threshold,
        )
        trajectories[sim_idx, :] = traj
        week_table[sim_idx, :] = traj[:n_write]
        r_draws[sim_idx] = R
        pmo_flags[sim_idx] = int(pmo)
        status[sim_idx] = stat

    return trajectories, week_table, r_draws, pmo_flags, status, sim_ids, w


def parse_observed_weeks(raw):
    if raw is None or str(raw).strip() == "":
        raise ValueError("Please enter observed weekly counts, for example 1,0,0")
    cleaned = str(raw).replace(";", ",").replace("\n", ",").replace(" ", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
    observed = []
    for p in parts:
        value = int(float(p))
        if value < 0:
            raise ValueError("Observed weekly counts must be non-negative integers")
        observed.append(value)
    if len(observed) == 0:
        raise ValueError("Please enter at least one observed weekly count")
    return np.asarray(observed, dtype=int)


def select_indices(indices, trajectories_subset, r_subset, pmo_subset, strategy, sample_size, random_seed=42):
    n = len(indices)
    if sample_size is None or sample_size >= n:
        return np.arange(n, dtype=int)

    rng = default_rng(random_seed)
    sample_size = int(max(1, sample_size))

    if strategy == "random":
        return rng.choice(n, size=sample_size, replace=False)

    cumulative = trajectories_subset.sum(axis=1)
    peak = trajectories_subset.max(axis=1)

    if strategy == "highest_cumulative":
        return np.argsort(-cumulative)[:sample_size]
    if strategy == "highest_peak":
        return np.argsort(-peak)[:sample_size]
    if strategy == "highest_R":
        return np.argsort(-r_subset)[:sample_size]
    if strategy == "hybrid":
        k = min(25, max(1, sample_size // 4))
        idx_set = set(np.argsort(-cumulative)[:k].tolist())
        idx_set.update(np.argsort(-peak)[:k].tolist())
        idx_set.update(np.argsort(-r_subset)[:k].tolist())
        remaining = sample_size - len(idx_set)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), np.fromiter(idx_set, int))
            chosen = rng.choice(pool, size=remaining, replace=False)
            idx_set.update(chosen.tolist())
        return np.fromiter(sorted(idx_set), dtype=int)

    raise ValueError(f"Unknown sample strategy: {strategy}")


def trajectory_match_pmo(observed_weeks):
    global SIM_WEEK_TABLE, SIM_PMO_FLAGS
    if SIM_WEEK_TABLE is None or SIM_PMO_FLAGS is None:
        raise ValueError("No simulation data exists yet. Click 'Generate simulation data' first.")

    observed = np.asarray(observed_weeks, dtype=int)
    k = observed.size
    if k > SIM_WEEK_TABLE.shape[1]:
        raise ValueError(
            f"Observed data has {k} weeks, but only {SIM_WEEK_TABLE.shape[1]} week columns were retained. "
            "Increase 'Week columns retained for matching' and regenerate simulations."
        )

    matches_mask = np.all(SIM_WEEK_TABLE[:, :k] == observed.reshape(1, -1), axis=1)
    matched_indices = np.nonzero(matches_mask)[0]
    n_matches = int(matches_mask.sum())
    n_major = 0
    pmo_fraction = None
    if n_matches > 0:
        flags = SIM_PMO_FLAGS[matches_mask].astype(int)
        n_major = int((flags == 1).sum())
        pmo_fraction = float(n_major) / float(n_matches)

    return {
        "matches_mask": matches_mask,
        "matched_indices": matched_indices,
        "n_matches": n_matches,
        "n_major": n_major,
        "pmo_fraction": pmo_fraction,
    }


def prepare_plot_data(match_indices, sample_strategy, sample_size, major_threshold):
    full_traj = SIM_TRAJECTORIES[match_indices, :]
    r_subset = SIM_R_DRAWS[match_indices]
    pmo_subset = SIM_PMO_FLAGS[match_indices]
    chosen_local = select_indices(
        match_indices,
        full_traj,
        r_subset,
        pmo_subset,
        strategy=sample_strategy,
        sample_size=sample_size,
        random_seed=42,
    )

    arr = full_traj[chosen_local, :].astype(float)
    flags = pmo_subset[chosen_local].astype(int)
    cumul = np.cumsum(arr, axis=1)
    reached = (cumul >= major_threshold).any(axis=1)
    hit_idx = np.argmax(cumul >= major_threshold, axis=1)
    return arr, flags, cumul, reached, hit_idx


def make_matched_plot(arr, pmo_flags, reached, hit_idx, n_matches, pmo_fraction, observed, major_threshold, sample_strategy):
    n_weeks = arr.shape[1]
    plotted = arr.shape[0]
    weeks = np.arange(1, n_weeks + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, weekly in enumerate(arr):
        color = BLUE if pmo_flags[i] == 1 else GRAY
        if reached[i]:
            k = int(hit_idx[i])
            ax.plot(weeks[:k + 1], weekly[:k + 1], color=color, alpha=ALPHA, linewidth=LINEWIDTH)
        else:
            ax.plot(weeks, weekly, color=color, alpha=ALPHA, linewidth=LINEWIDTH)

    if reached.any():
        local_rows = np.nonzero(reached)[0]
        xs = weeks[hit_idx[reached]]
        ys = arr[local_rows, hit_idx[reached]]
        ax.scatter(xs, ys, color="red", s=20, zorder=3, label="cutoff")

    pmo_label = f"PMO matched = {pmo_fraction:.3f}" if pmo_fraction is not None else "PMO matched = NA"
    ax.plot([], [], color=BLUE, label="PMO = 1 matched")
    ax.plot([], [], color=GRAY, label="PMO = 0 matched")
    ax.plot([], [], " ", label=pmo_label)

    ymax = max(float(np.nanmax(arr)), 1.0) + 1.0 if arr.size else 1.0
    ax.set_xlim(1, n_weeks + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(1, n_weeks + 1, 1))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    obs_str = ", ".join(str(int(x)) for x in observed)
    ax.set_title(
        f"Matched trajectories: plotted {plotted} of {n_matches}\n"
        f"Observed initial cases: {obs_str} | Cutoff: cumulative >= {major_threshold}\n"
        f"Sampling: {sample_strategy}"
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(which="major", color="0.85", linewidth=0.6, alpha=0.8)
    return fig_to_png(fig, dpi=170)


def compute_running_ci(pmo_flags, n_boot=300, ci=0.90, random_seed=42):
    pmo_flags = np.asarray(pmo_flags, dtype=int)
    if pmo_flags.size == 0 or n_boot <= 0:
        return np.array([]), np.array([])
    R = pmo_flags.size
    rng = default_rng(random_seed)
    runs = np.empty((int(n_boot), R), dtype=float)
    denom = np.arange(1, R + 1, dtype=float)
    for i in range(int(n_boot)):
        perm_flags = pmo_flags[rng.permutation(R)]
        runs[i, :] = np.cumsum(perm_flags, dtype=float) / denom
    alpha = 1.0 - float(ci)
    lower = np.percentile(runs, 100.0 * (alpha / 2.0), axis=0)
    upper = np.percentile(runs, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper


def make_running_pmo_plot(flags, observed, sample_strategy, ci_boot):
    flags = np.asarray(flags, dtype=int)
    R = flags.size
    fig, ax = plt.subplots(figsize=(8, 5))
    if R == 0:
        ax.text(0.5, 0.5, "No matched trajectories", ha="center", va="center")
        return fig_to_png(fig)

    running = np.cumsum(flags) / np.arange(1, R + 1, dtype=float)
    x = np.arange(1, R + 1)

    if ci_boot and ci_boot > 0:
        lower, upper = compute_running_ci(flags, n_boot=ci_boot, ci=0.90, random_seed=42)
        if lower.size == R:
            ax.fill_between(x, lower, upper, alpha=0.18, color=BLUE, label="90% band random shuffling")

    idx = np.linspace(0, R - 1, min(50, R), dtype=int)
    ax.scatter(x[idx], running[idx], s=28, c=BLUE, edgecolors="none", alpha=0.5)
    ax.plot(x, running, linewidth=2.0, color=BLUE, label=f"Running PMO = {running[-1]:.5f}")
    ax.set_xlabel("Sampled matched trajectory index")
    ax.set_ylabel("Cumulative PMO fraction")
    obs_str = ", ".join(str(int(v)) for v in observed)
    ax.set_title(f"Cumulative PMO across matched trajectories\nObserved initial cases: {obs_str} | Sampling: {sample_strategy}")
    ax.set_xlim(1, max(1, R))
    ax.set_ylim(-0.01, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, which="major", linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return fig_to_png(fig, dpi=170)


def reset_outputs_for_new_simulation():
    set_text("pmo-value-output", "-")
    set_text("matches-output", "-")
    set_text("major-output", "-")
    clear_image("matched-plot-image", "matched-plot-placeholder", "Matched trajectory plot will appear here.")
    clear_image("running-pmo-plot-image", "running-pmo-plot-placeholder", "Running PMO plot will appear here.")


def run_simulation(event=None):
    global SIM_TRAJECTORIES, SIM_WEEK_TABLE, SIM_R_DRAWS, SIM_PMO_FLAGS, SIM_STATUS, SIM_IDS, SIM_CONFIG
    global LATEST_MATCH_MASK, LATEST_OBSERVED

    set_button_busy("simulate-button", True, "Generating...", "Generate simulation data")
    set_text("diagnostics-output", "Generating simulation data. This may take a moment in the browser...")
    reset_outputs_for_new_simulation()

    try:
        N = int_value("num-simulations-input")
        max_weeks = int_value("max-weeks-input")
        write_weeks = int_value("write-weeks-input")
        major_threshold = int_value("major-threshold-input")
        r_min = float_value("r-min-input")
        r_max = float_value("r-max-input")
        seed = int_value("seed-input", 42)
        extinction_window = int_value("extinction-window-input", 10)

        if N < 1:
            raise ValueError("Number of simulations must be at least 1")
        if max_weeks < 1:
            raise ValueError("Maximum simulated weeks must be at least 1")
        if write_weeks > max_weeks:
            raise ValueError("Week columns retained cannot exceed maximum simulated weeks")

        cfg = SimConfig(
            N=N,
            max_weeks=max_weeks,
            k_max=max(50, max_weeks),
            nquad=32,
            R_min=r_min,
            R_max=r_max,
            initial_cases=(1,),
            extinction_window=extinction_window,
            major_threshold=major_threshold,
            seed=seed,
            write_weeks=write_weeks,
        )

        trajectories, week_table, r_draws, pmo_flags, status, sim_ids, w = simulate_paths(cfg)
        SIM_TRAJECTORIES = trajectories
        SIM_WEEK_TABLE = week_table
        SIM_R_DRAWS = r_draws
        SIM_PMO_FLAGS = pmo_flags
        SIM_STATUS = status
        SIM_IDS = sim_ids
        SIM_CONFIG = cfg
        LATEST_MATCH_MASK = None
        LATEST_OBSERVED = None

        pmo_rate = float(np.mean(pmo_flags)) if pmo_flags.size else float("nan")
        n_major = int((pmo_flags == 1).sum())
        n_minor = int((status == "minor").sum())
        n_ongoing = int((status == "ongoing").sum())
        set_text(
            "diagnostics-output",
            "Simulation complete.\n"
            f"Simulations: {N}\n"
            f"Retained week columns: {write_weeks}\n"
            f"R range: [{r_min:g}, {r_max:g}]\n"
            f"Major threshold: cumulative >= {major_threshold}\n"
            f"Overall simulated PMO rate: {pmo_rate:.4f}\n"
            f"Major: {n_major} | Minor: {n_minor} | Ongoing: {n_ongoing}\n"
            "Now enter observed weekly counts and click 'Match observed data and estimate PMO'."
        )

    except Exception as exc:
        set_text("diagnostics-output", f"Simulation error:\n{type(exc).__name__}: {exc}")

    finally:
        set_button_busy("simulate-button", False, "Generating...", "Generate simulation data")


def match_observed_data(event=None):
    global LATEST_MATCH_MASK, LATEST_OBSERVED

    set_button_busy("match-button", True, "Matching...", "Match observed data and estimate PMO")
    clear_image("matched-plot-image", "matched-plot-placeholder", "Matched trajectory plot will appear here.")
    clear_image("running-pmo-plot-image", "running-pmo-plot-placeholder", "Running PMO plot will appear here.")

    try:
        observed = parse_observed_weeks(value_of("observed-weeks-input"))
        sample_strategy = value_of("sample-strategy-select")
        sample_size = int_value("sample-size-input")
        ci_boot = int_value("ci-boot-input", 300)
        major_threshold = int_value("major-threshold-input")

        res = trajectory_match_pmo(observed)
        LATEST_MATCH_MASK = res["matches_mask"]
        LATEST_OBSERVED = observed

        n_matches = int(res["n_matches"])
        n_major = int(res["n_major"])
        pmo_fraction = res["pmo_fraction"]

        set_text("matches-output", n_matches)
        set_text("major-output", n_major)
        set_text("pmo-value-output", "NA" if pmo_fraction is None else f"{pmo_fraction:.5f}")

        if n_matches == 0:
            set_text(
                "diagnostics-output",
                "No matching trajectories found.\n"
                f"Observed weeks: {observed.tolist()}\n"
                "Try increasing the number of simulations, using fewer observed weeks, or regenerating with a different seed."
            )
            return

        match_indices = res["matched_indices"]
        arr, flags, cumul, reached, hit_idx = prepare_plot_data(
            match_indices,
            sample_strategy=sample_strategy,
            sample_size=sample_size,
            major_threshold=major_threshold,
        )

        matched_png = make_matched_plot(
            arr=arr,
            pmo_flags=flags,
            reached=reached,
            hit_idx=hit_idx,
            n_matches=n_matches,
            pmo_fraction=pmo_fraction,
            observed=observed,
            major_threshold=major_threshold,
            sample_strategy=sample_strategy,
        )
        show_png("matched-plot-image", "matched-plot-placeholder", matched_png)

        running_png = make_running_pmo_plot(
            flags=flags,
            observed=observed,
            sample_strategy=sample_strategy,
            ci_boot=ci_boot,
        )
        show_png("running-pmo-plot-image", "running-pmo-plot-placeholder", running_png)

        set_text(
            "diagnostics-output",
            "Match complete.\n"
            f"Observed weeks: {observed.tolist()}\n"
            f"Matching rule: first {observed.size} weekly counts exactly equal observed data\n"
            f"Matches: {n_matches} out of {SIM_WEEK_TABLE.shape[0]} simulations\n"
            f"Major outbreaks among matches: {n_major}\n"
            f"Matched empirical PMO: {pmo_fraction:.5f}\n"
            f"Plotted trajectories: {arr.shape[0]}\n"
            f"Sampling strategy: {sample_strategy}"
        )

    except Exception as exc:
        set_text("diagnostics-output", f"Match error:\n{type(exc).__name__}: {exc}")
        set_text("pmo-value-output", "-")
        set_text("matches-output", "-")
        set_text("major-output", "-")

    finally:
        set_button_busy("match-button", False, "Matching...", "Match observed data and estimate PMO")


# Initial page state.
set_text("diagnostics-output", "Ready. Generate simulation data first, then match observed data.")
set_text("pmo-value-output", "-")
set_text("matches-output", "-")
set_text("major-output", "-")

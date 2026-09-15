"""Run this file to plot the predicted outbreak probabilities from ML models(GB and)
trained on different number of simulations (data sizes)
and the analytical solution. Used to visualize when the ML models converges to the analytical solutions."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "nature", "no-latex"])
# consistent look across all package plots: open box (no top/right spine or ticks)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.top": False,
    "ytick.right": False,
})

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

MODEL_DIR = Path(__file__).resolve().parent / "Model_SIM"
DEFAULT_PLOT_DIR = MODEL_DIR / "ML_CONVERGENCE_PLOTS"
DATA_SIZES = [500 * i for i in range(1, 70)]  # up to 35k samples
MODEL_NAMES = ["GB", "RF"]

# analytical solutions for selected samples
SAMPLE_SOLUTIONS = {
    (1, 2, 0): 0.71617,
    (1, 1, 0): 0.53455,
    (1, 0, 0): 0.22406,
    (1, 2, 1): 0.90805,
    (1, 3, 1): 0.94251,
    (1, 0, 1): 0.74969,
    (1, 5, 3): 0.99822,
}


def plot_convergence(model_dir: Path = MODEL_DIR, plot_dir: Path = DEFAULT_PLOT_DIR,
                      data_sizes=DATA_SIZES, model_names=MODEL_NAMES, sample_solutions=None):
    model_dir = Path(model_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    sample_solutions = sample_solutions or SAMPLE_SOLUTIONS

    results = {sample: {"GB": [], "RF": []} for sample in sample_solutions.keys()}

    # load models and make predictions
    for model_name in model_names:
        for size in data_sizes:
            stem = f"ML_SIM_{size}_{model_name}"
            model_path = model_dir / f"{stem}.pkl"
            scaler_path = model_dir / f"{stem}_scaler.pkl"

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            for sample in sample_solutions.keys():
                sample_array = np.array(sample).reshape(1, -1)
                sample_scaled = scaler.transform(sample_array)
                pred_prob = model.predict_proba(sample_scaled)[0][1]
                results[sample][model_name].append(pred_prob)

    # plot results
    for sample in sample_solutions.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(data_sizes, results[sample]["GB"], label="GB Predictions", color="darkorange", linewidth=3)
        plt.plot(data_sizes, results[sample]["RF"], label="RF Predictions", color="royalblue", linewidth=3)
        # plot analytical solution line with label (exact value)
        plt.axhline(
            y=sample_solutions[sample],
            color="red",
            linestyle="--",
            label=f"Analytical Solution ({sample_solutions[sample]:.5f})",
            linewidth=2
        )
        # add confidence interval shading for analytical solution
        plt.fill_between(
            data_sizes,
            sample_solutions[sample] - 0.05,
            sample_solutions[sample] + 0.05,
            color="red",
            alpha=0.1,
            label="Analytical Solution ±5%"
        )
        plt.title(f"Convergence of ML Models for Sample {sample}")
        plt.xlabel("Training Data Size")
        plt.ylabel("Predicted Outbreak Probability")
        plt.ylim(0, 1.13)
        plt.legend()
        plt.grid(alpha=0.25, which="major", linestyle="--")
        plot_path = plot_dir / f"Convergence_Sample_{sample}_GB_RF.png"
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot ML model convergence vs analytical PMO across training sizes.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory containing trained Model_SIM models")
    parser.add_argument("--plot-dir", type=str, default=str(DEFAULT_PLOT_DIR), help="Directory to write convergence plots")
    args = parser.parse_args(argv)

    plot_convergence(model_dir=Path(args.model_dir), plot_dir=Path(args.plot_dir))


if __name__ == "__main__":
    main()

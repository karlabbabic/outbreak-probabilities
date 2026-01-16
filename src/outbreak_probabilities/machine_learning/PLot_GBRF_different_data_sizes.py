"""Run this file to plot the predicted outbreak probabilities from ML models(GB and) trained on different data sizes. It can be used to assess when the ML models converges to the analytical solutions."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import joblib


# set paths
BASE_DIR = Path(__file__).resolve().parents[3]
model_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" /"Model_SIM"
plot_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"/ "ML_CONVERGENCE_PLOTS"
plot_dir.mkdir(parents=True, exist_ok=True)

# analytical solutions for selected samples
sample_solutions = {
    (1, 2, 0): 0.71617,
    (1, 1, 0): 0.53455,
    (1, 0, 0): 0.22406,
    (1, 2, 1): 0.90805,
    (1,3,1): 0.94251,
    (1,0,1): 0.74969,
    # first time it appears is at sample 21441
    (1,5,3): 0.99822
}


data_sizes = [500 * i for i in range(1, 70)]  # up to 35k samples
# results to store results from both RF and GB
results = {sample: {"GB": [], "RF": []} for sample in sample_solutions.keys()}

model_names = ["GB", "RF"]
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
    plt.axhline(y=sample_solutions[sample], color="red", linestyle="--", label="Analytical Solution", linewidth=3)
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
    plt.ylim(0,1.13)
    plt.legend()
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.3)
    plot_path = plot_dir / f"Convergence_Sample_{sample}_GB_RF2.png"
    plt.savefig(plot_path)
# plot the results from ML_Sim.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[3]
model_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"
plot_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"/ "test_plots"
plot_dir.mkdir(parents=True, exist_ok=True)
import json
import numpy as np
import joblib
# Load results and plot predicitions vs data size to assess when does the model converge to the analytical solution
# FIRST use the model to predict on test samples (week_1, week_2, week_3) and then plot the results vs data size used for training

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
results = {sample: [] for sample in sample_solutions.keys()}
for size in data_sizes:
    stem = f"ML_SIM_{size}_RF"
    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"
    # load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    for sample in sample_solutions.keys():
        sample_array = np.array(sample).reshape(1, -1)
        sample_scaled = scaler.transform(sample_array)
        pred_prob = model.predict_proba(sample_scaled)[0][1]  # probability of class 1 (outbreak)
        results[sample].append(pred_prob)
# plot results
for sample in sample_solutions.keys():
    plt.figure()
    plt.plot(data_sizes, results[sample], marker='o', label='ML Prediction')
    plt.axhline(y=sample_solutions[sample], color='r', linestyle='--', label='Analytical Solution = ' + str(sample_solutions[sample]))
    # include analutic solution number and have light gray grid
    plt.title(f"Predicted Outbreak Probability for Sample {sample}")
    plt.xlabel("Training Data Size")
    plt.ylabel("Predicted Outbreak Probability")
    plt.ylim(0, 1.13)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.legend()
    plot_path = plot_dir / f"ML_SIM_convergence_sample_{sample[0]}_{sample[1]}_{sample[2]}.png"
    plt.savefig(plot_path)
    plt.close()
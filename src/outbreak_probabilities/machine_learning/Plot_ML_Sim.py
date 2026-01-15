# plot the results from ML_Sim.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[3]
plot_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"/ "plots2"
plot_dir.mkdir(parents=True, exist_ok=True)
import json
import numpy as np
import joblib
# Load results and plot predicitions vs data size to assess when does the model converge to the analytical solution
# FIRST use the model to predict on test samples (week_1, week_2, week_3) and then plot the results vs data size used for training
test_samples = [
        (1, 2, 0),
        (1, 1, 0),
        (1, 0, 0),
        (1,2,1),
        
    ]

for sample in test_samples:
    week_1, week_2, week_3 = sample
    X_test = np.array([[week_1, week_2, week_3]])
    predictions = []
    data_sizes = []
    for size in range(500, 15000 + 1, 500):
        stem = f"ML_SIM_{size}_RF"
        model_path = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM" / f"{stem}.pkl"
        scaler_path = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM" / f"{stem}_scaler.pkl"
        if model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict_proba(X_test_scaled)[0][1]  # probability of PMO==1
            predictions.append(pred)
            data_sizes.append(size)
    # Plot results and analytical solution as a horizontal line
    plt.figure()
    plt.plot(data_sizes, predictions, marker='o', label='ML Prediction')
    # Analytical solution (for demonstration, using a dummy value, replace with actual calculation)
    analytical_solution = 0.5  # Replace with actual analytical solution calculation
    plt.axhline(y=analytical_solution, color='r', linestyle='--', label='Analytical Solution')
    plt.xlabel('Training Data Size')
    plt.ylabel('Predicted Probability of Major Outbreak')
    plt.title(f'Predicted PMO Probability for Input ({week_1}, {week_2}, {week_3})')
    plt.legend()
    plt.grid()
    plot_path = plot_dir / f"PMO_Prob_{week_1}_{week_2}_{week_3}.png"
    plt.savefig(plot_path)
    plt.close()
  
    

    
import json
from datetime import datetime
from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[3]
data_path = BASE_DIR / "data" / "test_simulations.csv"
model_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"
plot_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "Model_SIM"/ "plots"

model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(data_path)

# remove first two rows
data = data.iloc[2:].reset_index(drop=True)
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data[["week_1", "week_2", "week_3", "PMO"]]
X = data[["week_1", "week_2","week_3",]].astype(float)
y = data["PMO"].astype(int)

models = {
    "RF": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train and dave
feature_names = list(X.columns)  
n_weeks = len(feature_names)

# data subsets (train on 500 samples, then those 500+500 more, then the 500+500+500 more, etc)
data_sizes = [500 * i for i in range(1, 40)]  # up to 5000 samples
results = {}
for size in data_sizes:
    X_subset = X_scaled[:size]
    y_subset = y[:size]
    results[size] = {}
    for model_name, model in models.items():
        model.fit(X_subset, y_subset)
        stem = f"ML_SIM_{size}_{model_name}"
        model_path = model_dir / f"{stem}.pkl"
        scaler_path = model_dir / f"{stem}_scaler.pkl"
        meta_json_path = model_dir / f"{stem}.json"
        meta_jbl_path = model_dir / f"{stem}_meta.pkl"
        joblib.dump(model, model_path, compress=3)
        joblib.dump(scaler, scaler_path, compress=3)
        results[size][model_name] = {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        }
        meta = {
            "model_name": model_name,
            "n_weeks": n_weeks,
            "feature_names": feature_names,
            "data_size": size,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_json_path, "w") as f:
            json.dump(meta, f, indent=4)
        joblib.dump(meta, meta_jbl_path, compress=3)
        
        print(f"Trained and saved model {model_name} with {size} samples.")
# Save overall results
results_path = model_dir / "training_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)
print("\nAll models trained and saved.")


# Prediction helper
def predict_pmo(model_name: str, week_1: float, week_2: float, week_3:float, threshold: float = 0.5):
    """
    Predict the probability of a major outbreak (PMO) given the model name and weekly case counts.
    """
    stem = f"ML_SIM_{size}_{model_name}"
    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"

    # load model and scaler
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # prepare input
    X_input = np.array([[week_1, week_2, week_3]])
    X_scaled = scaler.transform(X_input)

    # predict probability
    proba = clf.predict_proba(X_scaled)[0][1]  # probability of class 1 (major outbreak)

    pred = int(proba >= threshold)
    pred_label = "major" if pred == 1 else "minor"
    
    return {"model": model_name, "probability": proba, "PMO": pred, "predicted_label": pred_label}


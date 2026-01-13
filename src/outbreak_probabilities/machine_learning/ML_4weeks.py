import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# define models
models = {
    "RF": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "GB": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    ),
}

# set paths
BASE_DIR = Path(__file__).resolve().parents[3]
data_path = BASE_DIR / "data" / "simulated_cases_and_serial_interval_and_weights1.csv"
model_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "models_3weeks"

model_dir.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(data_path)

data = data.iloc[1:].reset_index(drop=True)
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data[["week_1", "week_2", "week_3", "PMO"]]

X = data[["week_1", "week_2", "week_3"]].astype(float)
y = data["PMO"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train models on FULL dataset
for model_name, clf in models.items():
    print(f"\nTraining model on full dataset: {model_name}")

    clf.fit(X_scaled, y)

    # save model and scaler
    joblib.dump(
        clf,
        model_dir / f"ML_3weeks_{model_name}.pkl",
        compress=3,
    )
    joblib.dump(
        scaler,
        model_dir / f"ML_3weeks_{model_name}_scaler.pkl",
        compress=3,
    )

print("\nAll models trained and saved.")

import joblib
import numpy as np

def predict_pmo(model_name, week_1, week_2, week_3, threshold=0.5):
    model_path = model_dir / f"ML_3weeks_{model_name}.pkl"
    scaler_path = model_dir / f"ML_3weeks_{model_name}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_new = np.array([[float(week_1), float(week_2)]])
    X_new_scaled = scaler.transform(X_new)

    proba = model.predict_proba(X_new_scaled)[:, 1][0]
    pred = int(proba >= threshold)
    # add major or minor in the reutn if pred == 1 else minor
    if pred == 1:
        pred_label = "major"
    else:
        pred_label = "minor"
    
    return {
        "model": model_name,
        "probability": float(proba),
        "predicted_class": pred,
        "predicted_label": pred_label
        
    }

# example usage
if __name__ == "__main__":
    result = predict_pmo("RF", 2, 1, 0)
    print(result)
import json
from datetime import datetime
from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Set Path
BASE_DIR = Path(__file__).resolve().parents[3]
data_path = BASE_DIR / "data" / "test_simulations.csv"
model_dir = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "models_3weeks"
model_dir.mkdir(parents=True, exist_ok=True)

# load data
data = pd.read_csv(data_path)

# remove first two rows
data = data.iloc[2:].reset_index(drop=True)
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data[["week_1", "week_2", "week_3", "PMO"]]
X = data[["week_1", "week_2", "week_3"]].astype(float)
y = data["PMO"].astype(int)

# define models
models = {
    "RF": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "GB": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    
    
}


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train and dave
feature_names = list(X.columns)  
n_weeks = len(feature_names)

for model_name, clf in models.items():
    print(f"\nTraining model on full dataset: {model_name}")
    clf.fit(X_scaled, y)

    stem = f"ML_{n_weeks}weeks_{model_name}"
    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"
    meta_json_path = model_dir / f"{stem}.json"
    meta_jbl_path = model_dir / f"{stem}_meta.pkl"

    # save model and scaler
    joblib.dump(clf, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    
    meta = {
        "model_name": model_name,
        "n_weeks": n_weeks,
        "feature_names": feature_names,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "hyperparams": clf.get_params(),
        "notes": "Trained on test_simulations.csv",
    }
    # write JSON and a joblib copy
    with open(meta_json_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    joblib.dump(meta, meta_jbl_path, compress=3)

    print(f"Saved: {model_path.name}, {scaler_path.name}, {meta_json_path.name}")

print("\nAll models trained and saved.")

# Prediction helper

def predict_pmo(model_name: str, week_1: float, week_2: float, week_3:float, threshold: float = 0.5):
    """
    Load model and scaler for model_name (e.g. "RF") and predict PMO probability + class + label.
    Returns dict with keys: model, probability, PMO (0/1), predicted_label ("major"/"minor")
    """
    stem = f"ML_{n_weeks}weeks_{model_name}"
    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_new = np.array([[float(week_1), float(week_2), float(week_3)]])
    X_new_scaled = scaler.transform(X_new)

    # Predict probability
    proba = model.predict_proba(X_new_scaled)[:, 1][0]
  
    # Determine prediction based on threshold
    pred = int(proba >= threshold)
    pred_label = "major" if pred == 1 else "minor"

    return {
        "model": model_name,
        "probability": float(proba),
        "PMO": pred,
        "predicted_label": pred_label,
    }
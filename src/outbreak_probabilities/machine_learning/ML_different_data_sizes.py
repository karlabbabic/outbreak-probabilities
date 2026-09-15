"""This script trains Random Forest and Gradient Boost models on increasing sizes of data (from 500 to 35000 samples) from a CSV file and saves the models, scalers, and metadata for each size. It also includes a helper function to predict the probability of a major outbreak using the trained models."""

import json
from datetime import datetime
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

FEATURE_NAMES = ["week_1", "week_2", "week_3"]
N_WEEKS = len(FEATURE_NAMES)
DATA_SIZES = [500 * i for i in range(1, 70)]  # up to 35000 samples
DEFAULT_DATA_PATH = "data/test_simulations.csv"
MODEL_DIR = Path(__file__).resolve().parent / "Model_SIM"


def load_training_data(data_path):
    data = pd.read_csv(data_path)
    # remove first two metadata rows and promote the real header
    data = data.iloc[2:].reset_index(drop=True)
    data.columns = data.iloc[0]
    data = data.iloc[1:].reset_index(drop=True)

    data = data[FEATURE_NAMES + ["PMO"]]
    X = data[FEATURE_NAMES].astype(float)
    y = data["PMO"].astype(int)
    return X, y


def train(data_path: str = DEFAULT_DATA_PATH, model_dir: Path = MODEL_DIR, data_sizes=DATA_SIZES):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data(data_path)

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

    # data subsets (train on 500 samples, then those 500+500 more, then the 500+500+500 more, etc)
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
                "n_weeks": N_WEEKS,
                "feature_names": FEATURE_NAMES,
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


def predict_pmo(model_name: str, size: int, week_1: float, week_2: float, week_3: float, threshold: float = 0.5, model_dir: Path = MODEL_DIR):
    """
    Predict the probability of a major outbreak (PMO) given the model name, training size used, and weekly case counts.
    """
    model_dir = Path(model_dir)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train RF/GB PMO classifiers on increasing training-data sizes (500..35000 samples).")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help=f"Input simulations CSV (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory to write trained models (default: package's Model_SIM directory)")
    args = parser.parse_args(argv)

    train(data_path=args.data, model_dir=Path(args.model_dir))


if __name__ == "__main__":
    main()

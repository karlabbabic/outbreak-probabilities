import json
from datetime import datetime
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

N_WEEKS = 5
FEATURE_NAMES = [f"week_{i + 1}" for i in range(N_WEEKS)]
DEFAULT_DATA_PATH = "data/test_simulations.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models_5weeks"


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


def train(data_path: str = DEFAULT_DATA_PATH, model_dir: Path = MODEL_DIR):
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
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for model_name, clf in models.items():
        print(f"\nTraining model on full dataset: {model_name}")
        clf.fit(X_scaled, y)

        stem = f"ML_{N_WEEKS}weeks_{model_name}"
        model_path = model_dir / f"{stem}.pkl"
        scaler_path = model_dir / f"{stem}_scaler.pkl"
        meta_json_path = model_dir / f"{stem}.json"
        meta_jbl_path = model_dir / f"{stem}_meta.pkl"

        joblib.dump(clf, model_path, compress=3)
        joblib.dump(scaler, scaler_path, compress=3)

        meta = {
            "model_name": model_name,
            "n_weeks": N_WEEKS,
            "feature_names": FEATURE_NAMES,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "hyperparams": clf.get_params(),
            "notes": "Trained on test_simulations.csv",
        }
        with open(meta_json_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        joblib.dump(meta, meta_jbl_path, compress=3)

        print(f"Saved: {model_path.name}, {scaler_path.name}, {meta_json_path.name}")

    print("\nAll models trained and saved.")


def predict_pmo(model_name: str, week_1: float, week_2: float, week_3: float, week_4: float, week_5: float, threshold: float = 0.5, model_dir: Path = MODEL_DIR):
    """
    Load model and scaler for model_name (e.g. "RF") and predict PMO probability + class + label.
    Returns dict with keys: model, probability, PMO (0/1), predicted_label ("major"/"minor")
    """
    model_dir = Path(model_dir)
    stem = f"ML_{N_WEEKS}weeks_{model_name}"
    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_new = np.array([[float(week_1), float(week_2), float(week_3), float(week_4), float(week_5)]])
    X_new_scaled = scaler.transform(X_new)

    proba = model.predict_proba(X_new_scaled)[:, 1][0]

    pred = int(proba >= threshold)
    pred_label = "major" if pred == 1 else "minor"

    return {
        "model": model_name,
        "probability": float(proba),
        "PMO": pred,
        "predicted_label": pred_label,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train RF PMO classifier on 5-week-ahead simulated data.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help=f"Input simulations CSV (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory to write trained models (default: package's models_5weeks directory)")
    args = parser.parse_args(argv)

    train(data_path=args.data, model_dir=Path(args.model_dir))


if __name__ == "__main__":
    main()

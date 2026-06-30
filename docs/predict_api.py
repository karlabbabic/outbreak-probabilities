from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"


def get_model_paths(n_weeks, model_name):
    model_dir = MODEL_ROOT / f"models_{n_weeks}weeks"
    stem = f"ML_{n_weeks}weeks_{model_name}"

    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"

    return model_path, scaler_path


def predict_pmo(model_name, week_values, threshold=0.5):
    n_weeks = len(week_values)

    model_path, scaler_path = get_model_paths(n_weeks, model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_new = np.array([week_values], dtype=float)
    X_scaled = scaler.transform(X_new)

    probability = model.predict_proba(X_scaled)[:, 1][0]

    prediction = int(probability >= threshold)
    label = "major" if prediction == 1 else "minor"

    return {
        "model": model_name,
        "n_weeks": n_weeks,
        "week_values": week_values,
        "probability": float(probability),
        "PMO": prediction,
        "predicted_label": label,
    }


@app.route("/")
def home():
    return send_file(BASE_DIR / "ML.html", mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({
        "message": "PMO prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        model_name = data.get("model", "RF")
        threshold = float(data.get("threshold", 0.5))
        week_values = data.get("weeks")

        if week_values is None:
            return jsonify({
                "error": "Missing weeks. Send values like: {'weeks': [12, 15]}"
            }), 400

        week_values = [float(value) for value in week_values]

        if len(week_values) < 1 or len(week_values) > 5:
            return jsonify({
                "error": "Please enter between 1 and 5 weekly values."
            }), 400

        result = predict_pmo(
            model_name=model_name,
            week_values=week_values,
            threshold=threshold
        )

        return jsonify(result)

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
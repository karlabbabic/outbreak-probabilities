#!/usr/bin/env python3
"""
Package-level PMO CLI using importlib.resources for packaged model assets.

Run examples:
  PYTHONPATH=src python -m outbreak_probabilities.predict --list
  PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 2 --model RF --week 2.1 --week 1.4
  PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 3 --model RF --batch inputs.csv --out preds.csv
"""

import argparse
import json
import joblib
import sys
from importlib import resources
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any


BASE_PACKAGE = "outbreak_probabilities.machine_learning"

def get_ml_pkg():
    """Return the importlib.resources Traversable for the base machine_learning package."""
    try:
        return resources.files(BASE_PACKAGE)
    except Exception as exc:
        raise RuntimeError(f"Could not access package {BASE_PACKAGE}: {exc}")

def iter_model_folders():
    """Yield Traversables for subfolders named models_*weeks inside the package."""
    root = get_ml_pkg()
    for entry in root.iterdir():
        # Only directories with the naming convention models_*weeks
        if entry.is_dir() and entry.name.startswith("models_") and "weeks" in entry.name:
            yield entry


def discover_models() -> List[Dict[str, Any]]:
    """
    Discover model files inside the package folders.
    Returns list of dicts with keys: stem, folder (Traversable), meta (dict)
    """
    discovered = []
    for folder in iter_model_folders():
        # list model pkl files (skip scalers)
        for item in folder.iterdir():
            if item.is_file() and item.name.endswith(".pkl") and not item.name.endswith("_scaler.pkl") and not item.name.endswith("_meta.pkl"):
                stem = item.stem  # e.g. ML_2weeks_RF
                meta = None
                meta_json = folder.joinpath(f"{stem}.json")
                meta_jbl = folder.joinpath(f"{stem}_meta.pkl")
                # prefer JSON metadata if present
                try:
                    if meta_json.exists():
                        with meta_json.open("r") as fh:
                            meta = json.load(fh)
                    elif meta_jbl.exists():
                        meta = joblib.load(meta_jbl)
                except Exception:
                    meta = None
                # fallback inference
                if meta is None:
                    parts = stem.split("_")
                    model_type = parts[-1] if len(parts) >= 3 else stem
                    n_weeks = None
                    feature_names = []
                    if len(parts) >= 3:
                        token = parts[1]  # e.g. "2weeks"
                        digits = "".join([c for c in token if c.isdigit()])
                        if digits:
                            n_weeks = int(digits)
                            feature_names = [f"week_{i+1}" for i in range(n_weeks)]
                    meta = {"model_name": model_type, "n_weeks": n_weeks, "feature_names": feature_names}
                discovered.append({"stem": stem, "folder": folder, "meta": meta})
    # sort by folder name then stem
    discovered.sort(key=lambda x: (x["folder"].name, x["stem"]))
    return discovered


def load_model_and_scaler_from_folder(folder, stem: str):
    """
    folder: Traversable (resources.files(...)/subfolder)
    stem: e.g. 'ML_2weeks_RF'
    Returns (model, scaler)
    """
    model_res = folder.joinpath(f"{stem}.pkl")
    scaler_res = folder.joinpath(f"{stem}_scaler.pkl")
    if not model_res.exists():
        raise FileNotFoundError(f"Model file not found in package folder: {model_res}")
    if not scaler_res.exists():
        raise FileNotFoundError(f"Scaler file not found in package folder: {scaler_res}")

    # resources.as_file gives a real path (context manager) compatible with joblib.load
    with resources.as_file(model_res) as model_path, resources.as_file(scaler_res) as scaler_path:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return model, scaler

def predict_proba(model, scaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xs)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(Xs)
        if scores.ndim > 1:
            scores = scores[:, 0]
        return 1.0 / (1.0 + np.exp(-scores))
    raise RuntimeError("Model lacks predict_proba and decision_function")


def main(argv=None):
    parser = argparse.ArgumentParser(description="predict (package-level) PMO CLI using importlib.resources")
    parser.add_argument("--list", action="store_true", help="List discovered models and metadata")
    parser.add_argument("--weeks", type=int, help="Number of weeks (e.g. 2, 3, 4, 5)")
    parser.add_argument("--model", type=str, default="RF", help="Model short name, e.g. RF, GB, SVM")
    parser.add_argument("--week", action="append", type=float, help="Provide one week value at a time in order: --week 1.2 --week 0.9 ...")
    parser.add_argument("--batch", type=str, help="CSV path for batch prediction (must contain required feature columns)")
    parser.add_argument("--out", type=str, help="Output CSV path for batch predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default=0.5)")
    parser.add_argument("--json", action="store_true", help="Output single prediction as JSON")
    args = parser.parse_args(argv)

    models = discover_models()
    if args.list:
        if not models:
            print("No packaged models found under package:", BASE_PACKAGE)
            return
        print("Discovered models:")
        for m in models:
            meta = m["meta"] or {}
            print(f"{meta.get('model_name')} | {meta.get('n_weeks')} weeks | {meta.get('feature_names')} | {m['folder'].name} | {m['stem']}")
        return

    if not models:
        print("No models found. Ensure model folders exist inside the package and are included in package data.", file=sys.stderr)
        sys.exit(2)

    # pick candidate models matching model name and weeks
    candidates = [m for m in models if (m["meta"].get("model_name","").lower() == args.model.lower()) and (m["meta"].get("n_weeks") == args.weeks)]
    if not candidates:
        print("No matching model found for --model and --weeks. Use --list to inspect available models.", file=sys.stderr)
        sys.exit(2)

    # pick most recently modified candidate
    chosen = sorted(candidates, key=lambda c: c["folder"].joinpath(f"{c['stem']}.pkl").stat().st_mtime, reverse=True)[0]
    stem = chosen["stem"]
    folder = chosen["folder"]
    meta = chosen["meta"]
    feature_names = meta.get("feature_names") or []

    try:
        model, scaler = load_model_and_scaler_from_folder(folder, stem)
    except Exception as e:
        print("Failed to load model/scaler from package:", e, file=sys.stderr)
        sys.exit(2)

    # single prediction
    if args.week and not args.batch:
        if len(args.week) != len(feature_names):
            print(f"Model expects {len(feature_names)} features: {feature_names}. You provided {len(args.week)}", file=sys.stderr)
            sys.exit(2)
        X = np.array([args.week], dtype=float)
        proba = predict_proba(model, scaler, X)[0]
        pred = int(proba >= args.threshold)
        out = {"model_stem": stem, "features": dict(zip(feature_names, args.week)), "probability": float(proba), "PMO": pred, "predicted_label": "major" if pred else "minor", "threshold": args.threshold}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"Model: {stem} (folder: {folder.name})")
            for k, v in out["features"].items():
                print(f"  {k}: {v}")
            print(f"PMO probability: {out['probability']:.6f}")
            print(f"Predicted class (threshold={out['threshold']}): {out['PMO']} ({out['predicted_label']})")
        return

    # batch prediction
    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print("Batch file not found:", batch_path, file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(batch_path)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            print(f"Batch CSV missing required columns for model {stem}: {missing}", file=sys.stderr)
            sys.exit(2)
        X = df[feature_names].astype(float).to_numpy()
        proba = predict_proba(model, scaler, X)
        df_out = df.copy()
        df_out["pmo_probability"] = proba
        df_out["pmo_predicted_class"] = (proba >= args.threshold).astype(int)
        df_out["pmo_predicted_label"] = df_out["pmo_predicted_class"].map({1: "major", 0: "minor"})
        if args.out:
            df_out.to_csv(args.out, index=False)
            print("Saved predictions to:", args.out)
        else:
            print(df_out.head().to_string(index=False))
        return

    print("No prediction mode selected. Use --week (repeat) for single prediction or --batch for CSV. Use --list to view available models.")
    sys.exit(1)

if __name__ == "__main__":
    main()

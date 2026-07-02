import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# DATA PREPROCESSING 

def clean_simulation_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw dirty dataframe, fixes headers, drops index offset rows,
    and returns a cleaned dataframe with target and feature columns.
    """
    df = raw_df.copy()
    expected_columns = ["week_1", "week_2", "week_3", "week_4", "week_5", "PMO"]

    # If the dataframe already has the expected columns, use it directly.
    if all(col in df.columns for col in expected_columns):
        return df.loc[:, expected_columns].dropna().reset_index(drop=True)

    # Otherwise, handle raw CSV-style input with metadata rows.
    if len(df) >= 3:
        df = df.iloc[2:].reset_index(drop=True)

        if len(df) > 0:
            header_row = df.iloc[0]
            if header_row.notna().all():
                df.columns = header_row
                df = df.iloc[1:].reset_index(drop=True)

    if all(col in df.columns for col in expected_columns):
        return df.loc[:, expected_columns].dropna().reset_index(drop=True)

    # Fallback for simple dataframes with positional columns.
    if len(df.columns) >= 3:
        df.columns = expected_columns
        return df.loc[:, expected_columns].dropna().reset_index(drop=True)

    raise KeyError(f"Expected columns {expected_columns} were not found in the input data.")


# 2. INFERENCE / PREDICTION HELPERS


def load_pipeline(model_dir: Path, model_name: str, n_weeks: int) -> Pipeline:
    """Loads and returns a unified preprocessing + ML model pipeline."""
    stem = f"ML_{n_weeks}weeks_{model_name}"
    pipeline_path = model_dir / f"{stem}_pipeline.pkl"

    if pipeline_path.exists():
        return joblib.load(pipeline_path)

    model_path = model_dir / f"{stem}.pkl"
    scaler_path = model_dir / f"{stem}_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return Pipeline([("scaler", scaler), ("classifier", model)])

    raise FileNotFoundError(f"Missing model pipeline artifact: {pipeline_path}")


def predict_pmo(pipeline: Pipeline, model_name: str, week_1: float, week_2: float, week_3: float, week_4: float, week_5: float, threshold: float = 0.5) -> dict:
    """
    Predicts PMO probability purely using an in-memory pipeline object.
    
    Passing a DataFrame instead of a raw numpy array fixes the 'UserWarning'
    about feature names.
    """
    # Create DataFrame to preserve feature names for the StandardScaler inside the pipeline
    X_new = pd.DataFrame([[float(week_1), float(week_2), float(week_3), float(week_4), float(week_5)]], columns=["week_1", "week_2", "week_3", "week_4", "week_5"])
    
    # The pipeline automatically scales the data and passes it to the model
    proba = pipeline.predict_proba(X_new)[:, 1][0]
    
    pred = int(proba >= threshold)
    pred_label = "major" if pred == 1 else "minor"

    return {
        "model": model_name,
        "probability": float(proba),
        "PMO": pred,
        "predicted_label": pred_label,
    }


# 3. TRAINING PIPELINE EXECUTION

def run_training_pipeline(data_path: Path, model_dir: Path):
    """Encapsulates the structural preprocessing, training, and artifact saving workflow."""
    print(f"Loading raw data from: {data_path}")
    raw_data = pd.read_csv(data_path)
    
    # Process structural cleaning
    cleaned_data = clean_simulation_data(raw_data)
    
    X = cleaned_data[["week_1", "week_2", "week_3", "week_4", "week_5"]].astype(float)
    y = cleaned_data["PMO"].astype(int)
    
    feature_names = list(X.columns)  
    n_weeks = len(feature_names)
    
    # Define models
    models = {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, clf in models.items():
        print(f"\nTraining unified pipeline for: {model_name}")
        
        # Combine Statistical Preprocessing (Scaler) and Model into one workflow object
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        # Fits both scaler parameters and model weights securely together
        pipeline.fit(X, y)

        # File paths
        stem = f"ML_{n_weeks}weeks_{model_name}"
        pipeline_path = model_dir / f"{stem}_pipeline.pkl"
        meta_json_path = model_dir / f"{stem}.json"

        # Save unified pipeline (combines model & scaler into 1 file)
        joblib.dump(pipeline, pipeline_path, compress=3)

        # Metadata tracking
        meta = {
            "model_name": model_name,
            "n_weeks": n_weeks,
            "feature_names": feature_names,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "hyperparams": clf.get_params(),
            "notes": "Trained with structured preprocessing pipeline on test_simulations.csv",
        }
        
        with open(meta_json_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"Saved: {pipeline_path.name} and {meta_json_path.name}")

    print("\nAll models trained and pipelines archived successfully.")


# 4. ENTRYPOINT


if __name__ == "__main__":
    # Resolve Paths locally at script execution runtime
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PATH = BASE_DIR / "data" / "test_simulations.csv"
    MODEL_DIR = BASE_DIR / "src" / "outbreak_probabilities" / "machine_learning" / "models_5weeks"

    # Execute the training run
    run_training_pipeline(DATA_PATH, MODEL_DIR)

    # Quick end-to-end inference verification check:
    print("\n--- Verifying Saved Artifact Inference ---")
    try:
        loaded_rf_pipeline = load_pipeline(MODEL_DIR, "RF", n_weeks=5)
        prediction_result = predict_pmo(loaded_rf_pipeline, "RF", week_1=1, week_2=0, week_3=1, week_4=0, week_5=1)
        print("Verification Result:", prediction_result)
    except Exception as e:
        print(f"Verification shortcut skipped or failed: {e}")
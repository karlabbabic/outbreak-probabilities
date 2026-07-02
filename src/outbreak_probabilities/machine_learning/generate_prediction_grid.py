from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from outbreak_probabilities.analytic.analytical_refractor import compute_pmo_from_string
from outbreak_probabilities.machine_learning.ML_2weeks import load_pipeline as load_pipeline_2w
from outbreak_probabilities.machine_learning.ML_2weeks import predict_pmo as predict_pmo_2w
from outbreak_probabilities.machine_learning.ML_2weeks import run_training_pipeline as run_training_pipeline_2w
from outbreak_probabilities.machine_learning.ML_3weeks import load_pipeline as load_pipeline_3w
from outbreak_probabilities.machine_learning.ML_3weeks import predict_pmo as predict_pmo_3w
from outbreak_probabilities.machine_learning.ML_3weeks import run_training_pipeline as run_training_pipeline_3w
from outbreak_probabilities.machine_learning.ML_4weeks import load_pipeline as load_pipeline_4w
from outbreak_probabilities.machine_learning.ML_4weeks import predict_pmo as predict_pmo_4w
from outbreak_probabilities.machine_learning.ML_4weeks import run_training_pipeline as run_training_pipeline_4w
from outbreak_probabilities.machine_learning.ML_5weeks import load_pipeline as load_pipeline_5w
from outbreak_probabilities.machine_learning.ML_5weeks import predict_pmo as predict_pmo_5w
from outbreak_probabilities.machine_learning.ML_5weeks import run_training_pipeline as run_training_pipeline_5w


MODEL_NAME = "RF"
MAX_WEEK_VALUE = 10
DEFAULT_MODEL_NAMES = ("RF", "GB")


def build_input_grid(n_weeks: int, max_value: int = MAX_WEEK_VALUE) -> list[tuple[int, ...]]:
    """Create all input tuples for the requested weeks horizon."""
    if n_weeks < 2:
        raise ValueError("n_weeks must be at least 2")

    remaining = [range(max_value + 1)] * (n_weeks - 1)
    return [tuple([1, *values]) for values in itertools.product(*remaining)]


def get_predictor(n_weeks: int):
    """Return the relevant ML prediction function and loader for the requested horizon."""
    if n_weeks == 2:
        return load_pipeline_2w, predict_pmo_2w, run_training_pipeline_2w
    if n_weeks == 3:
        return load_pipeline_3w, predict_pmo_3w, run_training_pipeline_3w
    if n_weeks == 4:
        return load_pipeline_4w, predict_pmo_4w, run_training_pipeline_4w
    if n_weeks == 5:
        return load_pipeline_5w, predict_pmo_5w, run_training_pipeline_5w
    raise ValueError(f"Unsupported number of weeks: {n_weeks}")


def generate_results(output_path: Path, max_value: int = MAX_WEEK_VALUE, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Generate a CSV with analytical and ML predictions across all requested horizons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_weeks in [2, 3, 4]:
        model_dir = REPO_ROOT / "src" / "outbreak_probabilities" / "machine_learning" / f"models_{n_weeks}weeks"
        load_pipeline, predict_pmo, run_training = get_predictor(n_weeks)

        pipelines = {}
        for model_name_value in DEFAULT_MODEL_NAMES:
            pipeline_path = model_dir / f"ML_{n_weeks}weeks_{model_name_value}_pipeline.pkl"
            model_path = model_dir / f"ML_{n_weeks}weeks_{model_name_value}.pkl"
            scaler_path = model_dir / f"ML_{n_weeks}weeks_{model_name_value}_scaler.pkl"

            if pipeline_path.exists():
                pipelines[model_name_value] = load_pipeline(model_dir, model_name_value, n_weeks)
            elif model_path.exists() and scaler_path.exists():
                import joblib
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                from sklearn.pipeline import Pipeline
                pipelines[model_name_value] = Pipeline([("scaler", scaler), ("classifier", model)])
            else:
                raise FileNotFoundError(f"No saved model artifact found for {n_weeks} weeks: expected either {pipeline_path} or {model_path}/{scaler_path}")

        for values in build_input_grid(n_weeks, max_value=max_value):
            week_values = [str(v) for v in values]
            initial_cases_string = ",".join(week_values)
            analytic_result = compute_pmo_from_string(initial_cases_string)

            row = {
                "n_weeks": n_weeks,
                "initial_cases": initial_cases_string,
                "analytical_pmo": float(analytic_result["PMO"]),
            }
            for model_name_value in DEFAULT_MODEL_NAMES:
                ml_result = predict_pmo(
                    pipelines[model_name_value],
                    model_name_value,
                    *values,
                )
                row[f"{model_name_value.lower()}_probability"] = float(ml_result["probability"])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analytical-vs-ML prediction comparison data.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "src" / "outbreak_probabilities" / "machine_learning" / "prediction_grid_comparison.csv")
    parser.add_argument("--max-value", type=int, default=MAX_WEEK_VALUE)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    df = generate_results(args.output, max_value=args.max_value, model_name=args.model_name)
    print(f"Saved {len(df)} rows to {args.output}")

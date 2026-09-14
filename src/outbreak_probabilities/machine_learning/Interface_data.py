from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


# Paths

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Imports

from outbreak_probabilities.analytic.analytical_refractor import (
    compute_pmo_from_string,
)

# Keep the standalone ML training scripts untouched. This wrapper loads the
# saved RF artifacts directly and does not depend on helper functions that are
# only defined in those separate training files.


def load_pipeline_2w(model_dir: Path, model_name: str, n_weeks: int):
    """Load the saved RF pipeline from the 2-week model directory."""
    model_path = model_dir / f"ML_{n_weeks}weeks_{model_name}.pkl"
    scaler_path = model_dir / f"ML_{n_weeks}weeks_{model_name}_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing saved model artifacts for {model_name} in {model_dir}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    from sklearn.pipeline import Pipeline
    return Pipeline([("scaler", scaler), ("classifier", model)])


def predict_pmo_2w(model_name: str, *args, **kwargs):
    raise NotImplementedError("Use the dedicated training script to generate predictions for 2-week RF models.")


def run_training_pipeline_2w(*args, **kwargs):
    raise NotImplementedError("Training is intentionally kept in the separate ML scripts.")


def load_pipeline_3w(model_dir: Path, model_name: str, n_weeks: int):
    """Load the saved RF pipeline from the 3-week model directory."""
    model_path = model_dir / f"ML_{n_weeks}weeks_{model_name}.pkl"
    scaler_path = model_dir / f"ML_{n_weeks}weeks_{model_name}_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing saved model artifacts for {model_name} in {model_dir}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    from sklearn.pipeline import Pipeline
    return Pipeline([("scaler", scaler), ("classifier", model)])


def predict_pmo_3w(model_name: str, *args, **kwargs):
    raise NotImplementedError("Use the dedicated training script to generate predictions for 3-week RF models.")


def run_training_pipeline_3w(*args, **kwargs):
    raise NotImplementedError("Training is intentionally kept in the separate ML scripts.")


# Configuration

MODEL_NAME = "RF"

DEFAULT_MODEL_NAMES = ("RF",)

MAX_WEEK_VALUE = 10


# Number of simulations that we RECORD / use for trajectory matching.

N_SIMULATIONS = 35_000


# Number of simulations used to TRAIN the ML models.

# RF and GB should use the full 1M simulations from test_simulations.
ML_TRAINING_SIZE = 1_000_000


# The simulation dataset can still retain this name.
TEST_SIMULATIONS_NAME = "test_simulations"


# Build input grid
def build_input_grid(
    n_weeks: int,
    max_value: int = MAX_WEEK_VALUE,
) -> list[tuple[int, ...]]:
    """
    Create all possible input tuples for the requested horizon.

    The first week is always fixed at 1.

    Example for 3 weeks:

        (1, 0, 0)
        (1, 0, 1)
        (1, 0, 2)
        ...
        (1, 10, 10)
    """

    if n_weeks < 2:
        raise ValueError("n_weeks must be at least 2")

    remaining = [range(max_value + 1)] * (n_weeks - 1)

    return [
        tuple([1, *values])
        for values in itertools.product(*remaining)
    ]


# Select appropriate ML functions

def get_predictor(n_weeks: int):
    """
    Return the relevant ML loader, prediction function,
    and training function.
    """

    if n_weeks == 2:
        return (
            load_pipeline_2w,
            predict_pmo_2w,
            run_training_pipeline_2w,
        )

    if n_weeks == 3:
        return (
            load_pipeline_3w,
            predict_pmo_3w,
            run_training_pipeline_3w,
        )

    raise ValueError(
        f"Unsupported number of weeks: {n_weeks}. "
        "Only 2 and 3 weeks are supported."
    )


# ====================================================================
# Load saved RF and GB models

def load_models(n_weeks: int):
    """
    Load RF models for the requested week horizon.

    Supports either:

        *_pipeline.pkl

    or the older:

        *.pkl
        *_scaler.pkl
    """

    model_dir = (
        REPO_ROOT
        / "src"
        / "outbreak_probabilities"
        / "machine_learning"
        / f"models_{n_weeks}weeks"
    )

    pipelines = {}

    for model_name in DEFAULT_MODEL_NAMES:
        pipeline_path = model_dir / f"ML_{n_weeks}weeks_{model_name}_pipeline.pkl"
        model_path = model_dir / f"ML_{n_weeks}weeks_{model_name}.pkl"
        scaler_path = model_dir / f"ML_{n_weeks}weeks_{model_name}_scaler.pkl"

        if pipeline_path.exists():
            pipelines[model_name] = load_pipeline_2w(model_dir, model_name, n_weeks) if n_weeks == 2 else load_pipeline_3w(model_dir, model_name, n_weeks)

        elif model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            from sklearn.pipeline import Pipeline
            pipelines[model_name] = Pipeline([
                ("scaler", scaler),
                ("classifier", model),
            ])
        else:
            raise FileNotFoundError(
                f"No saved model artifact found for {n_weeks} weeks.\n"
                f"Expected either:\n"
                f"  {pipeline_path}\n"
                f"or:\n"
                f"  {model_path}\n"
                f"  {scaler_path}"
            )

    return pipelines


# Generate JSON results

def generate_results(
    output_path: Path,
    max_value: int = MAX_WEEK_VALUE,
) -> dict:
    """
    Generate analytical-vs-ML comparison data for 2 and 3 weeks.

    IMPORTANT:

    N_SIMULATIONS = 35,000
        Number of simulations being recorded / used for trajectory
        matching.

    ML_TRAINING_SIZE = 1,000,000
        Number of simulations used to train RF and GB.

    The ML models loaded by this script should therefore have been
    trained on the full 1M-row test_simulations dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSON metadata

    output = {
        "metadata": {

            # Keep this at 35k
            "n_simulations": N_SIMULATIONS,

            # ML trained using all 1M
            "ml_training_size": ML_TRAINING_SIZE,

            # Maximum horizon is now 3 weeks
            "ml_features": [
                "week_1",
                "week_2",
                "week_3",
            ],

            "R_range": [
                0.0,
                10.0,
            ],

            "major_threshold": 100,

            "simulation_file": TEST_SIMULATIONS_NAME,

            "generated": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },

        "results": {},
    }


    
    # ONLY generate 2-week and 3-week results

    for n_weeks in [2, 3]:

        print(
            f"Generating {n_weeks}-week predictions..."
        )

        sys.stdout.flush()


        # Load RF only

        pipelines = load_models(n_weeks)


        # Generate all input combinations

        grid = build_input_grid(
            n_weeks,
            max_value=max_value,
        )

        feature_columns = [
            f"week_{i + 1}"
            for i in range(n_weeks)
        ]

        df_input = pd.DataFrame(
            grid,
            columns=feature_columns,
        ).astype(float)


       
        df_input["initial_cases_string"] = (
            df_input[feature_columns]
            .astype(int)
            .astype(str)
            .agg(",".join, axis=1)
        )


        # Analytical PMO

        df_input["analytic"] = [
            float(
                compute_pmo_from_string(
                    case_string
                )["PMO"]
            )
            for case_string
            in df_input["initial_cases_string"]
        ]


        # Random Forest predictions

        rf_probability = (
            pipelines["RF"]
            .predict_proba(
                df_input[feature_columns]
            )[:, 1]
        )

        df_input["ml_rf"] = (
            rf_probability.astype(float)
        )


        # Convert dataframe into JSON entries

        for _, row in df_input.iterrows():

            initial_cases = [
                int(row[column])
                for column in feature_columns
            ]

          

            cases_key = "_".join(
                str(value)
                for value in initial_cases
            )

            result_key = (
                f"{n_weeks}w_{cases_key}"
            )


            # Store result

            output["results"][result_key] = {

                "n_weeks": n_weeks,

                "initial_cases": initial_cases,

                "analytic": round(
                    float(row["analytic"]),
                    6,
                ),

                "ml_rf": round(
                    float(row["ml_rf"]),
                    6,
                ),
            }


    # =================================================================
    # Save JSON
    # =================================================================

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            allow_nan=False,
        )


    return output


# =====================================================================
# Command-line entry point
# =====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate analytical-vs-ML prediction comparison "
            "JSON for 2- and 3-week horizons."
        )
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "src"
            / "outbreak_probabilities"
            / "machine_learning"
            / "prediction_grid_comparison.json"
        ),
    )


    parser.add_argument(
        "--max-value",
        type=int,
        default=MAX_WEEK_VALUE,
    )


    args = parser.parse_args()


    results = generate_results(
        args.output,
        max_value=args.max_value,
    )


    print(
        f"Saved {len(results['results'])} results "
        f"to {args.output}"
    )

    print(
        f"Recorded simulations: {N_SIMULATIONS:,}"
    )

    print(
        f"ML training simulations: {ML_TRAINING_SIZE:,}"
    )
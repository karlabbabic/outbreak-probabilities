import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

from metrics_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

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
    "SVM": SVC(
        probability=True,
        random_state=42,
    ),
}

# set paths
BASE_DIR = Path(__file__).resolve().parents[3]
data_path = BASE_DIR / "data" / "simulated_cases_and_serial_interval_and_weights1.csv"
model_dir = BASE_DIR /"src"/"outbreak_probabilities"/ "machine_learning"/ "models_4weeks"
plots_dir = BASE_DIR / "src"/"outbreak_probabilities"/"machine_learning"/ "model_outputs_4weeks" / "plots"

model_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

#  Load data
data = pd.read_csv(data_path)

# remove first two rows
data = data.iloc[1:].reset_index(drop=True)
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data[["week_1", "week_2", "week_3", "week_4", "PMO"]]

X = data[["week_1", "week_2", "week_3", "week_4"]].astype(float)
y = data["PMO"].astype(int)

# train and evaluate models
y_pred_dict = {}
y_proba_dict = {}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for model_name, clf in models.items():
    print(f"\nTraining model: {model_name}")

    clf.fit(X_train_scaled, y_train)

    calibrated_clf = CalibratedClassifierCV(
        clf,
        method="isotonic",
        cv=5,
    )
    calibrated_clf.fit(X_train_scaled, y_train)

    y_pred = calibrated_clf.predict(X_test_scaled)
    y_proba = calibrated_clf.predict_proba(X_test_scaled)[:, 1]

    print("Sample calibrated probabilities:", y_proba[:10])
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # save model + scaler
    joblib.dump(
        calibrated_clf,
        model_dir / f"ML_4weeks_{model_name}_calibrated.pkl",
        compress=3,
    )
    joblib.dump(
        scaler,
        model_dir / f"ML_4weeks_{model_name}_scaler.pkl",
        compress=3,
    )

    # store results for plotting
    y_pred_dict[model_name] = y_pred
    y_proba_dict[model_name] = y_proba




# PLOT
for model_name in models.keys():
    plot_confusion_matrix(
        y_test,
        y_pred_dict[model_name],
        model_name,
        output_dir=plots_dir,
    )

    plot_roc_curve(
        y_test,
        y_proba_dict[model_name],
        model_name,
        output_dir=plots_dir,
    )

    plot_precision_recall_curve(
        y_test,
        y_proba_dict[model_name],
        model_name,
        output_dir=plots_dir,
    )
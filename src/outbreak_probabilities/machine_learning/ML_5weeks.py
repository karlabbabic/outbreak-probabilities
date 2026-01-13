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
model_dir = BASE_DIR /"src"/"outbreak_probabilities"/ "machine_learning"/ "models_5weeks"
plots_dir = BASE_DIR / "src"/"outbreak_probabilities"/"machine_learning"/ "model_outputs_5weeks" / "plots"

model_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

#  Load data
data = pd.read_csv(data_path)

# remove first two rows
data = data.iloc[1:].reset_index(drop=True)
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data[["week_1", "week_2", "PMO"]]

X = data[["week_1", "week_2"]].astype(float)
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


    # save model + scaler
    joblib.dump(
        calibrated_clf,
        model_dir / f"ML_5weeks_{model_name}_calibrated.pkl",
        compress=3,
    )
    joblib.dump(
        scaler,
        model_dir / f"ML_5weeks_{model_name}_scaler.pkl",
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
    
    
# TEST THE MODEL on 2 weeks data (2 and 1)

def predict_pmo(model_name, week_1, week_2, threshold=0.5):
    model_path = model_dir / f"ML_5weeks_{model_name}_calibrated.pkl"
    scaler_path = model_dir / f"ML_5weeks_{model_name}_scaler.pkl"

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
    # add major or minor in the return if pred == 1 else minor
    if pred == 1:
        pred_label = "major"
    else:
        pred_label = "minor"
    
    return {
        "model": model_name,
        "probability": float(proba),
        "PMO": pred,
        "predicted_label": pred_label
    }
# Example usage
result = predict_pmo("GB", week_1=2, week_2=1)
print(result)
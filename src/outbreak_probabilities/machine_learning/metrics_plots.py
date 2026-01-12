import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from pathlib import Path

# Paths



def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap=plt.cm.Blues
    )
    disp.ax_.set_title(f"Confusion Matrix - {model_name}")
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png")
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name, output_dir):
    disp = RocCurveDisplay.from_predictions(y_true, y_proba)
    disp.ax_.set_title(f"ROC Curve - {model_name}")
    plt.savefig(output_dir / f"roc_curve_{model_name}.png")
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, model_name, output_dir):
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    disp.ax_.set_title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(output_dir / f"precision_recall_curve_{model_name}.png")
    plt.close()



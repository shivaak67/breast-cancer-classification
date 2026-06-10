from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def load_artifacts():
    """Load saved model and scaler."""
    model = joblib.load(MODELS_DIR / "logistic_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def evaluate_model(model, X_test, y_test, target_names):
    """Return predictions and evaluation metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def save_confusion_matrix_plot(cm, save_path):
    """Save confusion matrix heatmap to outputs/figures/."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_report(report, accuracy, save_path):
    """Save text metrics to outputs/results/."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)


if __name__ == "__main__":
    data = preprocess_data()
    model, _ = load_artifacts()

    results = evaluate_model(
        model,
        data["X_test_scaled"],
        data["y_test"],
        data["target_names"],
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_plot(
        results["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix.png",
    )
    save_metrics_report(
        results["classification_report"],
        results["accuracy"],
        RESULTS_DIR / "metrics.txt",
    )

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(results["classification_report"])
    print("Saved confusion matrix to outputs/figures/confusion_matrix.png")
    print("Saved metrics to outputs/results/metrics.txt")

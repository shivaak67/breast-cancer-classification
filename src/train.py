from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"


def train_model(X_train, y_train):
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def save_artifacts(model, scaler):
    """Save trained model and scaler to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "logistic_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")


if __name__ == "__main__":
    data = preprocess_data()

    model = train_model(data["X_train_scaled"], data["y_train"])
    save_artifacts(model, data["scaler"])

    print("Model trained and saved to models/saved_models/")

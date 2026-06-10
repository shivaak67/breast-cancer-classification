from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data():
    """Load the Wisconsin Breast Cancer dataset as a DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.target_names


def preprocess_data(test_size=0.2, random_state=42):
    """Split features/target and return scaled train/test sets."""
    df, target_names = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "target_names": target_names,
        "feature_names": list(X.columns),
    }


if __name__ == "__main__":
    data = preprocess_data()

    print("Training set:", data["X_train_scaled"].shape)
    print("Test set:", data["X_test_scaled"].shape)
    print("Preprocessing complete.")

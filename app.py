from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

TOP_FEATURES = [
    "worst concave points",
    "worst perimeter",
    "mean concave points",
    "worst radius",
    "mean radius",
]
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "logistic_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    defaults = pd.Series(
        pd.DataFrame(data.data, columns=feature_names).mean(),
        index=feature_names,
    )
    target_names = list(data.target_names)
    return model, scaler, feature_names, defaults, target_names
model, scaler, feature_names, defaults, target_names = load_artifacts()
st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🎗️")
st.title("Breast Cancer Classification")
st.write(
    "Predict whether a tumor is **malignant** or **benign** "
    "using a logistic regression model trained on the Wisconsin Breast Cancer dataset."
)
st.sidebar.header("Example cases")
if st.sidebar.button("Load benign example"):
  st.session_state["example"] = 1
if st.sidebar.button("Load malignant example"):
  st.session_state["example"] = 0
data = load_breast_cancer()
if "example" in st.session_state:
    idx = np.where(data.target == st.session_state["example"])[0][0]
    example_values = pd.Series(data.data[idx], index=feature_names)
else:
    example_values = defaults
st.subheader("Key measurements")
inputs = {}
for feature in TOP_FEATURES:
    inputs[feature] = st.number_input(
        feature,
        value=float(example_values[feature]),
        format="%.4f",
    )
with st.expander("All other features (defaults used)"):
    for feature in feature_names:
        if feature not in TOP_FEATURES:
            inputs[feature] = st.number_input(
                feature,
                value=float(example_values[feature]),
                format="%.4f",
            )
if st.button("Predict"):
    feature_vector = np.array([[inputs[f] for f in feature_names]])
    scaled = scaler.transform(feature_vector)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    label = target_names[prediction]
    confidence = probability[prediction] * 100
    if prediction == 0:
        st.error(f"Prediction: **{label.title()}** ({confidence:.1f}% confidence)")
    else:
        st.success(f"Prediction: **{label.title()}** ({confidence:.1f}% confidence)")
    st.write("Probability breakdown:")
    st.progress(float(probability[1]))
    st.caption(f"Benign: {probability[1]*100:.1f}% | Malignant: {probability[0]*100:.1f}%")
st.caption(
    "For educational purposes only. Not for clinical use."
)
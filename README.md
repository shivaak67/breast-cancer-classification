# Breast Cancer Classification

End-to-end machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Breast Cancer dataset. The workflow covers exploratory data analysis, preprocessing, model training, evaluation, and an interactive Streamlit demo.

**Repository:** [github.com/shivaak67/breast-cancer-classification](https://github.com/shivaak67/breast-cancer-classification)

**Live demo:** [placeholder — add Streamlit Cloud URL after deploy]

---

## Problem

Given numeric measurements from digitized images of breast mass cell nuclei, predict whether a tumor is malignant or benign. This is a binary classification task with real-world relevance in medical diagnostics.

## Dataset

- **Source:** [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) via scikit-learn
- **Samples:** 569
- **Features:** 30 numeric measurements (mean, standard error, and worst values for radius, texture, perimeter, area, etc.)
- **Target:** `0` = malignant, `1` = benign
- **Class balance:** ~63% benign, ~37% malignant

---

## Results

| Model | Test accuracy |
|-------|---------------|
| Logistic Regression | **98.25%** |
| Random Forest | 95.61% |

**Logistic Regression — test set metrics:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Malignant | 0.98 | 0.98 | 0.98 | 42 |
| Benign | 0.99 | 0.99 | 0.99 | 72 |

Confusion matrix (114 test samples): 41 malignant correct, 71 benign correct, 2 total errors.

---

## Screenshots

### Exploratory data analysis

[placeholder — EDA correlation heatmap or class distribution plot]

### Model evaluation

[placeholder — confusion matrix heatmap from `outputs/figures/confusion_matrix.png`]

### Streamlit app

[placeholder — Streamlit app home screen]

[placeholder — Streamlit prediction result]

---

## Project structure

```
breast-cancer-ml/
├── app.py                      # Streamlit demo
├── notebooks/
│   ├── 1_eda.ipynb             # Exploratory data analysis
│   ├── 2_preprocessing.ipynb   # Train/test split and scaling
│   └── 3_modeling.ipynb        # Model training and evaluation
├── src/
│   ├── preprocess.py           # Load data, split, scale
│   ├── train.py                # Train and save model
│   └── evaluate.py             # Evaluate and save metrics/plots
├── models/saved_models/        # Trained model and scaler (.pkl)
├── outputs/
│   ├── figures/                # Saved plots
│   └── results/                # Saved metrics
├── data/
│   ├── raw/                    # Reserved for raw data files
│   └── processed/              # Reserved for processed data files
├── requirements.txt
└── README.md
```

---

## Tech stack

- **Python**
- **pandas** — data manipulation
- **scikit-learn** — preprocessing, modeling, metrics
- **matplotlib & seaborn** — visualizations
- **Streamlit** — interactive web app
- **joblib** — model persistence

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/shivaak67/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline (scripts)

From the project root:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

This trains a logistic regression model, saves artifacts to `models/saved_models/`, and writes evaluation outputs to `outputs/`.

### 5. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### 6. Run the notebooks (optional)

Open and run in order:

1. `notebooks/1_eda.ipynb`
2. `notebooks/2_preprocessing.ipynb`
3. `notebooks/3_modeling.ipynb`

---

## Approach

1. **EDA** — Inspected shape, missing values, class balance, feature correlations, and feature distributions by diagnosis.
2. **Preprocessing** — Stratified 80/20 train/test split; `StandardScaler` fit on training data only.
3. **Modeling** — Logistic regression (primary) and random forest (comparison).
4. **Evaluation** — Accuracy, confusion matrix, and classification report on the held-out test set.
5. **Deployment** — Streamlit app loads the saved model and scaler for interactive predictions.

---

## Disclaimer

This project is for **educational and portfolio purposes only**. It is not intended for clinical use or medical decision-making.

---

## Author

Shivaa Karthikgavaskar — [GitHub](https://github.com/shivaak67)

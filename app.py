from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Sleep Debt Classification", layout="wide")
st.title("Sleep Debt and Screen Time — Model Evaluation")
st.write(
    "Upload the **held-out test CSV**, including the target column "
    "`sleep_debt_category`. Models are loaded from saved files; "
    "**no retraining occurs**."
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
TARGET = "sleep_debt_category"


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])
selected = st.selectbox("Choose model for detailed evaluation", list(MODELS))

if uploaded is None:
    st.info("Upload the held-out test_data.csv from your repository to evaluate all five saved models.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if df.empty:
    st.error("Uploaded CSV is empty.")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Test CSV must contain the target column: {TARGET}")
    st.stop()
if df[TARGET].isna().any():
    st.error("The target column contains missing labels.")
    st.stop()

X_test = df.drop(columns=[TARGET, "user_id"], errors="ignore")
y_test = df[TARGET]
results = []
details = {}

for name, filename in MODELS.items():
    try:
        model = load_model(filename)
        predictions = model.predict(X_test)
        labels = model.classes_
        probabilities = model.predict_proba(X_test)

        auc = np.nan
        if set(y_test.unique()) == set(labels):
            try:
                auc = roc_auc_score(
                    y_test,
                    probabilities,
                    labels=labels,
                    multi_class="ovr",
                    average="weighted",
                )
            except ValueError:
                pass

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": auc,
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        })
        details[name] = (predictions, labels)
    except Exception as exc:
        st.error(f"Failed to evaluate {name}: {exc}")
        st.stop()

st.subheader("Comparison: all five saved classifiers")
scores = pd.DataFrame(results).set_index("Model")
st.dataframe(scores.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
st.caption(
    "AUC: multiclass weighted One-vs-Rest; Precision/Recall/F1: weighted. "
    "N/A means AUC cannot be calculated for the uploaded class distribution."
)

st.subheader(f"Detailed results: {selected}")
row = scores.loc[selected]
cols = st.columns(3)
for i, metric in enumerate(["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = row[metric]
    cols[i % 3].metric(metric, f"{value:.4f}" if pd.notna(value) else "N/A")

predictions, labels = details[selected]
st.subheader("Confusion Matrix")
normalize = st.toggle("Show row-normalized percentages", value=False)

counts = confusion_matrix(y_test, predictions, labels=labels)
if normalize:
    row_totals = counts.sum(axis=1, keepdims=True)
    display_matrix = np.divide(
        counts.astype(float), row_totals,
        out=np.zeros_like(counts, dtype=float), where=row_totals != 0
    ) * 100
    annotations = np.array([[f"{v:.1f}%" for v in row] for row in display_matrix])
    colorbar_label = "Percentage of actual class"
else:
    display_matrix = counts
    annotations = counts
    colorbar_label = "Number of observations"

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    display_matrix,
    annot=annotations,
    fmt="" if normalize else "d",
    cmap="BuPu",
    linewidths=0.8,
    linecolor="white",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": colorbar_label},
    ax=ax,
    annot_kws={"size": 11},
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title(f"Confusion Matrix — {selected}", fontsize=14, pad=14)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if any(counts.sum(axis=1) == 0):
    st.caption("A class with no actual test observations has an all-zero row; its recall cannot be assessed from this upload.")

st.subheader("Classification Report")
report = classification_report(
    y_test, predictions, labels=labels, output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

output = df.copy()
output["predicted_sleep_debt_category"] = predictions
st.download_button(
    "Download predictions",
    output.to_csv(index=False).encode("utf-8"),
    file_name="sleep_debt_predictions.csv",
    mime="text/csv",
)

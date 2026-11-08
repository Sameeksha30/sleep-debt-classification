import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Sleep Debt Classifier", page_icon="😴", layout="wide")
st.title("Sleep Debt Category Prediction")
st.write("Upload a CSV containing the sleep and screen-time input features. "
         "The app predicts No Debt, Mild, Moderate, or Severe.")

MODEL_DIR = Path(__file__).parent / "model"
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
choice = st.selectbox("Choose a classification model", list(model_files))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.caption("The user_id and sleep_debt_category columns are not used as model inputs. "
           "If a target column is included, it is retained only for optional comparison.")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    target = "sleep_debt_category"
    features = data.drop(columns=[target, "user_id"], errors="ignore")

    model_path = MODEL_DIR / model_files[choice]
    model = joblib.load(model_path)

    try:
        predictions = model.predict(features)
        output = data.copy()
        output["predicted_sleep_debt_category"] = predictions

        if target in data.columns:
            output["prediction_correct"] = (
                output[target].astype(str) == output["predicted_sleep_debt_category"].astype(str)
            )
            st.metric("Accuracy on uploaded labelled rows",
                      f"{output['prediction_correct'].mean():.2%}")

        st.subheader("Predictions")
        st.dataframe(output, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="sleep_debt_predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
        st.info("Check that the CSV contains the same feature columns used during training.")
else:
    st.info("Try the included test_data.csv file to see example predictions.")

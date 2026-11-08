import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title='Sleep Debt Classification', layout='wide')
st.title('Sleep Debt and Screen Time — Model Evaluation')
st.write('Upload the **held-out test CSV**, including the target column `sleep_debt_category`. Models are loaded from saved files; no retraining occurs.')

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODELS = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Gaussian Naive Bayes': 'gaussian_naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}
TARGET = 'sleep_debt_category'

@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)

uploaded = st.file_uploader('Upload test_data.csv', type=['csv'])
selected = st.selectbox('Choose model for detailed evaluation', list(MODELS))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if df.empty:
        st.error('Uploaded CSV is empty.')
        st.stop()
    if TARGET not in df.columns:
        st.error(f'Test CSV must contain the target column: {TARGET}')
        st.stop()
    if df[TARGET].isna().any():
        st.error('The target column contains missing labels.')
        st.stop()
    X_test = df.drop(columns=[TARGET, 'user_id'], errors='ignore')
    y_test = df[TARGET]
    results = []
    details = {}
    for name, filename in MODELS.items():
        try:
            model = load_model(filename)
            y_pred = model.predict(X_test)
            labels = model.classes_
            y_prob = model.predict_proba(X_test)
            # Multiclass AUC requires at least two classes and may be undefined
            # if some trained classes are absent from the uploaded test sample.
            try:
                if set(y_test.unique()) == set(labels):
                    auc = roc_auc_score(y_test, y_prob, labels=labels,
                                        multi_class='ovr', average='weighted')
                else:
                    auc = float('nan')
            except ValueError:
                auc = float('nan')
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC': auc,
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'MCC': matthews_corrcoef(y_test, y_pred),
            })
            details[name] = (y_pred, labels)
        except Exception as exc:
            st.error(f'Failed to evaluate {name}: {exc}')
            st.stop()

    st.subheader('Comparison: all five saved classifiers')
    scores = pd.DataFrame(results).set_index('Model')
    st.dataframe(scores.style.format('{:.4f}', na_rep='N/A'), use_container_width=True)
    st.caption('AUC: multiclass weighted One-vs-Rest; Precision/Recall/F1: weighted. N/A means AUC is undefined for the uploaded class distribution.')

    st.subheader(f'Detailed results: {selected}')
    row = scores.loc[selected]
    cols = st.columns(3)
    for i, metric in enumerate(['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']):
        val = row[metric]
        cols[i % 3].metric(metric, f'{val:.4f}' if pd.notna(val) else 'N/A')

    y_pred, labels = details[selected]
    st.markdown('**Confusion matrix** (rows = actual, columns = predicted)')
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(matrix, index=[f'Actual: {x}' for x in labels],
                         columns=[f'Predicted: {x}' for x in labels])
    st.dataframe(cm_df, use_container_width=True)
    st.markdown('**Classification report**')
    report = classification_report(y_test, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    predictions = df.copy()
    predictions['predicted_sleep_debt_category'] = y_pred
    st.download_button('Download predictions', predictions.to_csv(index=False).encode('utf-8'),
                       file_name='sleep_debt_predictions.csv', mime='text/csv')
else:
    st.info('Upload the test_data.csv from your repository to evaluate all five saved models.')

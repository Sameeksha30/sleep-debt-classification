import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
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
    st.caption('AUC: multiclass weighted One-vs-Rest; Precision/Recall/F1: weighted. N/A means AUC cannot be calculated for the uploaded class distribution.')

    st.subheader(f'Detailed results: {selected}')
    row = scores.loc[selected]
    cols = st.columns(3)
    for i, metric in enumerate(['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']):
        val = row[metric]
        cols[i % 3].metric(metric, f'{val:.4f}' if pd.notna(val) else 'N/A')

    y_pred, labels = details[selected]
    st.markdown('### Error diagnostics')
    left, right = st.columns([1.12, 1], gap='medium')
    with left:
        st.markdown('**Confusion matrix**')
        show_percentages = st.toggle('Show row-normalized percentages', value=False)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        if show_percentages:
            row_totals = cm.sum(axis=1, keepdims=True)
            plot_values = np.divide(cm.astype(float), row_totals,
                                    out=np.zeros_like(cm, dtype=float),
                                    where=row_totals != 0) * 100
            annotations = np.array([[f'{v:.1f}%' for v in r] for r in plot_values])
            vmax = 100
        else:
            plot_values = cm
            annotations = True
            vmax = None

        fig, ax = plt.subplots(figsize=(5.3, 3.5), dpi=110)
        fig.patch.set_facecolor('white')
        sns.heatmap(plot_values, annot=annotations, fmt='' if show_percentages else 'd',
                    cmap='BuPu', xticklabels=labels, yticklabels=labels,
                    linewidths=0.6, linecolor='white', cbar=False,
                    vmin=0, vmax=vmax, annot_kws={'size': 10}, ax=ax,
                    square=False)
        ax.set_xlabel('Predicted label', fontsize=9, labelpad=5)
        ax.set_ylabel('Actual label', fontsize=9, labelpad=5)
        ax.tick_params(axis='both', labelsize=8, length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown('**Class-wise performance**')
        report = classification_report(y_test, y_pred, labels=labels,
                                       output_dict=True, zero_division=0)
        class_rows = []
        for label in labels:
            stats = report[str(label)]
            class_rows.append({'Class': str(label), 'Precision': stats['precision'],
                               'Recall': stats['recall'], 'F1-score': stats['f1-score'],
                               'Support': int(stats['support'])})
        for label, key in [('Macro avg', 'macro avg'), ('Weighted avg', 'weighted avg')]:
            stats = report[key]
            class_rows.append({'Class': label, 'Precision': stats['precision'],
                               'Recall': stats['recall'], 'F1-score': stats['f1-score'],
                               'Support': int(stats['support'])})
        report_df = pd.DataFrame(class_rows).set_index('Class')
        st.dataframe(report_df.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}',
                                             'F1-score': '{:.3f}', 'Support': '{:d}'}),
                     use_container_width=True, hide_index=False)
        st.caption('Support = number of actual test samples in each class. Overall accuracy is shown above.')

    predictions = df.copy()
    predictions['predicted_sleep_debt_category'] = y_pred
    st.download_button('Download predictions', predictions.to_csv(index=False).encode('utf-8'),
                       file_name='sleep_debt_predictions.csv', mime='text/csv')
else:
    st.info('Upload the test_data.csv from your repository to evaluate all five saved models.')

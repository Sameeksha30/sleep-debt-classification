# Sleep Debt and Screen Time – Multi-class Classification

A Streamlit application and reproducible training pipeline for predicting `sleep_debt_category` using five machine-learning classifiers.

## Models included
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM, RBF kernel)

## Repository structure

```text
sleep_debt_github_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    ├── svm.joblib
    └── evaluation_metrics.csv
```

Each `.joblib` file stores the complete scikit-learn Pipeline, including preprocessing and the trained classifier. The preprocessing artifacts therefore do not need to be saved separately.

## Dataset

- Dataset used: Sleep Debt and Screen Time.
- Rows: 8,500.
- Columns in original file: 23.
- Prediction target: `sleep_debt_category` (four classes).
- `user_id` is excluded because it is an identifier, not a predictive feature.
- The original Excel dataset is not included in this repository. Add `sleep_debt_screen_time.csv.xlsx` to the project root before retraining.

**Dataset citation:** Add the original Kaggle or UCI dataset URL, dataset author, and license here if applicable. Do not claim a repository source unless verified.

## Evaluation

The models were evaluated on a stratified 80:20 train/test split (`random_state=42`). AUC is multiclass One-vs-Rest with weighted averaging. Precision, recall, and F1 use weighted averaging; MCC uses the multiclass implementation.

| Model | Accuracy | AUC (OvR weighted) | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8124 | 0.9332 | 0.8132 | 0.8124 | 0.8114 | 0.6911 |\n| decision_tree | 0.9994 | 0.9996 | 0.9994 | 0.9994 | 0.9994 | 0.9990 |\n| knn | 0.6647 | 0.7695 | 0.6760 | 0.6647 | 0.6524 | 0.4281 |\n| gaussian_naive_bayes | 0.5359 | 0.7424 | 0.6143 | 0.5359 | 0.5642 | 0.3125 |\n| svm | 0.8306 | 0.9341 | 0.8297 | 0.8306 | 0.8286 | 0.7203 |\n
## Run the application locally

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Launch Streamlit:

   ```bash
   streamlit run app.py
   ```

6. Upload `test_data.csv` or another CSV with the same predictor columns.

## Retrain the models

1. Put the original Excel dataset named `sleep_debt_screen_time.csv.xlsx` in the project root.
2. Run:

   ```bash
   python train_model.py
   ```

3. The five trained pipelines and `evaluation_metrics.csv` will be regenerated in `model/`.

## Important modelling limitation

The `Severe` class is rare (25 observations in the full dataset). In addition, `total_sleep_hours` may be closely related to how the target category was defined. Check for target leakage before interpreting near-perfect Decision Tree performance as evidence of real-world generalization. Consider class-wise precision/recall, confusion matrices, cross-validation, and a sensitivity analysis excluding any feature used to derive the target.

## Notes

- Do not upload private or sensitive data to a public repository.
- Include the dataset only if its license permits redistribution.
- The saved models were trained using scikit-learn; loading joblib files from untrusted sources is unsafe.

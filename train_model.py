"""Train five sleep-debt classifiers and save complete preprocessing/model pipelines."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

DATA_PATH = Path("sleep_debt_screen_time.csv.xlsx")  # Place original dataset here
OUT = Path("model")
OUT.mkdir(exist_ok=True)
TARGET = "sleep_debt_category"
ID_COLUMN = "user_id"

df = pd.read_excel(DATA_PATH)
X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y = df[TARGET]

categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object", "bool"]).columns.tolist()

def build_preprocessor(scale):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numerical),
        ("cat", categorical_pipe, categorical)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

classifiers = {
    "logistic_regression": (LogisticRegression(max_iter=3000), True),
    "decision_tree": (DecisionTreeClassifier(random_state=42), False),
    "knn": (KNeighborsClassifier(n_neighbors=5), True),
    "gaussian_naive_bayes": (GaussianNB(), False),
    "svm": (SVC(kernel="rbf", probability=True, random_state=42), True),
}
results = []
for name, (classifier, scale) in classifiers.items():
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(scale)),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "auc_ovr_weighted": roc_auc_score(
            y_test, probabilities, multi_class="ovr",
            average="weighted", labels=pipeline.classes_
        ),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_test, predictions)
    })
    joblib.dump(pipeline, OUT / f"{name}.joblib")

pd.DataFrame(results).to_csv(OUT / "evaluation_metrics.csv", index=False)
print(pd.DataFrame(results).round(4).to_string(index=False))

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "real"
    / "step11_cost_ml_training_ready.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "FutureCostOverrun"

ID_COLUMNS = [
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]

LEAKAGE_COLUMNS = [
    "FutureCostOverrun",
    "FutureTimeOverrun",
    "OverallRiskEvent",
]
# Features that should NEVER be used as predictors.
EXCLUDED_COLUMNS = [
    TARGET,
    *ID_COLUMNS,
    *LEAKAGE_COLUMNS,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PRISM — COST OVERRUN MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.drop_duplicates().copy()

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = df[TARGET].astype(int)


# ============================================================
# REMOVE LEAKAGE / IDENTIFIERS
# ============================================================

features = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


# ============================================================
# TARGET CHECK
# ============================================================

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTarget percentage:")

print(
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
)


# ============================================================
# PROJECT-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating project-level train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        df[TARGET],
        groups=df["ProjectCode"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print(f"Training rows: {len(train_df)}")
print(f"Testing rows : {len(test_df)}")

print(
    f"Training projects: "
    f"{train_df['ProjectCode'].nunique()}"
)

print(
    f"Testing projects : "
    f"{test_df['ProjectCode'].nunique()}"
)

# ============================================================
# ROBUST FEATURE TYPE DETECTION
# ============================================================

categorical_features = []
numeric_features = []

for column in features:

    converted = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # If every non-null value can be converted to a number,
    # treat the feature as numeric.
    if converted.notna().sum() == df[column].notna().sum():

        numeric_features.append(column)

    else:

        categorical_features.append(column)


print("\nCategorical features:")
for column in categorical_features:
    print(f"  - {column}")

print("\nNumeric features:")
for column in numeric_features:
    print(f"  - {column}")

print(
    f"\nNumeric feature count: "
    f"{len(numeric_features)}"
)

print(
    f"Categorical feature count: "
    f"{len(categorical_features)}"
)


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric feature count:")
print(len(numeric_features))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


X_train = train_df[features]
y_train = train_df[TARGET]

X_test = test_df[features]
y_test = test_df[TARGET]


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 — LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        ),
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_prob = (
    logistic_model
    .predict_proba(X_test)[:, 1]
)

logistic_pred = (
    logistic_prob >= 0.50
).astype(int)


logistic_roc = roc_auc_score(
    y_test,
    logistic_prob
)

logistic_pr = average_precision_score(
    y_test,
    logistic_prob
)


print(
    f"\nROC-AUC: {logistic_roc:.4f}"
)

print(
    f"PR-AUC : {logistic_pr:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_pred,
        digits=4
    )
)


# ============================================================
# MODEL 2 — XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 — XGBOOST")
print("=" * 70)


xgb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.80,
                colsample_bytree=0.80,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        ),
    ]
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_prob = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)

xgb_pred = (
    xgb_prob >= 0.50
).astype(int)


xgb_roc = roc_auc_score(
    y_test,
    xgb_prob
)

xgb_pr = average_precision_score(
    y_test,
    xgb_prob
)


print(
    f"\nROC-AUC: {xgb_roc:.4f}"
)

print(
    f"PR-AUC : {xgb_pr:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_pred,
        digits=4
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'ROC-AUC':<15}"
    f"{'PR-AUC':<15}"
)

print(
    f"{'Logistic Regression':<25}"
    f"{logistic_roc:<15.4f}"
    f"{logistic_pr:<15.4f}"
)

print(
    f"{'XGBoost':<25}"
    f"{xgb_roc:<15.4f}"
    f"{xgb_pr:<15.4f}"
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if xgb_pr >= logistic_pr:

    best_model = xgb_model
    best_name = "XGBoost"
    best_roc = xgb_roc
    best_pr = xgb_pr

else:

    best_model = logistic_model
    best_name = "LogisticRegression"
    best_roc = logistic_roc
    best_pr = logistic_pr


print(
    f"\nSelected model: {best_name}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "cost_overrun_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model": best_name,

    "target": TARGET,

    "features": features,

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "validation": {
        "method": "GroupShuffleSplit by ProjectCode",
        "test_size": 0.20,
        "random_state": 42,
    },

    "metrics": {
        "roc_auc": float(best_roc),
        "pr_auc": float(best_pr),
    },

    "training_rows": len(train_df),

    "testing_rows": len(test_df),

    "training_projects":
        int(train_df["ProjectCode"].nunique()),

    "testing_projects":
        int(test_df["ProjectCode"].nunique()),
}


metadata_path = (
    MODEL_DIR
    / "cost_overrun_metadata.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print("\nSaved:")

print(model_path)

print(metadata_path)

print("\n" + "=" * 70)
print("COST MODEL TRAINING COMPLETE")
print("=" * 70)
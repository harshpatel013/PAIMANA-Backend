import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "real"
    / "step20_overall_risk_ml_training_ready.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "overall_risk_model.joblib"
)

METADATA_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "overall_risk_metadata.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PRISM — OVERALL RISK MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "OverallRiskEvent"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

y = df[TARGET].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True))


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = [
    "State",
    "Agency",
]

NUMERIC_FEATURES = [
    "OriginalCost",
    "RevisedCost",
    "Expenditure",
    "Physical Progress",
    "DateInconsistency",
    "ProjectAgeDays",
    "ApprovalToStartDays",
    "CurrentExpenditureRatio",
    "RobustCurrentExpenditurePct",
    "RobustCurrentGap",
    "HasRevisedCost",
    "HasRevisedDoc",
    "ProgressVelocity",
    "ProgressScheduleGap",
    "ExpenditureProgressGap",
    "CostGrowth",
    "ScheduleElapsedRatio",
    "RemainingPlannedDays",
    "ElapsedDays",
    "PlannedDurationDays",
    "ProgressBehindSchedule10",
    "ProgressBehindSchedule20",
    "ProgressBehindSchedule30",
    "NegativeProgressVelocity",
    "NoProgress",
    "LateStageLowProgress",
    "ExpenditureAhead10",
    "ExpenditureAhead20",
    "ExpenditureAhead30",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )

X = df[FEATURES].copy()


# ============================================================
# PROJECT-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating project-level train/test split...")

if "ProjectCode" not in df.columns:
    raise ValueError("ProjectCode is required for project-level splitting.")

project_codes = df["ProjectCode"].astype(str)

unique_projects = project_codes.unique()

rng = np.random.RandomState(42)
rng.shuffle(unique_projects)

split_index = int(len(unique_projects) * 0.80)

train_projects = set(unique_projects[:split_index])
test_projects = set(unique_projects[split_index:])

train_mask = project_codes.isin(train_projects)
test_mask = project_codes.isin(test_projects)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")

print(f"Training projects: {len(train_projects)}")
print(f"Testing projects : {len(test_projects)}")


# ============================================================
# PREPROCESSING
# ============================================================

print("\nCategorical features:")
for feature in CATEGORICAL_FEATURES:
    print(f"  - {feature}")

print("\nNumeric feature count:")
print(len(NUMERIC_FEATURES))


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ]
)


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
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


logistic_model.fit(
    X_train,
    y_train,
)

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

logistic_pred = (
    logistic_prob >= 0.50
).astype(int)


logistic_roc = roc_auc_score(
    y_test,
    logistic_prob,
)

logistic_pr = average_precision_score(
    y_test,
    logistic_prob,
)


print(f"\nROC-AUC: {logistic_roc:.4f}")
print(f"PR-AUC : {logistic_pr:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_pred,
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_pred,
        zero_division=0,
    )
)


# ============================================================
# MODEL 2 — XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 — XGBOOST")
print("=" * 70)


# Calculate imbalance ratio
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = (
    negative_count / positive_count
)

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


xgb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


xgb_model.fit(
    X_train,
    y_train,
)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_pred = (
    xgb_prob >= 0.50
).astype(int)


xgb_roc = roc_auc_score(
    y_test,
    xgb_prob,
)

xgb_pr = average_precision_score(
    y_test,
    xgb_prob,
)


print(f"\nROC-AUC: {xgb_roc:.4f}")
print(f"PR-AUC : {xgb_pr:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_pred,
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_pred,
        zero_division=0,
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
    selected_model = xgb_model
    selected_name = "XGBoost"
    selected_prob = xgb_prob
    selected_roc = xgb_roc
    selected_pr = xgb_pr
else:
    selected_model = logistic_model
    selected_name = "Logistic Regression"
    selected_prob = logistic_prob
    selected_roc = logistic_roc
    selected_pr = logistic_pr


print(
    f"\nSelected model: {selected_name}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

for threshold in np.arange(
    0.10,
    0.61,
    0.05,
):

    predictions = (
        selected_prob >= threshold
    ).astype(int)

    precision, recall, _ = precision_recall_curve(
        y_test,
        selected_prob,
    )

    pred_precision = (
        (y_test[predictions == 1] == 1).mean()
        if np.sum(predictions == 1) > 0
        else 0
    )

    pred_recall = (
        (predictions[y_test == 1] == 1).mean()
        if np.sum(y_test == 1) > 0
        else 0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    alert_rate = predictions.mean()

    threshold_results.append(
        {
            "threshold": round(
                float(threshold),
                2,
            ),
            "precision": round(
                float(pred_precision),
                4,
            ),
            "recall": round(
                float(pred_recall),
                4,
            ),
            "f1": round(
                float(f1),
                4,
            ),
            "alert_rate": round(
                float(alert_rate),
                4,
            ),
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

print(threshold_df.to_string(index=False))


# ============================================================
# SELECT EARLY WARNING THRESHOLD
# ============================================================

best_f1_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_f1_threshold = float(
    best_f1_row["threshold"]
)


# Prefer a threshold with recall >= 0.60
early_warning_candidates = threshold_df[
    threshold_df["recall"] >= 0.60
]

if len(early_warning_candidates) > 0:

    early_warning_row = (
        early_warning_candidates
        .sort_values(
            ["f1", "threshold"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    early_warning_threshold = float(
        early_warning_row["threshold"]
    )

else:

    early_warning_threshold = (
        best_f1_threshold
    )


print(
    f"\nBest F1 threshold: "
    f"{best_f1_threshold:.2f}"
)

print(
    f"Early-warning threshold: "
    f"{early_warning_threshold:.2f}"
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_file = (
    BASE_DIR
    / "ml"
    / "models"
    / "risk_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False,
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    selected_model,
    MODEL_FILE,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model_name": selected_name,
    "target": TARGET,
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_projects": int(len(train_projects)),
    "testing_projects": int(len(test_projects)),
    "roc_auc": round(
        float(selected_roc),
        6,
    ),
    "pr_auc": round(
        float(selected_pr),
        6,
    ),
    "best_f1_threshold": best_f1_threshold,
    "early_warning_threshold": early_warning_threshold,
    "random_state": 42,
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nSaved:")

print(MODEL_FILE)

print(METADATA_FILE)

print(threshold_file)

print("\n" + "=" * 70)
print("OVERALL RISK MODEL TRAINING COMPLETE")
print("=" * 70)
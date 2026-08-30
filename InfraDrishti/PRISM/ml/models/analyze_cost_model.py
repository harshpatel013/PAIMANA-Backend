from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


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

MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "cost_overrun_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET = "FutureCostOverrun"

ID_COLUMNS = [
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PRISM — COST MODEL ANALYSIS")
print("=" * 70)

print("\nLoading model...")

pipeline = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

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
# SAME PROJECT-LEVEL TEST SPLIT
# ============================================================

print("\nRecreating project-level test split...")

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

test_df = df.iloc[test_idx].copy()

print(
    f"Test rows: {len(test_df)}"
)

print(
    f"Test projects: "
    f"{test_df['ProjectCode'].nunique()}"
)


# ============================================================
# GET FEATURES FROM SAVED PIPELINE
# ============================================================

features = [
    column
    for column in df.columns
    if column not in [
        TARGET,
        *ID_COLUMNS,
        "FutureTimeOverrun",
        "OverallRiskEvent",
    ]
]

X_test = test_df[features]

y_test = test_df[TARGET]


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

print("\nGenerating predictions...")

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = np.arange(
    0.20,
    0.61,
    0.05
)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": round(
            float(threshold),
            2
        ),
        "precision": round(
            float(precision),
            4
        ),
        "recall": round(
            float(recall),
            4
        ),
        "f1": round(
            float(f1),
            4
        ),
    })

threshold_df = pd.DataFrame(
    results
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT THRESHOLD
# ============================================================

# Select the threshold with the highest F1.
best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

BEST_THRESHOLD = float(
    best_row["threshold"]
)

print(
    f"\nRecommended threshold "
    f"based on F1: {BEST_THRESHOLD:.2f}"
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_file = (
    OUTPUT_DIR
    / "cost_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

print(
    f"\nSaved threshold analysis:"
)

print(threshold_file)


# ============================================================
# SHAP PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("SHAP FEATURE IMPORTANCE")
print("=" * 70)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]


# Transform test data exactly as the model sees it.

X_transformed = preprocessor.transform(
    X_test
)


# ============================================================
# FEATURE NAMES
# ============================================================

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

except Exception:

    feature_names = np.array([
        f"feature_{i}"
        for i in range(
            X_transformed.shape[1]
        )
    ])


# ============================================================
# SHAP
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_transformed
)


# ============================================================
# HANDLE SPARSE MATRIX
# ============================================================

if hasattr(
    X_transformed,
    "toarray"
):

    X_for_shap = (
        X_transformed.toarray()
    )

else:

    X_for_shap = X_transformed


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


print("\nTop 15 features:")

print(
    importance_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = (
    OUTPUT_DIR
    / "cost_shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print(
    f"\nSaved SHAP importance:"
)

print(importance_file)


# ============================================================
# PROJECT-LEVEL EXPLANATIONS
# ============================================================

print("\nGenerating project explanations...")


top_n = 10

explanation_rows = []

for row_position in range(
    min(
        len(test_df),
        100
    )
):

    project = test_df.iloc[
        row_position
    ]

    row_shap = shap_values[
        row_position
    ]

    top_indices = np.argsort(
        np.abs(row_shap)
    )[::-1][:top_n]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        explanation_rows.append({

            "ProjectCode":
                project["ProjectCode"],

            "ProjectName":
                project["ProjectName"],

            "ReportMonth":
                project["ReportMonth"],

            "ActualOutcome":
                int(
                    project[TARGET]
                ),

            "PredictedRisk":
                round(
                    float(
                        probabilities[
                            row_position
                        ]
                    ),
                    4
                ),

            "Feature":
                feature_names[index],

            "SHAPValue":
                round(
                    float(
                        row_shap[index]
                    ),
                    6
                ),

            "Impact":
                (
                    "Increases Risk"
                    if row_shap[index] > 0
                    else "Decreases Risk"
                ),

            "Rank":
                rank,
        })


explanations_df = pd.DataFrame(
    explanation_rows
)


explanation_file = (
    OUTPUT_DIR
    / "cost_project_explanations.csv"
)

explanations_df.to_csv(
    explanation_file,
    index=False
)

print(
    f"\nSaved project explanations:"
)

print(explanation_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COST MODEL ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nBest threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Threshold precision: "
    f"{best_row['precision']:.4f}"
)

print(
    f"Threshold recall: "
    f"{best_row['recall']:.4f}"
)

print(
    f"Threshold F1: "
    f"{best_row['f1']:.4f}"
)

print("\nOutput files:")

print(threshold_file)
print(importance_file)
print(explanation_file)
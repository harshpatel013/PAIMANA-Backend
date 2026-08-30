from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "real"
    / "step15_time_ml_training_ready.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "time_overrun_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

TARGET = "FutureTimeOverrun"

ID_COLUMNS = [
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]

LEAKAGE_COLUMNS = [
    "FutureCostOverrun",
    "FutureTimeOverrun",
    "OverallRiskEvent",
    "HasPreviousTargetDoc",
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PRISM — TIME MODEL EXPLAINABILITY")
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
# RECREATE TEST SPLIT
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
# FEATURES
# ============================================================

features = [
    column
    for column in df.columns
    if column not in [
        TARGET,
        *ID_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]
]

X_test = test_df[features]


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = (
    pipeline.predict_proba(X_test)[:, 1]
)


# ============================================================
# GET PIPELINE COMPONENTS
# ============================================================

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]


# ============================================================
# TRANSFORM FEATURES
# ============================================================

print("\nTransforming features...")

X_transformed = (
    preprocessor.transform(X_test)
)

if hasattr(
    X_transformed,
    "toarray"
):

    X_transformed = (
        X_transformed.toarray()
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
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = (
    np.abs(shap_values)
    .mean(axis=0)
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
# SAVE GLOBAL IMPORTANCE
# ============================================================

importance_file = (
    OUTPUT_DIR
    / "time_shap_feature_importance.csv"
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

print(
    "\nGenerating project-level explanations..."
)

explanation_rows = []

TOP_N = 10

for row_position in range(
    len(test_df)
):

    project = test_df.iloc[
        row_position
    ]

    row_shap = shap_values[
        row_position
    ]

    top_indices = np.argsort(
        np.abs(row_shap)
    )[::-1][:TOP_N]

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

            "PredictedTimeRisk":
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


# ============================================================
# SAVE PROJECT EXPLANATIONS
# ============================================================

explanation_file = (
    OUTPUT_DIR
    / "time_project_explanations.csv"
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
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TIME MODEL EXPLAINABILITY COMPLETE")
print("=" * 70)

print("\nOutput files:")

print(importance_file)

print(explanation_file)
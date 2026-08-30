from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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
print("PRISM — TIME MODEL ANALYSIS")
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
# PROJECT-LEVEL TEST SPLIT
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

y_test = test_df[TARGET]


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating predictions...")

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TIME OVERRUN THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = np.arange(
    0.10,
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

    alert_rate = predictions.mean()

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
        "alert_rate": round(
            float(alert_rate),
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
# BEST F1
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

BEST_THRESHOLD = float(
    best_row["threshold"]
)

print(
    f"\nRecommended threshold "
    f"based on F1: "
    f"{BEST_THRESHOLD:.2f}"
)


# ============================================================
# EARLY WARNING THRESHOLD
# ============================================================

# We also identify a threshold that achieves
# at least 60% recall while keeping the
# highest possible precision.

recall_candidates = threshold_df[
    threshold_df["recall"] >= 0.60
]

if not recall_candidates.empty:

    early_warning_row = (
        recall_candidates
        .sort_values(
            "precision",
            ascending=False
        )
        .iloc[0]
    )

    EARLY_WARNING_THRESHOLD = float(
        early_warning_row["threshold"]
    )

    print(
        f"\nEarly-warning threshold "
        f"(recall >= 60%): "
        f"{EARLY_WARNING_THRESHOLD:.2f}"
    )

    print(
        f"Precision: "
        f"{early_warning_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{early_warning_row['recall']:.4f}"
    )

else:

    EARLY_WARNING_THRESHOLD = None

    print(
        "\nNo threshold achieved "
        "60% recall."
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "time_threshold_analysis.csv"
)

threshold_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved:"
)

print(output_file)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TIME MODEL ANALYSIS COMPLETE")
print("=" * 70)
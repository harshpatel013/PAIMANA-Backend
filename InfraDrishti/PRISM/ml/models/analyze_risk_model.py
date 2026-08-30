from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
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

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

SHAP_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "risk_shap_feature_importance.csv"
)

PROJECT_EXPLANATIONS_FILE = (
    OUTPUT_DIR
    / "risk_project_explanations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Analyze a subset for SHAP to keep runtime reasonable.
MAX_SHAP_ROWS = 1000

# Number of top features retained per project.
TOP_PROJECT_FEATURES = 5


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRISM — OVERALL RISK MODEL EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features"]
categorical_features = metadata["categorical_features"]
numeric_features = metadata["numeric_features"]

print(f"Expected features: {len(features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numeric features: {len(numeric_features)}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Required features are missing:")
    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "Dataset does not contain all features expected by the model."
    )


print("\nFeature validation successful.")


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

X = df[features].copy()

print(f"\nModel input shape: {X.shape}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(X)[:, 1]

df["OverallRiskProbability"] = probabilities

print("Predictions generated.")


# ============================================================
# THRESHOLD
# ============================================================

threshold = metadata.get(
    "best_f1_threshold",
    0.45
)

df["OverallRiskPrediction"] = (
    df["OverallRiskProbability"] >= threshold
).astype(int)

print(
    f"Decision threshold: {threshold:.2f}"
)


# ============================================================
# TRANSFORM FEATURES
# ============================================================

print("\nTransforming features...")

# The saved model is a Pipeline.
# Its first step is expected to be the preprocessing transformer.

if hasattr(model, "named_steps"):

    print(
        "Pipeline steps:",
        list(model.named_steps.keys())
    )

    preprocess = model.named_steps.get(
        "preprocess"
    )

    if preprocess is None:

        # Try to locate a ColumnTransformer automatically.
        preprocess = None

        for name, step in model.named_steps.items():

            if hasattr(
                step,
                "transformers_"
            ):
                preprocess = step
                print(
                    f"Using preprocessing step: {name}"
                )
                break

    if preprocess is None:

        raise ValueError(
            "Could not locate preprocessing transformer."
        )

    X_transformed = preprocess.transform(X)

    # Convert sparse matrix if necessary.
    if hasattr(
        X_transformed,
        "toarray"
    ):
        X_transformed = X_transformed.toarray()

    try:

        feature_names = (
            preprocess
            .get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"feature_{i}"
            for i in range(
                X_transformed.shape[1]
            )
        ]

else:

    raise ValueError(
        "Expected the saved model to be a sklearn Pipeline."
    )


X_transformed = np.asarray(
    X_transformed
)

print(
    f"Transformed shape: {X_transformed.shape}"
)

print(
    f"Transformed feature count: "
    f"{len(feature_names)}"
)


# ============================================================
# SAMPLE DATA FOR SHAP
# ============================================================

if len(X_transformed) > MAX_SHAP_ROWS:

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shap_indices = rng.choice(
        len(X_transformed),
        size=MAX_SHAP_ROWS,
        replace=False
    )

else:

    shap_indices = np.arange(
        len(X_transformed)
    )


X_shap = X_transformed[
    shap_indices
]

print(
    f"\nSHAP rows: {len(X_shap)}"
)


# ============================================================
# SHAP
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model.named_steps[
        list(model.named_steps.keys())[-1]
    ]
)

shap_values = explainer.shap_values(
    X_shap
)

# XGBoost binary classification normally
# returns an array of shape:
# (rows, features)

if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[-1]

shap_values = np.asarray(
    shap_values
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }
)

importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("TOP 15 OVERALL RISK FEATURES")
print("=" * 70)

print(
    importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE GLOBAL SHAP IMPORTANCE
# ============================================================

importance_df.to_csv(
    SHAP_IMPORTANCE_FILE,
    index=False
)

print(
    "\nSaved SHAP importance:"
)

print(
    SHAP_IMPORTANCE_FILE
)


# ============================================================
# PROJECT-LEVEL EXPLANATIONS
# ============================================================

print(
    "\nGenerating project-level explanations..."
)


# Map SHAP sample rows back to original dataframe.
sample_df = df.iloc[
    shap_indices
].copy()

sample_shap = shap_values


project_rows = []


for local_index in range(
    len(sample_df)
):

    row = sample_df.iloc[
        local_index
    ]

    values = sample_shap[
        local_index
    ]

    # Sort by absolute SHAP impact.
    ranked_indices = np.argsort(
        np.abs(values)
    )[::-1]

    explanation = {
        "ProjectCode": row.get(
            "ProjectCode",
            np.nan
        ),

        "ProjectName": row.get(
            "ProjectName",
            ""
        ),

        "ReportMonth": row.get(
            "ReportMonth",
            ""
        ),

        "OverallRiskProbability": round(
            float(
                row["OverallRiskProbability"]
            ),
            6
        ),

        "OverallRiskPrediction": int(
            row["OverallRiskPrediction"]
        ),
    }


    # --------------------------------------------------------
    # Top contributing features
    # --------------------------------------------------------

    for rank in range(
        TOP_PROJECT_FEATURES
    ):

        feature_index = ranked_indices[
            rank
        ]

        feature_name = feature_names[
            feature_index
        ]

        shap_value = float(
            values[
                feature_index
            ]
        )

        # Remove preprocessing prefixes
        # to make explanations easier to read.

        clean_name = feature_name

        if "__" in clean_name:

            clean_name = clean_name.split(
                "__",
                1
            )[1]

        explanation[
            f"TopFactor{rank + 1}"
        ] = clean_name

        explanation[
            f"TopFactor{rank + 1}_SHAP"
        ] = round(
            shap_value,
            6
        )


    # --------------------------------------------------------
    # Human-readable explanation
    # --------------------------------------------------------

    positive_features = []

    for feature_index in ranked_indices:

        shap_value = float(
            values[
                feature_index
            ]
        )

        if shap_value > 0:

            feature_name = feature_names[
                feature_index
            ]

            if "__" in feature_name:

                feature_name = (
                    feature_name
                    .split(
                        "__",
                        1
                    )[1]
                )

            positive_features.append(
                feature_name
            )

        if len(
            positive_features
        ) >= 3:

            break


    explanation[
        "RiskExplanation"
    ] = "; ".join(
        positive_features
    )


    project_rows.append(
        explanation
    )


project_explanations = pd.DataFrame(
    project_rows
)


# ============================================================
# SAVE PROJECT EXPLANATIONS
# ============================================================

project_explanations.to_csv(
    PROJECT_EXPLANATIONS_FILE,
    index=False
)

print(
    "\nSaved project explanations:"
)

print(
    PROJECT_EXPLANATIONS_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RISK EXPLAINABILITY COMPLETE")
print("=" * 70)

print(
    f"\nModel threshold: {threshold:.2f}"
)

print(
    f"Projects analyzed for SHAP: "
    f"{len(project_explanations)}"
)

print(
    "\nOutput files:"
)

print(
    SHAP_IMPORTANCE_FILE
)

print(
    PROJECT_EXPLANATIONS_FILE
)
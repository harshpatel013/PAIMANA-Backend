from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

COST_MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "cost_overrun_model.joblib"
)

TIME_DATA_FILE = (
    BASE_DIR
    / "data"
    / "real"
    / "step15_time_ml_training_ready.csv"
)

TIME_MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "time_overrun_model.joblib"
)

OVERALL_RISK_DATA_FILE = (
    BASE_DIR
    / "data"
    / "real"
    / "step20_overall_risk_ml_training_ready.csv"
)

OVERALL_RISK_MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "models"
    / "overall_risk_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "risk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

COST_THRESHOLD = 0.20
TIME_THRESHOLD = 0.30

OVERALL_RISK_THRESHOLD = 0.45


# Risk weighting
COST_WEIGHT = 0.30
TIME_WEIGHT = 0.30
OVERALL_RISK_WEIGHT = 0.25
INDICATOR_WEIGHT = 0.15
# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# NORMALIZATION
# ============================================================

def clamp(value, minimum=0, maximum=100):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# CURRENT INDICATOR SCORE
# ============================================================

def calculate_indicator_score(row):

    indicators = []


    # --------------------------------------------------------
    # Progress / Schedule Gap
    # --------------------------------------------------------

    gap = row.get(
        "ProgressScheduleGap",
        np.nan
    )

    if pd.notna(gap):

        gap_score = clamp(
            abs(float(gap)) * 2
        )

        indicators.append(
            gap_score
        )


    # --------------------------------------------------------
    # Schedule Delay
    # --------------------------------------------------------

    delay = row.get(
        "CurrentScheduleDelayDays",
        np.nan
    )

    if pd.notna(delay):

        delay_score = clamp(
            float(delay) / 3
        )

        indicators.append(
            delay_score
        )


    # --------------------------------------------------------
    # Cost Growth
    # --------------------------------------------------------

    cost_growth = row.get(
        "CostGrowth",
        np.nan
    )

    if pd.notna(cost_growth):

        cost_score = clamp(
            abs(float(cost_growth)) * 100
        )

        indicators.append(
            cost_score
        )


    # --------------------------------------------------------
    # Expenditure vs Progress
    # --------------------------------------------------------

    expenditure_ratio = row.get(
        "CurrentExpenditureRatio",
        np.nan
    )

    physical_progress = row.get(
        "Physical Progress",
        np.nan
    )

    if (
        pd.notna(expenditure_ratio)
        and pd.notna(physical_progress)
    ):

        expenditure_ratio = float(
            expenditure_ratio
        )

        physical_progress = float(
            physical_progress
        )

        imbalance = (
            expenditure_ratio
            - physical_progress
        )

        imbalance_score = clamp(
            abs(imbalance) * 2
        )

        indicators.append(
            imbalance_score
        )


    if not indicators:

        return 0.0

    return float(
        np.mean(indicators)
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("PRISM — PROJECT RISK ENGINE")
print("=" * 70)

print("\nLoading cost model...")

cost_model = joblib.load(
    COST_MODEL_FILE
)

print("Cost model loaded.")

print("\nLoading time model...")

time_model = joblib.load(
    TIME_MODEL_FILE
)

print("Time model loaded.")

print("\nLoading overall risk model...")

overall_risk_model = joblib.load(
    OVERALL_RISK_MODEL_FILE
)

print("Overall risk model loaded.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading project data...")

cost_df = pd.read_csv(
    DATA_FILE
)

time_df = pd.read_csv(
    TIME_DATA_FILE
)

overall_risk_df = pd.read_csv(
    OVERALL_RISK_DATA_FILE
)

# ============================================================
# NORMALIZE REPORT MONTH
# ============================================================

cost_df["ReportMonth"] = pd.to_datetime(
    cost_df["ReportMonth"],
    errors="coerce"
).dt.to_period("M").astype(str)

time_df["ReportMonth"] = pd.to_datetime(
    time_df["ReportMonth"],
    errors="coerce"
).dt.to_period("M").astype(str)

overall_risk_df["ReportMonth"] = pd.to_datetime(
    overall_risk_df["ReportMonth"],
    errors="coerce"
).dt.to_period("M").astype(str)

# ============================================================
# MERGE COST + TIME DATA
# ============================================================

common_columns = [
    "ProjectCode",
    "ReportMonth",
]

# Columns already available in cost dataset
# do not need to be duplicated from time dataset.

columns_already_present = set(
    cost_df.columns
)

time_additional_columns = [
    column
    for column in time_df.columns
    if (
        column in common_columns
        or column not in columns_already_present
    )
]

time_subset = time_df[
    time_additional_columns
].copy()


df = cost_df.merge(
    time_subset,
    on=common_columns,
    how="inner"
)

overall_columns = [
    "ProjectCode",
    "ReportMonth",
]

for column in overall_risk_df.columns:

    if column not in overall_columns and column not in df.columns:
        overall_columns.append(column)

overall_subset = overall_risk_df[
    overall_columns
].copy()

df = df.merge(
    overall_subset,
    on=[
        "ProjectCode",
        "ReportMonth"
    ],
    how="inner"
)


print(
    f"\nProjects/records available: "
    f"{len(df)}"
)

print(
    f"\nRecords after overall-risk merge: {len(df)}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = (
    df
    .drop_duplicates(
        subset=[
            "ProjectCode",
            "ReportMonth"
        ]
    )
    .copy()
)


print(
    f"\nProjects/records available: "
    f"{len(df)}"
)


# ============================================================
# COST PREDICTIONS
# ============================================================

print("\nGenerating cost predictions...")

cost_excluded = [
    "FutureCostOverrun",
    "FutureTimeOverrun",
    "OverallRiskEvent",
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]

cost_features = [
    column
    for column in cost_df.columns
    if column not in cost_excluded
]

cost_X = df[cost_features]

df["CostOverrunProbability"] = (
    cost_model
    .predict_proba(cost_X)[:, 1]
)


# ============================================================
# TIME PREDICTIONS
# ============================================================

print("Generating time predictions...")

time_excluded = [
    "FutureTimeOverrun",
    "FutureCostOverrun",
    "OverallRiskEvent",
    "HasPreviousTargetDoc",
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]

time_features = [
    column
    for column in time_df.columns
    if column not in time_excluded
]


# Some merged columns may have _time suffixes.
# Construct the time model input carefully.

time_input = pd.DataFrame(
    index=df.index
)

for feature in time_features:

    if feature in df.columns:

        time_input[feature] = df[
            feature
        ]

    elif (
        f"{feature}_time"
        in df.columns
    ):

        time_input[feature] = df[
            f"{feature}_time"
        ]

    else:

        time_input[feature] = np.nan


df["TimeOverrunProbability"] = (
    time_model
    .predict_proba(
        time_input
    )[:, 1]
)

# ============================================================
# OVERALL RISK MODEL PREDICTIONS
# ============================================================

print(
    "Generating overall risk predictions..."
)

overall_excluded = [
    "OverallRiskEvent",
    "FutureCostOverrun",
    "FutureTimeOverrun",
    "ProjectCode",
    "ProjectName",
    "ReportMonth",
]

overall_features = [
    column
    for column in overall_risk_df.columns
    if column not in overall_excluded
]

overall_input = pd.DataFrame(
    index=df.index
)

for feature in overall_features:

    if feature in df.columns:

        overall_input[feature] = df[
            feature
        ]

    else:

        overall_input[feature] = np.nan


df["OverallRiskProbability"] = (
    overall_risk_model
    .predict_proba(
        overall_input
    )[:, 1]
)


# ============================================================
# WARNING FLAGS
# ============================================================

df["CostWarning"] = (
    df["CostOverrunProbability"]
    >= COST_THRESHOLD
)

df["TimeWarning"] = (
    df["TimeOverrunProbability"]
    >= TIME_THRESHOLD
)

df["OverallRiskWarning"] = (
    df["OverallRiskProbability"]
    >= OVERALL_RISK_THRESHOLD
)

# ============================================================
# CURRENT INDICATOR SCORE
# ============================================================

print(
    "\nCalculating current-condition indicators..."
)

df["IndicatorScore"] = df.apply(
    calculate_indicator_score,
    axis=1
)


# ============================================================
# OVERALL RISK SCORE
# ============================================================

df["OverallRiskScore"] = (
    df["CostOverrunProbability"] * 100 * COST_WEIGHT
    + df["TimeOverrunProbability"] * 100 * TIME_WEIGHT
    + df["OverallRiskProbability"] * 100 * OVERALL_RISK_WEIGHT
    + df["IndicatorScore"] * INDICATOR_WEIGHT
)

df["OverallRiskScore"] = (
    df["OverallRiskScore"]
    .clip(0, 100)
)


# ============================================================
# RISK LEVEL
# ============================================================

df["RiskLevel"] = (
    df["OverallRiskScore"]
    .apply(get_risk_level)
)


# ============================================================
# ALERT TYPE
# ============================================================

def get_alert_type(row):

    cost = row["CostWarning"]
    time = row["TimeWarning"]
    overall = row["OverallRiskWarning"]

    if cost and time:
        return "COST_AND_TIME"

    elif cost:
        return "COST_RISK"

    elif time:
        return "TIME_RISK"

    elif overall:
        return "OVERALL_RISK"

    else:
        return "MONITOR"

df["AlertType"] = df.apply(
    get_alert_type,
    axis=1
)


# ============================================================
# RECOMMENDATION
# ============================================================

def recommendation(row):

    alert = row["AlertType"]

    if alert == "COST_AND_TIME":

        return (
            "Immediate multidisciplinary review "
            "recommended. Investigate expenditure "
            "trajectory, schedule slippage and "
            "milestone recovery actions."
        )

    elif alert == "COST_RISK":

        return (
            "Review expenditure growth, revised "
            "cost trajectory and financial controls."
        )

    elif alert == "TIME_RISK":

        return (
            "Review milestone progress, schedule "
            "slippage and recovery plan."
        )

    else:

        return (
            "Continue routine monitoring."
        )


df["RecommendedAction"] = df.apply(
    recommendation,
    axis=1
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "ProjectCode",
    "ProjectName",
    "ReportMonth",

    "State",
    "Agency",

    "OriginalCost",
    "RevisedCost",
    "Expenditure",

    "Physical Progress",

    "CostOverrunProbability",
    "TimeOverrunProbability",

    "CostWarning",
    "TimeWarning",

    "IndicatorScore",

    "OverallRiskScore",
    "RiskLevel",

    "AlertType",
    "RecommendedAction",
]


available_columns = [
    column
    for column in output_columns
    if column in df.columns
]


risk_df = df[
    available_columns
].copy()


# ============================================================
# SORT BY RISK
# ============================================================

risk_df = risk_df.sort_values(
    "OverallRiskScore",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "project_risk_scores.csv"
)

risk_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK ENGINE SUMMARY")
print("=" * 70)

print(
    "\nRisk distribution:"
)

print(
    risk_df["RiskLevel"]
    .value_counts()
)


print(
    "\nAlert distribution:"
)

print(
    risk_df["AlertType"]
    .value_counts()
)


print(
    "\nTop 10 highest-risk projects:"
)

print(
    risk_df[
        [
            "ProjectCode",
            "ProjectName",
            "OverallRiskScore",
            "RiskLevel",
            "CostOverrunProbability",
            "TimeOverrunProbability",
            "AlertType",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(output_file)


print("\n" + "=" * 70)
print("PRISM RISK ENGINE COMPLETE")
print("=" * 70)
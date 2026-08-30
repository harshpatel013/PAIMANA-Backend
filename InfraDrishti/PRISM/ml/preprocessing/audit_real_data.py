from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS = {
    "COST": BASE_DIR / "data" / "real" / "step11_cost_ml_training_ready.csv",
    "TIME": BASE_DIR / "data" / "real" / "step15_time_ml_training_ready.csv",
    "RISK": BASE_DIR / "data" / "real" / "step20_overall_risk_ml_training_ready.csv",
}


def audit_dataset(name, path):

    print("\n" + "=" * 80)
    print(f"{name} DATASET AUDIT")
    print("=" * 80)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        return

    df = pd.read_csv(path)

    print(f"\nShape:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "ProjectCode" in df.columns:
        print("\nUnique projects:")
        print(df["ProjectCode"].nunique())

        print("\nObservations per project:")
        project_counts = df["ProjectCode"].value_counts()

        print(project_counts.describe())

        print("\nProjects with multiple observations:")
        print(
            (project_counts > 1).sum()
        )

    if "ReportMonth" in df.columns:

        print("\nReportMonth information:")

        dates = pd.to_datetime(
            df["ReportMonth"],
            errors="coerce"
        )

        print(f"Minimum: {dates.min()}")
        print(f"Maximum: {dates.max()}")

        print(
            f"Invalid dates: {dates.isna().sum()}"
        )

    print("\nMissing values:")
    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:
        print("No missing values.")
    else:
        for column, count in missing.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{column}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

    print("\nNumeric summary:")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:
        print(
            df[numeric_columns]
            .describe()
            .T
            .to_string()
        )

    print("\nCategorical summary:")

    categorical_columns = df.select_dtypes(
        include=["object", "bool"]
    ).columns

    for column in categorical_columns:

        print(
            f"\n{column}"
        )

        print(
            df[column]
            .value_counts(dropna=False)
            .head(10)
            .to_string()
        )


def main():

    print("\n")
    print("#" * 80)
    print("# PRISM REAL DATA AUDIT")
    print("#" * 80)

    for name, path in DATASETS.items():

        audit_dataset(
            name,
            path
        )

    print("\n" + "#" * 80)
    print("# AUDIT COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()
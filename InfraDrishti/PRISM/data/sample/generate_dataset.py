import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_PROJECTS = 2000

ministries = [
    "Ministry of Road Transport",
    "Ministry of Railways",
    "Ministry of Power",
    "Ministry of Jal Shakti",
    "Ministry of Coal",
    "Ministry of Steel",
    "Ministry of Communications",
    "Ministry of Housing",
]

sectors = [
    "Transport & Logistics",
    "Energy",
    "Water & Sanitation",
    "Communication",
    "Social Infrastructure",
    "Coal",
    "Steel",
    "Mining",
]

states = [
    "Uttar Pradesh",
    "Maharashtra",
    "Gujarat",
    "Rajasthan",
    "Karnataka",
    "Tamil Nadu",
    "Madhya Pradesh",
    "Bihar",
    "West Bengal",
    "Odisha",
    "Andhra Pradesh",
    "Telangana",
]

agencies = [
    "Central Government Agency",
    "Public Sector Undertaking",
    "State Government Agency",
    "Infrastructure Development Authority",
]


def generate_project(project_number):

    original_cost = np.random.uniform(150, 50000)

    # Some projects experience significant cost escalation
    cost_pressure = np.random.beta(2, 8)

    revised_cost = original_cost * (
        1 + cost_pressure * np.random.uniform(0.1, 1.2)
    )

    cumulative_expenditure = revised_cost * np.random.uniform(0.05, 0.85)

    planned_duration = np.random.randint(24, 120)

    elapsed_months = np.random.randint(
        3,
        max(4, planned_duration + 10)
    )

    expected_delay = max(
        0,
        np.random.normal(
            planned_duration * cost_pressure * 0.35,
            5
        )
    )

    physical_progress = np.clip(
        (elapsed_months / planned_duration) * 100
        + np.random.normal(0, 8)
        - expected_delay * 0.5,
        0,
        100
    )

    expected_progress = np.clip(
        (elapsed_months / planned_duration) * 100,
        0,
        100
    )

    progress_gap = expected_progress - physical_progress

    total_milestones = np.random.randint(5, 30)

    delayed_milestones = np.clip(
        np.random.poisson(
            max(0.5, progress_gap / 8)
        ),
        0,
        total_milestones
    )

    completed_milestones = np.random.randint(
        0,
        total_milestones + 1
    )

    financial_progress = np.clip(
        (cumulative_expenditure / revised_cost) * 100
        + np.random.normal(0, 5),
        0,
        100
    )

    previous_physical_progress = np.clip(
        physical_progress - np.random.uniform(0, 8),
        0,
        100
    )

    progress_change = (
        physical_progress -
        previous_physical_progress
    )

    previous_expenditure = max(
        0,
        cumulative_expenditure -
        np.random.uniform(
            0,
            cumulative_expenditure * 0.15
        )
    )

    expenditure_change = (
        cumulative_expenditure -
        previous_expenditure
    )

    cost_overrun_percent = (
        (revised_cost - original_cost)
        / original_cost
    ) * 100

    time_overrun_percent = (
        expected_delay / planned_duration
    ) * 100

    # Labels used only for development/testing.
    cost_overrun = int(cost_overrun_percent >= 10)
    time_overrun = int(time_overrun_percent >= 10)

        # ---------------------------------------------------------
    # DEVELOPMENT-ONLY RISK SCORE
    # ---------------------------------------------------------
    #
    # This is NOT an ML prediction.
    # It is only used to create a realistic development dataset.
    #

    progress_risk = np.clip(
        progress_gap * 2.0,
        0,
        100
    )

    milestone_risk = (
        delayed_milestones /
        max(total_milestones, 1)
    ) * 100

    cost_risk = np.clip(
        cost_overrun_percent * 2.5,
        0,
        100
    )

    time_risk = np.clip(
        time_overrun_percent * 2.5,
        0,
        100
    )

    expenditure_risk = np.clip(
        (100 - financial_progress) * 0.8,
        0,
        100
    )]
    risk_score = (
        progress_risk * 0.30
        + milestone_risk * 0.20
        + cost_risk * 0.20
        + time_risk * 0.20
        + expenditure_risk * 0.10
    )

    # Add realistic noise so the development dataset
    # isn't perfectly deterministic.
    risk_score += np.random.normal(0, 5)

    risk_score = np.clip(
        risk_score,
        0,
        100
    )

    if risk_score >= 75:
        risk_level = "Critical"
    elif risk_score >= 55:
        risk_level = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "project_id": f"PRJ-{project_number:05d}",
        "project_name": f"Infrastructure Project {project_number:05d}",

        "ministry": np.random.choice(ministries),
        "sector": np.random.choice(sectors),
        "state": np.random.choice(states),
        "implementing_agency": np.random.choice(agencies),

        "original_cost": round(original_cost, 2),
        "revised_cost": round(revised_cost, 2),
        "cumulative_expenditure": round(
            cumulative_expenditure, 2
        ),

        "planned_duration_months": planned_duration,
        "elapsed_months": elapsed_months,
        "expected_delay_months": round(
            expected_delay, 2
        ),

        "physical_progress": round(
            physical_progress, 2
        ),

        "expected_progress": round(
            expected_progress, 2
        ),

        "financial_progress": round(
            financial_progress, 2
        ),

        "previous_physical_progress": round(
            previous_physical_progress, 2
        ),

        "progress_change": round(
            progress_change, 2
        ),

        "previous_expenditure": round(
            previous_expenditure, 2
        ),

        "expenditure_change": round(
            expenditure_change, 2
        ),

        "total_milestones": total_milestones,
        "completed_milestones": completed_milestones,
        "delayed_milestones": delayed_milestones,

        "progress_gap": round(
            progress_gap, 2
        ),

        "cost_overrun_percent": round(
            cost_overrun_percent, 2
        ),

        "time_overrun_percent": round(
            time_overrun_percent, 2
        ),

        "cost_overrun": cost_overrun,
        "time_overrun": time_overrun,

        "risk_score": round(
            risk_score, 2
        ),

        "risk_level": risk_level,
    }


def main():

    projects = [
        generate_project(i)
        for i in range(1, NUM_PROJECTS + 1)
    ]

    df = pd.DataFrame(projects)

    output_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_file = os.path.join(
        output_dir,
        "prism_development_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("=" * 60)
    print("PRISM DEVELOPMENT DATASET GENERATED")
    print("=" * 60)

    print(f"Projects: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"File: {output_file}")

    print("\nRisk Distribution:")
    print(df["risk_level"].value_counts())

    print("\nCost Overrun Distribution:")
    print(df["cost_overrun"].value_counts())

    print("\nTime Overrun Distribution:")
    print(df["time_overrun"].value_counts())

    print("\nDataset Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
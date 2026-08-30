from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "project_risk_scores.csv"


class RiskService:
    def __init__(self):
        self.df = pd.read_csv(DATA_FILE)

        # Normalize column names
        self.df.columns = [str(column).strip() for column in self.df.columns]

    def dashboard(self):

        risk_distribution = self.df["RiskLevel"].value_counts().to_dict()

        alert_distribution = self.df["AlertType"].value_counts().to_dict()

        return {
            "total_projects": int(self.df["ProjectCode"].nunique()),
            "total_records": len(self.df),
            "risk_distribution": risk_distribution,
            "alert_distribution": alert_distribution,
        }

    def get_projects(
        self,
        risk=None,
        state=None,
        agency=None,
        limit=100,
    ):

        result = self.df.copy()

        if risk:
            result = result[result["RiskLevel"].str.upper() == risk.upper()]

        if state and "State" in result.columns:
            result = result[result["State"].astype(str).str.lower() == state.lower()]

        if agency and "Agency" in result.columns:
            result = result[result["Agency"].astype(str).str.lower() == agency.lower()]

        result = result.sort_values("OverallRiskScore", ascending=False)

        return result.head(limit).to_dict(orient="records")

    def get_project(self, project_code):

        result = self.df[self.df["ProjectCode"].astype(str) == str(project_code)]

        if result.empty:
            return None

        # Most recent record
        if "ReportMonth" in result.columns:
            result = result.sort_values("ReportMonth")

        return result.iloc[-1].to_dict()

    def get_project_explanation(self, project_code):
        project = self.get_project(project_code)

        if project is None:
            return None

        cost_probability = float(project.get("CostOverrunProbability", 0))
        time_probability = float(project.get("TimeOverrunProbability", 0))
        physical_progress = float(project.get("Physical Progress", 0))
        risk_score = float(project.get("OverallRiskScore", 0))

        factors = []

        # Cost risk
        if cost_probability >= 0.70:
            factors.append(
                {
                    "feature": "Cost Overrun Probability",
                    "value": round(cost_probability * 100, 2),
                    "impact": "HIGH",
                    "description": (
                        "The model predicts a high probability of cost overrun."
                    ),
                }
            )

        # Time risk
        if time_probability >= 0.70:
            factors.append(
                {
                    "feature": "Time Overrun Probability",
                    "value": round(time_probability * 100, 2),
                    "impact": "HIGH",
                    "description": (
                        "The model predicts a high probability of schedule/time overrun."
                    ),
                }
            )

        # Physical progress
        if physical_progress < 50:
            factors.append(
                {
                    "feature": "Physical Progress",
                    "value": round(physical_progress, 2),
                    "impact": "HIGH",
                    "description": (
                        "Physical progress is significantly behind the expected project trajectory."
                    ),
                }
            )

        # Cost growth
        original_cost = float(project.get("OriginalCost", 0))
        revised_cost = float(project.get("RevisedCost", 0))

        if original_cost > 0 and revised_cost > original_cost:
            cost_growth = ((revised_cost - original_cost) / original_cost) * 100

            if cost_growth >= 20:
                factors.append(
                    {
                        "feature": "Cost Growth",
                        "value": round(cost_growth, 2),
                        "impact": "HIGH",
                        "description": (
                            "The revised project cost is substantially higher than the original approved cost."
                        ),
                    }
                )

        # Expenditure
        expenditure = float(project.get("Expenditure", 0))

        if original_cost > 0:
            expenditure_ratio = (expenditure / original_cost) * 100

            if expenditure_ratio >= 80 and physical_progress < 70:
                factors.append(
                    {
                        "feature": "Expenditure vs Physical Progress",
                        "value": round(expenditure_ratio, 2),
                        "impact": "MEDIUM",
                        "description": (
                            "Expenditure is high relative to the current physical progress."
                        ),
                    }
                )

        # Fallback
        if not factors:
            factors.append(
                {
                    "feature": "Overall Risk Assessment",
                    "value": round(risk_score, 2),
                    "impact": "MEDIUM",
                    "description": (
                        "The project has been flagged based on the combined risk assessment."
                    ),
                }
            )

        return {
            "project_code": int(project["ProjectCode"]),
            "project_name": project.get("ProjectName"),
            "report_month": str(project.get("ReportMonth")),
            "risk_level": project.get("RiskLevel"),
            "overall_risk_score": round(risk_score, 2),
            "cost_overrun_probability": round(cost_probability * 100, 2),
            "time_overrun_probability": round(time_probability * 100, 2),
            "top_factors": factors[:5],
            "explanation": project.get(
                "RecommendedAction",
                "Project requires monitoring and investigation.",
            ),
        }

    def analytics(self):
        df = self.df.copy()

        # Risk by agency
        agency_risk = (
            df.groupby("Agency")
            .agg(
                projects=("ProjectCode", "nunique"),
                average_risk=("OverallRiskScore", "mean"),
                high_risk=("RiskLevel", lambda x: (x == "HIGH").sum()),
            )
            .reset_index()
            .sort_values("average_risk", ascending=False)
            .head(10)
        )

        # Risk trend by month
        monthly = (
            df.groupby("ReportMonth")
            .agg(
                risk=("OverallRiskScore", "mean"),
                warnings=("RiskLevel", lambda x: (x != "LOW").sum()),
            )
            .reset_index()
            .sort_values("ReportMonth")
        )

        return {
            "agency_risk": agency_risk.to_dict(orient="records"),
            "risk_trend": monthly.to_dict(orient="records"),
        }

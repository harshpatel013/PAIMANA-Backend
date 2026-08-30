from typing import Any
import re


class AssistantService:

    def __init__(self, risk_service):
        self.risk_service = risk_service

    # =========================================================
    # MAIN ASSISTANT
    # =========================================================

    def answer(self, question: str) -> dict[str, Any]:

        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "intent": "unknown",
                "data": None,
            }

        lower = question.lower()

        # -----------------------------------------------------
        # PROJECT ID DETECTION
        # -----------------------------------------------------

        project_code = self._extract_project_code(question)

        # -----------------------------------------------------
        # PROJECT-SPECIFIC QUESTIONS
        # -----------------------------------------------------

        if project_code:

            project = self.risk_service.get_project(project_code)

            if project is None:
                return {
                    "answer": (
                        f"I could not find project {project_code} "
                        "in the PRISM dataset."
                    ),
                    "intent": "project_not_found",
                    "data": None,
                }

            explanation = self.risk_service.get_project_explanation(
                project_code
            )

            if any(
                word in lower
                for word in [
                    "why",
                    "risk",
                    "problem",
                    "reason",
                    "explain",
                    "status",
                    "condition",
                    "about",
                    "tell me",
                ]
            ):
                return self._project_risk_response(
                    project,
                    explanation,
                )

        # -----------------------------------------------------
        # HIGH / CRITICAL RISK PROJECTS
        # -----------------------------------------------------

        if any(
            phrase in lower
            for phrase in [
                "highest risk",
                "high risk projects",
                "most risky",
                "critical projects",
                "top risk projects",
                "riskiest projects",
            ]
        ):

            projects = self.risk_service.get_projects(
                risk="HIGH",
                limit=10,
            )

            return {
                "answer": self._format_project_list(
                    "Highest-risk projects in the current PRISM assessment:",
                    projects,
                ),
                "intent": "high_risk_projects",
                "data": projects,
            }

        # -----------------------------------------------------
        # COST RISK
        # -----------------------------------------------------

        if (
            "cost" in lower
            and any(
                word in lower
                for word in [
                    "risk",
                    "overrun",
                    "problem",
                    "warning",
                ]
            )
            and "time" not in lower
        ):

            df = self.risk_service.df

            if "CostWarning" in df.columns:

                projects = (
                    df[df["CostWarning"] == True]
                    .sort_values(
                        "OverallRiskScore",
                        ascending=False,
                    )
                    .head(10)
                )

                records = projects.to_dict(
                    orient="records"
                )

                return {
                    "answer": self._format_project_list(
                        "Projects showing cost-overrun warning signals:",
                        records,
                    ),
                    "intent": "cost_risk",
                    "data": records,
                }

        # -----------------------------------------------------
        # TIME / SCHEDULE RISK
        # -----------------------------------------------------

        if (
            any(
                word in lower
                for word in [
                    "time risk",
                    "schedule risk",
                    "schedule delay",
                    "time overrun",
                    "delay risk",
                ]
            )
            and "cost" not in lower
        ):

            df = self.risk_service.df

            if "TimeWarning" in df.columns:

                projects = (
                    df[df["TimeWarning"] == True]
                    .sort_values(
                        "OverallRiskScore",
                        ascending=False,
                    )
                    .head(10)
                )

                records = projects.to_dict(
                    orient="records"
                )

                return {
                    "answer": self._format_project_list(
                        "Projects showing schedule/time warning signals:",
                        records,
                    ),
                    "intent": "time_risk",
                    "data": records,
                }

        # -----------------------------------------------------
        # COST + TIME RISK
        # -----------------------------------------------------

        if (
            "cost" in lower
            and "time" in lower
            and any(
                word in lower
                for word in [
                    "risk",
                    "overrun",
                    "project",
                    "projects",
                    "warning",
                ]
            )
        ):

            df = self.risk_service.df

            if (
                "CostWarning" in df.columns
                and "TimeWarning" in df.columns
            ):

                projects = (
                    df[
                        (df["CostWarning"] == True)
                        & (df["TimeWarning"] == True)
                    ]
                    .sort_values(
                        "OverallRiskScore",
                        ascending=False,
                    )
                    .head(10)
                )

                records = projects.to_dict(
                    orient="records"
                )

                return {
                    "answer": self._format_project_list(
                        "Projects showing both cost and time warning signals:",
                        records,
                    ),
                    "intent": "cost_and_time_risk",
                    "data": records,
                }

        # -----------------------------------------------------
        # AGENCY ANALYSIS
        # -----------------------------------------------------

        if any(
            phrase in lower
            for phrase in [
                "agency",
                "agencies",
                "ministry",
                "department",
            ]
        ) and any(
            word in lower
            for word in [
                "highest",
                "risk",
                "risky",
                "performance",
                "average",
            ]
        ):

            analytics = self.risk_service.analytics()

            agency_risk = analytics.get(
                "agency_risk",
                [],
            )

            if agency_risk:

                top = agency_risk[:10]

                lines = [
                    "Agencies with the highest average project risk:"
                ]

                for index, item in enumerate(
                    top,
                    start=1,
                ):

                    agency = item.get(
                        "Agency",
                        "Unknown",
                    )

                    average = float(
                        item.get(
                            "average_risk",
                            0,
                        )
                    )

                    projects = item.get(
                        "projects",
                        0,
                    )

                    lines.append(
                        f"{index}. {agency} — "
                        f"Average risk: {average:.2f}% — "
                        f"Projects: {projects}"
                    )

                return {
                    "answer": "\n".join(lines),
                    "intent": "agency_analysis",
                    "data": top,
                }

        # -----------------------------------------------------
        # STATE ANALYSIS
        # -----------------------------------------------------

        if "state" in lower and any(
            word in lower
            for word in [
                "risk",
                "risky",
                "highest",
                "projects",
            ]
        ):

            df = self.risk_service.df

            if "State" in df.columns:

                state_risk = (
                    df.groupby("State")
                    .agg(
                        projects=(
                            "ProjectCode",
                            "nunique",
                        ),
                        average_risk=(
                            "OverallRiskScore",
                            "mean",
                        ),
                    )
                    .reset_index()
                    .sort_values(
                        "average_risk",
                        ascending=False,
                    )
                    .head(10)
                )

                records = state_risk.to_dict(
                    orient="records"
                )

                lines = [
                    "States with the highest average project risk:"
                ]

                for index, item in enumerate(
                    records,
                    start=1,
                ):

                    lines.append(
                        f"{index}. "
                        f"{item['State']} — "
                        f"Average risk: "
                        f"{float(item['average_risk']):.2f}% — "
                        f"Projects: {item['projects']}"
                    )

                return {
                    "answer": "\n".join(lines),
                    "intent": "state_analysis",
                    "data": records,
                }

        # -----------------------------------------------------
        # PORTFOLIO / DASHBOARD
        # -----------------------------------------------------

        if any(
            word in lower
            for word in [
                "portfolio",
                "dashboard",
                "overall status",
                "overall situation",
                "overall health",
                "how many projects",
                "portfolio status",
            ]
        ):

            dashboard = self.risk_service.dashboard()

            risk_distribution = dashboard.get(
                "risk_distribution",
                {},
            )

            return {
                "answer": (
                    f"PRISM is currently monitoring "
                    f"{dashboard['total_projects']} unique projects "
                    f"across {dashboard['total_records']} records. "
                    f"There are "
                    f"{risk_distribution.get('HIGH', 0)} "
                    f"HIGH-risk records, "
                    f"{risk_distribution.get('MEDIUM', 0)} "
                    f"MEDIUM-risk records, and "
                    f"{risk_distribution.get('LOW', 0)} "
                    f"LOW-risk records."
                ),
                "intent": "portfolio_summary",
                "data": dashboard,
            }

        # -----------------------------------------------------
        # RISK COUNT
        # -----------------------------------------------------

        if (
            any(
                word in lower
                for word in [
                    "how many",
                    "number of",
                    "count",
                ]
            )
            and "high" in lower
            and "risk" in lower
        ):

            dashboard = self.risk_service.dashboard()

            count = dashboard.get(
                "risk_distribution",
                {},
            ).get(
                "HIGH",
                0,
            )

            return {
                "answer": (
                    f"There are currently {count} "
                    "HIGH-risk records in the PRISM assessment."
                ),
                "intent": "high_risk_count",
                "data": {
                    "high_risk_records": count
                },
            }

        # -----------------------------------------------------
        # HELP
        # -----------------------------------------------------

        return {
            "answer": (
                "I can help analyse PRISM project intelligence. "
                "You can ask about a project ID, highest-risk projects, "
                "cost risk, schedule risk, projects with both cost and "
                "time risk, agency performance, state-level risk, or "
                "overall portfolio status."
            ),
            "intent": "help",
            "data": None,
        }

    # =========================================================
    # PROJECT CODE EXTRACTION
    # =========================================================

    def _extract_project_code(
        self,
        question: str,
    ):

        matches = re.findall(
            r"\b\d{6}\b",
            question,
        )

        if matches:
            return int(matches[0])

        return None

    # =========================================================
    # PROJECT RISK RESPONSE
    # =========================================================

    def _project_risk_response(
        self,
        project,
        explanation,
    ):

        project_code = project["ProjectCode"]

        name = project.get(
            "ProjectName",
            f"Project {project_code}",
        )

        risk_level = project.get(
            "RiskLevel",
            "UNKNOWN",
        )

        overall_score = float(
            project.get(
                "OverallRiskScore",
                0,
            )
        )

        cost_probability = float(
            project.get(
                "CostOverrunProbability",
                0,
            )
        )

        # Handle both decimal and percentage formats
        if cost_probability <= 1:
            cost_probability *= 100

        time_probability = float(
            project.get(
                "TimeOverrunProbability",
                0,
            )
        )

        if time_probability <= 1:
            time_probability *= 100

        factors = (
            explanation.get(
                "top_factors",
                [],
            )
            if explanation
            else []
        )

        answer = (
            f"Project {project_code} — {name} "
            f"is currently classified as {risk_level} risk "
            f"with an overall risk score of "
            f"{overall_score:.2f}. "
            f"Cost-overrun probability is "
            f"{cost_probability:.2f}% and time-overrun "
            f"probability is {time_probability:.2f}%."
        )

        if factors:

            factor_text = []

            for factor in factors[:3]:

                feature = factor.get(
                    "feature",
                    "Unknown factor",
                )

                value = factor.get(
                    "value",
                    "",
                )

                impact = factor.get(
                    "impact",
                    "",
                )

                text = f"{feature}"

                if value != "":
                    text += f" ({value})"

                if impact:
                    text += f" [{impact}]"

                factor_text.append(text)

            answer += (
                " The main contributing signals are: "
                + ", ".join(factor_text)
                + "."
            )

        if explanation:

            recommended_action = explanation.get(
                "explanation"
            )

            if recommended_action:

                answer += (
                    f" Recommended action: "
                    f"{recommended_action}"
                )

        return {
            "answer": answer,
            "intent": "project_risk_explanation",
            "data": {
                "project": project,
                "explanation": explanation,
            },
        }

    # =========================================================
    # PROJECT LIST FORMATTER
    # =========================================================

    def _format_project_list(
        self,
        heading,
        projects,
    ):

        if not projects:
            return (
                heading
                + " No matching projects were found."
            )

        lines = [heading]

        for index, project in enumerate(
            projects[:10],
            start=1,
        ):

            code = project.get(
                "ProjectCode",
                "N/A",
            )

            score = float(
                project.get(
                    "OverallRiskScore",
                    0,
                )
            )

            risk = project.get(
                "RiskLevel",
                "UNKNOWN",
            )

            name = project.get(
                "ProjectName",
                f"Project {code}",
            )

            lines.append(
                f"{index}. {code} — "
                f"{name} — "
                f"{risk} — "
                f"Risk Score: {score:.2f}"
            )

        return "\n".join(lines)
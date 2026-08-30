from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.services.risk_service import RiskService
from backend.services.assistant_service import AssistantService


app = FastAPI(
    title="PRISM Risk Intelligence API",
    description=(
        "AI-powered infrastructure project "
        "risk monitoring and prediction API"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVICE
# ============================================================

risk_service = RiskService()
assistant_service = AssistantService(risk_service)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "system": "PRISM",
        "status": "running",
        "message": "Infrastructure Project Risk Intelligence API",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "records_loaded": len(
            risk_service.df
        ),
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    return risk_service.dashboard()


# ============================================================
# PROJECTS
# ============================================================

@app.get("/api/projects")
def projects(
    risk: str | None = Query(
        default=None
    ),
    state: str | None = Query(
        default=None
    ),
    agency: str | None = Query(
        default=None
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000
    ),
):

    return risk_service.get_projects(
        risk=risk,
        state=state,
        agency=agency,
        limit=limit,
    )


# ============================================================
# HIGH-RISK PROJECTS
# ============================================================

@app.get("/api/projects/high-risk")
def high_risk_projects(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    )
):

    return risk_service.get_projects(
        risk="HIGH",
        limit=limit,
    )

@app.get("/api/analytics")
def analytics():

    return risk_service.analytics()


# ============================================================
# SINGLE PROJECT
# ============================================================

@app.get("/api/projects/{project_code}")
def project(project_code: int):

    result = risk_service.get_project(
        project_code
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    return result

@app.get("/api/projects/{project_code}/explanation")
def project_explanation(project_code: int):

    result = risk_service.get_project_explanation(
        project_code
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Risk explanation not found"
        )

    return result

@app.get("/api/assistant")
def assistant(
    question: str = Query(..., min_length=1)
):
    return assistant_service.answer(question)
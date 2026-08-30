from typing import Optional

from pydantic import BaseModel


class ProjectPredictionRequest(BaseModel):

    ProjectCode: int

    OriginalCost: Optional[float] = None
    RevisedCost: Optional[float] = None
    Expenditure: Optional[float] = None

    PhysicalProgress: Optional[float] = None

    State: Optional[str] = None
    Agency: Optional[str] = None
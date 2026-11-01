from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from property_guardian_ai.models.schemas import (
    RiskAssessment,
    RiskChangeExplanation,
    RiskSnapshot,
    ValidationReport,
)
from property_guardian_ai.services.container import Services, get_services
from property_guardian_ai.services.logging_config import configure_logging
from property_guardian_ai.services.property_repository import PropertyNotFoundError

configure_logging()
app = FastAPI(title="Property Guardian AI", version="0.1.0")

ServicesDep = Annotated[Services, Depends(get_services)]


class AssessRequest(BaseModel):
    property_id: str


class PropertySummary(BaseModel):
    property_id: str
    address: str
    latest_month: date | None
    latest_score: float | None


def _require_property(services: Services, property_id: str) -> None:
    try:
        services.repository.get(property_id)
    except PropertyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown property_id {property_id!r}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/properties")
def list_properties(services: ServicesDep) -> list[PropertySummary]:
    summaries = []
    for pid in services.repository.property_ids():
        history = services.memory.history(pid)
        summaries.append(PropertySummary(
            property_id=pid,
            address=services.repository.get(pid).property.address,
            latest_month=history[-1].month if history else None,
            latest_score=history[-1].risk_score if history else None,
        ))
    return summaries


@app.post("/assess")
def assess(request: AssessRequest, services: ServicesDep) -> RiskAssessment:
    _require_property(services, request.property_id)
    return services.orchestrator.assess(request.property_id)


@app.get("/properties/{property_id}/history")
def history(property_id: str, services: ServicesDep) -> list[RiskSnapshot]:
    _require_property(services, property_id)
    return services.memory.history(property_id)


@app.get("/properties/{property_id}/risk-change")
def risk_change(property_id: str, services: ServicesDep) -> RiskChangeExplanation:
    _require_property(services, property_id)
    explanation = services.memory.explain_change(property_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Not enough history to explain a change")
    return explanation


@app.get("/model/validation")
def model_validation(services: ServicesDep) -> ValidationReport:
    return services.model.validation


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("property_guardian_ai.app.main:app", host="127.0.0.1", port=8000, reload=True)

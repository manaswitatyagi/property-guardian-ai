from __future__ import annotations

from fastapi.testclient import TestClient

from property_guardian_ai.app.main import app
from property_guardian_ai.services.container import get_services


def test_assessment_always_explains_its_score(services) -> None:
    assessment = services.orchestrator.assess("KC-003")

    assert 0 <= assessment.risk_score <= 100
    assert assessment.scoring_method.startswith("xgboost")
    assert assessment.primary_drivers and all(d.evidence for d in assessment.primary_drivers)
    assert assessment.recommendations
    assert assessment.confidence.reasons
    assert assessment.risk_change is not None
    assert "simulated" in " ".join(assessment.confidence.reasons)


def test_assess_persists_recommendations_to_memory(services) -> None:
    assessment = services.orchestrator.assess("KC-007")
    latest = services.memory.history("KC-007")[-1]

    assert latest.recommendations == assessment.recommendations


def test_api_endpoints(services) -> None:
    app.dependency_overrides[get_services] = lambda: services
    client = TestClient(app)
    try:
        assert client.get("/health").json() == {"status": "ok"}
        assert len(client.get("/properties").json()) == 10
        assert client.post("/assess", json={"property_id": "KC-001"}).status_code == 200
        assert client.post("/assess", json={"property_id": "NOPE"}).status_code == 404
        assert len(client.get("/properties/KC-001/history").json()) == 12
        assert client.get("/properties/KC-001/risk-change").json()["property_id"] == "KC-001"
        assert client.get("/model/validation").json()["n_folds"] == 5
    finally:
        app.dependency_overrides.clear()

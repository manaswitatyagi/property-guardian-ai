from __future__ import annotations

from datetime import date

from property_guardian_ai.agents.memory_agent import MemoryAgent
from property_guardian_ai.memory.store import InMemoryRiskStore
from property_guardian_ai.models.schemas import RiskSnapshot


def _snapshot(month: int, score: float, rain: float, rain_pts: float) -> RiskSnapshot:
    return RiskSnapshot(
        property_id="P", month=date(2026, month, 1), risk_score=score,
        features={"heavy_rain_events": rain, "roof_age": 20},
        contributions={"heavy_rain_events": rain_pts, "roof_age": 10.0},
    )


def test_explain_change_needs_two_months() -> None:
    agent = MemoryAgent(InMemoryRiskStore(), {})
    agent.record(_snapshot(1, 50, 1, 0.0))
    assert agent.explain_change("P") is None


def test_explain_change_attributes_delta_to_feature_contributions() -> None:
    agent = MemoryAgent(InMemoryRiskStore(), {"heavy_rain_events": "Heavy rain"})
    agent.record(_snapshot(2, 58, 5, 8.0))
    agent.record(_snapshot(1, 50, 1, 0.0))

    change = agent.explain_change("P")

    assert change is not None
    assert change.direction == "increased"
    assert change.score_change == 8
    assert change.top_changes[0].feature == "heavy_rain_events"
    assert change.top_changes[0].contribution_change == 8.0
    assert "Heavy rain 1 → 5" in change.narrative


def test_upsert_replaces_same_month() -> None:
    store = InMemoryRiskStore()
    store.upsert(_snapshot(1, 50, 1, 0.0))
    store.upsert(_snapshot(1, 70, 1, 0.0))
    assert [s.risk_score for s in store.history("P")] == [70]

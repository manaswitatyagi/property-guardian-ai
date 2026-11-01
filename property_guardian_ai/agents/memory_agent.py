from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent
from property_guardian_ai.memory.store import RiskMemoryStore
from property_guardian_ai.models.schemas import FeatureChange, RiskChangeExplanation, RiskSnapshot

STABLE_BAND_PTS = 1.0
TOP_CHANGES = 3


def _direction(delta: float) -> str:
    if delta > STABLE_BAND_PTS:
        return "increased"
    if delta < -STABLE_BAND_PTS:
        return "decreased"
    return "remained stable"


def _feature_changes(previous: RiskSnapshot, current: RiskSnapshot, labels: dict[str, str]) -> list[FeatureChange]:
    changes = [
        FeatureChange(
            feature=name,
            label=labels.get(name, name),
            previous_value=previous.features.get(name, 0.0),
            current_value=current.features.get(name, 0.0),
            contribution_change=round(current.contributions.get(name, 0.0) - previous.contributions.get(name, 0.0), 2),
        )
        for name in current.contributions
    ]
    return sorted(changes, key=lambda c: abs(c.contribution_change), reverse=True)[:TOP_CHANGES]


def _narrative(direction: str, delta: float, changes: list[FeatureChange]) -> str:
    if direction == "remained stable" or not changes:
        return f"Risk {direction} ({delta:+.1f} pts); no single driver moved materially."
    parts = [
        f"{c.label} {c.previous_value:g} → {c.current_value:g} ({c.contribution_change:+.1f} pts)"
        for c in changes if abs(c.contribution_change) >= 0.1
    ]
    return f"Risk {direction} by {abs(delta):.1f} pts. Main model contributions: " + "; ".join(parts) + "."


class MemoryAgent(BaseAgent):
    """Remembers monthly scores, features and recommendations; explains how risk moved."""

    name = "memory-agent"

    def __init__(self, store: RiskMemoryStore, feature_labels: dict[str, str]) -> None:
        self._store = store
        self._labels = feature_labels

    def run(self, property_id: str) -> RiskChangeExplanation | None:
        return self.explain_change(property_id)

    def record(self, snapshot: RiskSnapshot) -> None:
        self._store.upsert(snapshot)

    def history(self, property_id: str) -> list[RiskSnapshot]:
        return self._store.history(property_id)

    def explain_change(self, property_id: str) -> RiskChangeExplanation | None:
        history = self._store.history(property_id)
        if len(history) < 2:
            return None
        previous, current = history[-2], history[-1]
        delta = round(current.risk_score - previous.risk_score, 2)
        direction = _direction(delta)
        changes = _feature_changes(previous, current, self._labels)
        return RiskChangeExplanation(
            property_id=property_id,
            previous_month=previous.month,
            current_month=current.month,
            previous_score=previous.risk_score,
            current_score=current.risk_score,
            score_change=delta,
            direction=direction,
            top_changes=changes,
            narrative=_narrative(direction, delta, changes),
        )

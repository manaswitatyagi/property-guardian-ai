from __future__ import annotations

from datetime import date
from typing import Protocol

from property_guardian_ai.models.schemas import RiskSnapshot


class RiskMemoryStore(Protocol):
    """Persistence contract for risk snapshots. A PostgreSQL implementation can drop in here."""

    def upsert(self, snapshot: RiskSnapshot) -> None: ...

    def history(self, property_id: str) -> list[RiskSnapshot]: ...


class InMemoryRiskStore:
    """Process-local store keyed by (property_id, month); upserts replace the month's snapshot."""

    def __init__(self) -> None:
        self._snapshots: dict[str, dict[date, RiskSnapshot]] = {}

    def upsert(self, snapshot: RiskSnapshot) -> None:
        self._snapshots.setdefault(snapshot.property_id, {})[snapshot.month] = snapshot

    def history(self, property_id: str) -> list[RiskSnapshot]:
        by_month = self._snapshots.get(property_id, {})
        return [by_month[month] for month in sorted(by_month)]

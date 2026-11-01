from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from property_guardian_ai.agents.memory_agent import MemoryAgent
from property_guardian_ai.memory.store import InMemoryRiskStore
from property_guardian_ai.models.schemas import RiskSnapshot
from property_guardian_ai.services.property_repository import DEFAULT_DATA_DIR, PropertyRepository
from property_guardian_ai.services.risk_model import FEATURES, RiskModel, feature_vector
from property_guardian_ai.workflows.orchestrator import Orchestrator


@dataclass(frozen=True)
class Services:
    repository: PropertyRepository
    model: RiskModel
    memory: MemoryAgent
    orchestrator: Orchestrator


def seed_memory(memory: MemoryAgent, repository: PropertyRepository, model: RiskModel) -> None:
    """Back-fill monthly snapshots so the Memory Agent can explain trends from day one."""
    for property_id in repository.property_ids():
        record = repository.get(property_id)
        for month in repository.monthly_for(property_id):
            scored = model.score(record, month)
            memory.record(RiskSnapshot(
                property_id=property_id,
                month=month.month,
                risk_score=scored.score,
                features=feature_vector(record, month),
                contributions={c.feature: c.contribution for c in scored.contributions},
            ))


def build_services(data_dir: Path = DEFAULT_DATA_DIR) -> Services:
    repository = PropertyRepository.from_csv(data_dir)
    model = RiskModel.train(repository)
    memory = MemoryAgent(InMemoryRiskStore(), {spec.name: spec.label for spec in FEATURES})
    seed_memory(memory, repository, model)
    return Services(repository, model, memory, Orchestrator(repository, memory, model))


@lru_cache(maxsize=1)
def get_services() -> Services:
    return build_services()

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from property_guardian_ai.agents.climate_agent import ClimateAgent
from property_guardian_ai.agents.flood_agent import FloodAgent
from property_guardian_ai.agents.inflation_agent import InflationAgent
from property_guardian_ai.agents.memory_agent import MemoryAgent
from property_guardian_ai.agents.neighborhood_agent import NeighborhoodAgent
from property_guardian_ai.agents.property_agent import PropertyAgent
from property_guardian_ai.agents.recommendation_agent import RecommendationAgent, RecommendationContext
from property_guardian_ai.agents.risk_reasoning_agent import ReasoningInputs, RiskReasoningAgent
from property_guardian_ai.agents.weather_agent import WeatherAgent
from property_guardian_ai.models.schemas import (
    ClimateRisk,
    FloodRisk,
    ModelScore,
    NeighborhoodRisk,
    PropertyProfile,
    PropertyRecord,
    Recommendation,
    ReplacementCostRisk,
    RiskAssessment,
    RiskChangeExplanation,
    RiskSnapshot,
    WeatherRisk,
)
from property_guardian_ai.services.property_repository import PropertyRepository
from property_guardian_ai.services.risk_model import RiskModel, feature_vector

logger = logging.getLogger(__name__)

HAZARD_NODES = ("weather", "flood", "climate", "neighborhood", "inflation")


class GuardianState(TypedDict, total=False):
    record: PropertyRecord
    profile: PropertyProfile
    weather: WeatherRisk
    flood: FloodRisk
    climate: ClimateRisk
    neighborhood: NeighborhoodRisk
    cost: ReplacementCostRisk
    model_score: ModelScore | None
    recommendations: list[Recommendation]
    risk_change: RiskChangeExplanation | None
    assessment: RiskAssessment


Node = Callable[[GuardianState], dict[str, Any]]


def _logged(name: str, node: Node) -> Node:
    def wrapper(state: GuardianState) -> dict[str, Any]:
        started = time.perf_counter()
        update = node(state)
        logger.info("agent_completed", extra={
            "agent": name,
            "property_id": state["record"].property.property_id,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return update
    return wrapper


class Orchestrator:
    """Coordinates the agents as a LangGraph workflow. Agents only exchange typed objects via state."""

    def __init__(self, repository: PropertyRepository, memory_agent: MemoryAgent, model: RiskModel | None) -> None:
        self._repository = repository
        self._memory = memory_agent
        self._model = model
        self._property = PropertyAgent()
        self._weather = WeatherAgent()
        self._flood = FloodAgent()
        self._climate = ClimateAgent()
        self._neighborhood = NeighborhoodAgent()
        self._inflation = InflationAgent()
        self._recommender = RecommendationAgent()
        self._reasoner = RiskReasoningAgent()
        self._graph = self._build_graph()

    def assess(self, property_id: str) -> RiskAssessment:
        record = self._repository.get(property_id)
        final_state = self._graph.invoke({"record": record})
        return final_state["assessment"]

    # -- nodes -------------------------------------------------------------

    def _property_node(self, state: GuardianState) -> dict[str, Any]:
        return {"profile": self._property.run(state["record"].property)}

    def _weather_node(self, state: GuardianState) -> dict[str, Any]:
        return {"weather": self._weather.run(state["profile"], state["record"].weather)}

    def _flood_node(self, state: GuardianState) -> dict[str, Any]:
        return {"flood": self._flood.run(state["profile"], state["record"].flood)}

    def _climate_node(self, state: GuardianState) -> dict[str, Any]:
        return {"climate": self._climate.run(state["profile"], state["record"].climate)}

    def _neighborhood_node(self, state: GuardianState) -> dict[str, Any]:
        return {"neighborhood": self._neighborhood.run(state["profile"], state["record"].neighborhood)}

    def _inflation_node(self, state: GuardianState) -> dict[str, Any]:
        return {"cost": self._inflation.run(state["profile"], state["record"].cost)}

    def _model_node(self, state: GuardianState) -> dict[str, Any]:
        latest = self._repository.latest_month(state["record"].property.property_id)
        if self._model is None or latest is None:
            return {"model_score": None}
        return {"model_score": self._model.score(state["record"], latest)}

    def _recommendation_node(self, state: GuardianState) -> dict[str, Any]:
        context = RecommendationContext(
            profile=state["profile"], weather=state["weather"], flood=state["flood"],
            climate=state["climate"], neighborhood=state["neighborhood"], cost=state["cost"],
        )
        return {"recommendations": self._recommender.run(context)}

    def _memory_node(self, state: GuardianState) -> dict[str, Any]:
        return {"risk_change": self._memory.explain_change(state["record"].property.property_id)}

    def _reasoning_node(self, state: GuardianState) -> dict[str, Any]:
        inputs = ReasoningInputs(
            profile=state["profile"], weather=state["weather"], flood=state["flood"],
            climate=state["climate"], neighborhood=state["neighborhood"], cost=state["cost"],
            recommendations=state["recommendations"], model_score=state.get("model_score"),
            risk_change=state.get("risk_change"),
        )
        return {"assessment": self._reasoner.run(inputs)}

    def _persist_node(self, state: GuardianState) -> dict[str, Any]:
        """Upsert this month's snapshot (with recommendations) so history stays current."""
        record, assessment = state["record"], state["assessment"]
        latest = self._repository.latest_month(record.property.property_id)
        if latest is None:
            return {}
        model_score = assessment.model_explanation
        self._memory.record(RiskSnapshot(
            property_id=record.property.property_id,
            month=latest.month,
            risk_score=assessment.risk_score,
            features=feature_vector(record, latest),
            contributions={c.feature: c.contribution for c in model_score.contributions} if model_score else {},
            recommendations=assessment.recommendations,
        ))
        return {}

    # -- graph -------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(GuardianState)
        nodes: dict[str, Node] = {
            "property": self._property_node,
            "weather": self._weather_node,
            "flood": self._flood_node,
            "climate": self._climate_node,
            "neighborhood": self._neighborhood_node,
            "inflation": self._inflation_node,
            "model_scoring": self._model_node,
            "recommendation": self._recommendation_node,
            "memory": self._memory_node,
            "reasoning": self._reasoning_node,
            "persist": self._persist_node,
        }
        for name, node in nodes.items():
            graph.add_node(name, _logged(name, node))

        graph.add_edge(START, "property")
        for hazard in HAZARD_NODES:
            graph.add_edge("property", hazard)
        graph.add_edge(list(HAZARD_NODES), "model_scoring")
        graph.add_edge("model_scoring", "recommendation")
        graph.add_edge("recommendation", "memory")
        graph.add_edge("memory", "reasoning")
        graph.add_edge("reasoning", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

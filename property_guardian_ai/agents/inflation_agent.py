from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.models.schemas import CostObservation, PropertyProfile, ReplacementCostRisk


class InflationAgent(BaseAgent):
    """Replacement-cost pressure: how fast rebuild costs are outrunning coverage."""

    name = "inflation-agent"

    def run(self, property_profile: PropertyProfile, observation: CostObservation) -> ReplacementCostRisk:
        risk_score = bounded_score(
            observation.construction_inflation_yoy_pct * 7
            + (observation.labor_cost_index - 100) * 0.6
            + property_profile.property_age * 0.04
        )

        return ReplacementCostRisk(
            construction_inflation=observation.construction_inflation_yoy_pct,
            labor_cost_index=observation.labor_cost_index,
            risk_score=risk_score,
            provenance=observation.provenance,
        )

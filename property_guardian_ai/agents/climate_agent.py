from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.models.schemas import ClimateObservation, ClimateRisk, PropertyProfile


class ClimateAgent(BaseAgent):
    name = "climate-agent"

    def run(self, property_profile: PropertyProfile, observation: ClimateObservation) -> ClimateRisk:
        risk_score = bounded_score(
            abs(observation.heat_trend_c_per_decade) * 40
            + abs(observation.precipitation_trend_pct_per_decade) * 2.5
            + max(0.0, observation.wildfire_trend_index_pts_per_decade) * 2.5
            + property_profile.property_age * 0.05
        )

        return ClimateRisk(
            heat_trend=observation.heat_trend_c_per_decade,
            precipitation_trend=observation.precipitation_trend_pct_per_decade,
            wildfire_trend=observation.wildfire_trend_index_pts_per_decade,
            risk_score=risk_score,
            provenance=observation.provenance,
        )

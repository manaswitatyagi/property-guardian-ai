from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.models.schemas import NeighborhoodObservation, NeighborhoodRisk, PropertyProfile


class NeighborhoodAgent(BaseAgent):
    name = "neighborhood-agent"

    def run(self, property_profile: PropertyProfile, observation: NeighborhoodObservation) -> NeighborhoodRisk:
        risk_score = bounded_score(
            abs(observation.population_trend_pct_5yr) * 1.5
            + observation.crime_index * 0.8
            + (observation.urban_density_per_sqmi / 1000) * 3
            + property_profile.property_age * 0.05
        )

        return NeighborhoodRisk(
            population_trend=observation.population_trend_pct_5yr,
            crime_index=observation.crime_index,
            urban_density=observation.urban_density_per_sqmi,
            risk_score=risk_score,
            provenance=observation.provenance,
        )

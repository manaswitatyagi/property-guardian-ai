from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.models.schemas import FloodObservation, FloodRisk, PropertyProfile

# Relative weight of FEMA zone designations (X_SHADED = 0.2% annual chance).
ZONE_WEIGHTS: dict[str, float] = {"VE": 1.0, "AE": 0.95, "A": 0.85, "X_SHADED": 0.5, "X": 0.2}
UNKNOWN_ZONE_WEIGHT = 0.6
WATER_PROXIMITY_KM = 2.0


class FloodAgent(BaseAgent):
    name = "flood-agent"

    def run(self, property_profile: PropertyProfile, observation: FloodObservation) -> FloodRisk:
        zone_factor = ZONE_WEIGHTS.get(observation.flood_zone, UNKNOWN_ZONE_WEIGHT)
        proximity = max(0.0, (WATER_PROXIMITY_KM - observation.distance_to_water_km) / WATER_PROXIMITY_KM)
        history_points = 20 if observation.historical_flooding else 0
        risk_score = bounded_score(
            zone_factor * 50 + proximity * 30 + history_points + property_profile.property_age * 0.05
        )

        return FloodRisk(
            flood_zone=observation.flood_zone,
            historical_flooding=observation.historical_flooding,
            distance_to_water_km=observation.distance_to_water_km,
            risk_score=risk_score,
            provenance=observation.provenance,
        )

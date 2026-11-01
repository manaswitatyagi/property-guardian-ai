from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.models.schemas import PropertyProfile, WeatherObservation, WeatherRisk


class WeatherAgent(BaseAgent):
    """Heuristic weather-exposure index from annualised storm event counts."""

    name = "weather-agent"

    def run(self, property_profile: PropertyProfile, observation: WeatherObservation) -> WeatherRisk:
        annualise = 12 / observation.window_months
        roof_factor = min(1.0, property_profile.roof_age / 25)
        risk_score = bounded_score(
            observation.hail_events * annualise * 10
            + observation.wind_events * annualise * 5
            + observation.heavy_rainfall_events * annualise * 2.5
            + observation.extreme_temperature_days * annualise * 1.0
            + roof_factor * 20
        )

        return WeatherRisk(
            hail_frequency=observation.hail_events,
            wind_events=observation.wind_events,
            heavy_rainfall_events=observation.heavy_rainfall_events,
            extreme_temperature_days=observation.extreme_temperature_days,
            risk_score=risk_score,
            provenance=observation.provenance,
        )

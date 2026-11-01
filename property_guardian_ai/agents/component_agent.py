from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from property_guardian_ai.agents.base import BaseAgent, bounded_score
from property_guardian_ai.agents.recommendation_agent import RecommendationContext
from property_guardian_ai.agents.risk_reasoning_agent import risk_label
from property_guardian_ai.models.schemas import ComponentRisk, ModelScore, Recommendation

Scorer = Callable[[RecommendationContext], float]
EvidenceBuilder = Callable[[RecommendationContext], list[str]]


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    name: str
    explanation: str
    score: Scorer
    evidence: EvidenceBuilder
    model_features: tuple[str, ...]
    recommendation_drivers: tuple[str, ...]


def _flood_evidence(c: RecommendationContext) -> str:
    history = {True: "prior flooding recorded", False: "no prior flooding", None: "flood history unknown"}
    return f"Flood zone {c.flood.flood_zone}, {c.flood.distance_to_water_km:g} km to water, " \
           f"{history[c.flood.historical_flooding]}"


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        key="roof", name="Roof",
        explanation="Shingle wear from age, hail impact and wind uplift.",
        score=lambda c: c.profile.roof_age * 2.6 + c.weather.hail_frequency * 8 + c.weather.wind_events * 3
        + c.weather.extreme_temperature_days * 0.5,
        evidence=lambda c: [
            f"Roof age {c.profile.roof_age} years" + (" (estimated)" if c.profile.roof_age_estimated else ""),
            f"{c.weather.hail_frequency:g} hail and {c.weather.wind_events:g} wind events in 12 months",
        ],
        model_features=("roof_age", "hail_events", "wind_events"),
        recommendation_drivers=("Roof age", "Hail exposure"),
    ),
    ComponentSpec(
        key="gutters", name="Gutters & drainage",
        explanation="Heavy rain overwhelms gutters and pools water against the foundation.",
        score=lambda c: c.weather.heavy_rainfall_events * 4 + abs(c.climate.precipitation_trend) * 2
        + min(1.0, c.profile.roof_age / 25) * 10,
        evidence=lambda c: [
            f"{c.weather.heavy_rainfall_events:g} heavy-rain events in 12 months",
            f"Precipitation trending {c.climate.precipitation_trend:+g}% per decade",
        ],
        model_features=("heavy_rain_events", "precipitation_trend"),
        recommendation_drivers=("Heavy rainfall",),
    ),
    ComponentSpec(
        key="foundation", name="Foundation & basement",
        explanation="Flood water and saturated soil cause seepage, settlement and basement flooding.",
        score=lambda c: c.flood.risk_score * 0.8 + c.weather.heavy_rainfall_events * 1.5
        + c.profile.property_age * 0.15,
        evidence=lambda c: [_flood_evidence(c), f"Built {c.profile.year_built} ({c.profile.property_age} years old)"],
        model_features=("flood_zone_code", "distance_to_water_km", "property_age"),
        recommendation_drivers=("Flood exposure",),
    ),
    ComponentSpec(
        key="plumbing", name="Plumbing",
        explanation="Older supply lines and freeze-thaw cycles drive leaks and burst pipes.",
        score=lambda c: c.profile.property_age * 0.5 + c.weather.extreme_temperature_days * 2
        + c.weather.heavy_rainfall_events * 1.0,
        evidence=lambda c: [
            f"{c.profile.property_age}-year-old building (pipe age unknown)",
            f"{c.weather.extreme_temperature_days:g} extreme-temperature days in 12 months",
        ],
        model_features=("property_age", "extreme_temperature_days"),
        recommendation_drivers=("Temperature extremes", "Heavy rainfall"),
    ),
    ComponentSpec(
        key="exterior", name="Exterior walls & siding",
        explanation="Wind-driven debris and wildfire embers attack the building envelope.",
        score=lambda c: c.weather.wind_events * 6 + max(0.0, c.climate.wildfire_trend) * 2.5
        + c.profile.property_age * 0.3,
        evidence=lambda c: [
            f"{c.weather.wind_events:g} notable wind events in 12 months",
            f"Wildfire index trending {c.climate.wildfire_trend:+g} pts per decade",
        ],
        model_features=("wind_events", "wildfire_trend"),
        recommendation_drivers=("Wind exposure", "Wildfire trend"),
    ),
    ComponentSpec(
        key="openings", name="Doors & windows",
        explanation="Entry points for break-ins and wind-driven rain.",
        score=lambda c: c.neighborhood.crime_index * 1.2 + (c.neighborhood.urban_density / 1000) * 2
        + c.weather.wind_events * 2,
        evidence=lambda c: [
            f"Crime index {c.neighborhood.crime_index:g}/100",
            f"Density {c.neighborhood.urban_density:,.0f} people per sq mi",
        ],
        model_features=("crime_index", "urban_density"),
        recommendation_drivers=("Neighborhood crime",),
    ),
    ComponentSpec(
        key="yard", name="Yard & trees",
        explanation="Falling limbs in storms and dry vegetation near the home.",
        score=lambda c: c.weather.wind_events * 8 + max(0.0, c.climate.wildfire_trend) * 2,
        evidence=lambda c: [
            f"{c.weather.wind_events:g} notable wind events in 12 months",
            f"Wildfire index trending {c.climate.wildfire_trend:+g} pts per decade",
        ],
        model_features=("wind_events", "wildfire_trend"),
        recommendation_drivers=("Wind exposure", "Wildfire trend"),
    ),
)


class ComponentRiskAgent(BaseAgent):
    """Breaks property risk down by physical component of the home.

    Component scores are transparent heuristic indices over the agents' observations; the validated
    model's SHAP contributions for the related features are attached so the two can be compared.
    """

    name = "component-risk-agent"

    def run(
        self,
        context: RecommendationContext,
        recommendations: list[Recommendation],
        model_score: ModelScore | None = None,
    ) -> list[ComponentRisk]:
        contributions = {c.feature: c for c in model_score.contributions} if model_score else {}
        results = []
        for spec in COMPONENTS:
            score = bounded_score(spec.score(context))
            results.append(ComponentRisk(
                key=spec.key,
                name=spec.name,
                score=score,
                band=risk_label(score),
                explanation=spec.explanation,
                evidence=spec.evidence(context),
                model_contributions=[contributions[f] for f in spec.model_features if f in contributions],
                recommendations=[r for r in recommendations if r.driver in spec.recommendation_drivers],
            ))
        return results

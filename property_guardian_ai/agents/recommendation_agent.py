from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from property_guardian_ai.agents.base import BaseAgent
from property_guardian_ai.models.schemas import (
    ClimateRisk,
    FloodRisk,
    NeighborhoodRisk,
    Priority,
    PropertyProfile,
    Recommendation,
    ReplacementCostRisk,
    WeatherRisk,
)

PRIORITY_ORDER = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}


@dataclass(frozen=True)
class RecommendationContext:
    profile: PropertyProfile
    weather: WeatherRisk
    flood: FloodRisk
    climate: ClimateRisk
    neighborhood: NeighborhoodRisk
    cost: ReplacementCostRisk


Rule = Callable[[RecommendationContext], Recommendation | None]


def _roof_rule(ctx: RecommendationContext) -> Recommendation | None:
    roof_age = ctx.profile.roof_age
    if roof_age < 15:
        return None
    return Recommendation(
        title="Schedule a roof inspection",
        description="Book a licensed roof inspection; if granules, lifted or missing shingles are found, "
                    "plan shingle replacement before the next storm season.",
        priority=Priority.HIGH if roof_age >= 20 else Priority.MEDIUM,
        driver="Roof age",
        evidence=f"Roof age is {roof_age} years"
                 + (" (estimated from property age)" if ctx.profile.roof_age_estimated else "") + ".",
    )


def _hail_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.weather.hail_frequency < 1 or ctx.profile.roof_age < 10:
        return None
    return Recommendation(
        title="Upgrade to impact-resistant (Class 4) shingles at next re-roof",
        description="Hail on an ageing roof is a leading cause of roof claims; Class 4 shingles reduce hail damage.",
        priority=Priority.MEDIUM,
        driver="Hail exposure",
        evidence=f"{ctx.weather.hail_frequency:g} hail event(s) in the last 12 months on a "
                 f"{ctx.profile.roof_age}-year-old roof.",
    )


def _wind_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.weather.wind_events < 3:
        return None
    return Recommendation(
        title="Trim overhanging trees and secure outdoor equipment",
        description="Cut back limbs within 3 m of the roofline and anchor patio furniture, grills and sheds.",
        priority=Priority.MEDIUM,
        driver="Wind exposure",
        evidence=f"{ctx.weather.wind_events:g} notable wind events in the last 12 months.",
    )


def _flood_rule(ctx: RecommendationContext) -> Recommendation | None:
    high_zone = ctx.flood.flood_zone in {"A", "AE", "VE"}
    near_water = ctx.flood.distance_to_water_km < 1.0
    if not (high_zone or near_water):
        return None
    return Recommendation(
        title="Verify flood zone and install backflow valve / sump pump",
        description="Confirm the FEMA flood map and request an elevation certificate; install a sewer backflow "
                    "valve and battery-backed sump pump.",
        priority=Priority.HIGH if high_zone and near_water else Priority.MEDIUM,
        driver="Flood exposure",
        evidence=f"Flood zone {ctx.flood.flood_zone}, {ctx.flood.distance_to_water_km:g} km from water.",
    )


def _water_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.weather.heavy_rainfall_events < 10:
        return None
    return Recommendation(
        title="Install leak sensors and clear gutters and downspouts",
        description="Place water-leak sensors near the water heater, basement and under sinks; clear gutters and "
                    "extend downspouts at least 1.5 m from the foundation.",
        priority=Priority.MEDIUM,
        driver="Heavy rainfall",
        evidence=f"{ctx.weather.heavy_rainfall_events:g} heavy-rain events in the last 12 months.",
    )


def _freeze_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.weather.extreme_temperature_days < 14:
        return None
    return Recommendation(
        title="Insulate exposed pipes and service HVAC",
        description="Insulate pipes in crawlspaces and exterior walls, and service HVAC before peak season.",
        priority=Priority.LOW,
        driver="Temperature extremes",
        evidence=f"{ctx.weather.extreme_temperature_days:g} extreme-temperature days in the last 12 months.",
    )


def _wildfire_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.climate.wildfire_trend < 8:
        return None
    return Recommendation(
        title="Create defensible space around the home",
        description="Clear dry vegetation within 1.5 m of the structure and use ember-resistant vents.",
        priority=Priority.MEDIUM,
        driver="Wildfire trend",
        evidence=f"Wildfire index trending up {ctx.climate.wildfire_trend:g} pts per decade.",
    )


def _replacement_cost_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.cost.construction_inflation < 4.0:
        return None
    return Recommendation(
        title="Update the dwelling coverage limit (insurance-to-value)",
        description="Re-run the replacement cost estimate; rebuild costs may now exceed the Coverage A limit.",
        priority=Priority.HIGH if ctx.cost.construction_inflation >= 6.0 else Priority.MEDIUM,
        driver="Replacement cost",
        evidence=f"Construction inflation {ctx.cost.construction_inflation:g}% YoY; "
                 f"labor index {ctx.cost.labor_cost_index:g}.",
    )


def _security_rule(ctx: RecommendationContext) -> Recommendation | None:
    if ctx.neighborhood.crime_index < 50:
        return None
    return Recommendation(
        title="Offer a monitored security system discount",
        description="Encourage a monitored alarm and exterior lighting.",
        priority=Priority.LOW,
        driver="Neighborhood crime",
        evidence=f"Crime index {ctx.neighborhood.crime_index:g}/100.",
    )


RULES: tuple[Rule, ...] = (
    _roof_rule, _hail_rule, _wind_rule, _flood_rule, _water_rule,
    _freeze_rule, _wildfire_rule, _replacement_cost_rule, _security_rule,
)


class RecommendationAgent(BaseAgent):
    """Turns observed risk drivers into specific preventive actions. Every action cites its evidence."""

    name = "recommendation-agent"

    def run(self, context: RecommendationContext) -> list[Recommendation]:
        recommendations = [rec for rule in RULES if (rec := rule(context)) is not None]
        return sorted(recommendations, key=lambda rec: PRIORITY_ORDER[rec.priority])

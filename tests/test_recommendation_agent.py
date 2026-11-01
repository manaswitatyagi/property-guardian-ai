from __future__ import annotations

from property_guardian_ai.agents.climate_agent import ClimateAgent
from property_guardian_ai.agents.flood_agent import FloodAgent
from property_guardian_ai.agents.inflation_agent import InflationAgent
from property_guardian_ai.agents.neighborhood_agent import NeighborhoodAgent
from property_guardian_ai.agents.recommendation_agent import RecommendationAgent, RecommendationContext
from property_guardian_ai.agents.weather_agent import WeatherAgent
from property_guardian_ai.models.schemas import FloodObservation, PropertyRecord, Priority
from tests.conftest import make_record, profile_for


def _recommend(record: PropertyRecord):
    profile = profile_for(record)
    context = RecommendationContext(
        profile=profile,
        weather=WeatherAgent().run(profile, record.weather),
        flood=FloodAgent().run(profile, record.flood),
        climate=ClimateAgent().run(profile, record.climate),
        neighborhood=NeighborhoodAgent().run(profile, record.neighborhood),
        cost=InflationAgent().run(profile, record.cost),
    )
    return RecommendationAgent().run(context)


def test_low_risk_property_gets_no_actions() -> None:
    assert _recommend(make_record()) == []


def test_old_roof_in_flood_zone_gets_high_priority_actions_first() -> None:
    base = make_record()
    record = make_record(
        property=base.property.model_copy(update={"roof_age": 24}),
        flood=FloodObservation(flood_zone="AE", distance_to_water_km=0.4),
    )
    recommendations = _recommend(record)
    titles = [r.title for r in recommendations]

    assert "Schedule a roof inspection" in titles
    assert any("flood" in t.lower() for t in titles)
    assert recommendations[0].priority == Priority.HIGH
    assert all(r.evidence for r in recommendations)

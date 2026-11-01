from __future__ import annotations

from property_guardian_ai.agents.climate_agent import ClimateAgent
from property_guardian_ai.agents.flood_agent import FloodAgent
from property_guardian_ai.agents.inflation_agent import InflationAgent
from property_guardian_ai.agents.neighborhood_agent import NeighborhoodAgent
from property_guardian_ai.agents.property_agent import PropertyAgent
from property_guardian_ai.agents.weather_agent import WeatherAgent
from property_guardian_ai.models.schemas import FloodObservation, WeatherObservation
from tests.conftest import make_record, profile_for


def test_property_agent_uses_as_of_date_and_flags_estimated_roof() -> None:
    record = make_record()
    known = PropertyAgent().run(record.property)
    estimated = PropertyAgent().run(record.property.model_copy(update={"roof_age": None}))

    assert known.property_age == 36
    assert known.roof_age == 8 and not known.roof_age_estimated
    assert estimated.roof_age == 18 and estimated.roof_age_estimated


def test_flood_agent_ranks_high_risk_zone_near_water_above_zone_x() -> None:
    profile = profile_for(make_record())
    agent = FloodAgent()
    low = agent.run(profile, FloodObservation(flood_zone="X", distance_to_water_km=5.0))
    high = agent.run(profile, FloodObservation(flood_zone="AE", distance_to_water_km=0.2))

    assert high.risk_score > low.risk_score
    assert high.historical_flooding is None


def test_weather_agent_annualises_short_windows() -> None:
    profile = profile_for(make_record())
    yearly = WeatherObservation(hail_events=2, wind_events=4, heavy_rainfall_events=8, extreme_temperature_days=6)
    half_year = yearly.model_copy(update={
        "hail_events": 1, "wind_events": 2, "heavy_rainfall_events": 4,
        "extreme_temperature_days": 3, "window_months": 6,
    })

    assert WeatherAgent().run(profile, yearly).risk_score == WeatherAgent().run(profile, half_year).risk_score


def test_all_hazard_scores_stay_within_bounds(services) -> None:
    for property_id in services.repository.property_ids():
        record = services.repository.get(property_id)
        profile = profile_for(record)
        scores = [
            WeatherAgent().run(profile, record.weather).risk_score,
            FloodAgent().run(profile, record.flood).risk_score,
            ClimateAgent().run(profile, record.climate).risk_score,
            NeighborhoodAgent().run(profile, record.neighborhood).risk_score,
            InflationAgent().run(profile, record.cost).risk_score,
        ]
        assert all(0 <= s <= 100 for s in scores)

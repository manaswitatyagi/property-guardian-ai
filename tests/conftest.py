from __future__ import annotations

from datetime import date

import pytest

from property_guardian_ai.agents.property_agent import PropertyAgent
from property_guardian_ai.models.schemas import (
    ClimateObservation,
    CostObservation,
    FloodObservation,
    NeighborhoodObservation,
    PropertyInput,
    PropertyProfile,
    PropertyRecord,
    WeatherObservation,
)
from property_guardian_ai.services.container import Services, build_services


@pytest.fixture(scope="session")
def services() -> Services:
    return build_services()


def make_record(**overrides: object) -> PropertyRecord:
    parts = {
        "property": PropertyInput(
            property_id="T-1", address="Test property", latitude=47.5, longitude=-122.3,
            year_built=1990, home_value=500_000, square_footage=1800, roof_age=8, as_of=date(2026, 10, 1),
        ),
        "weather": WeatherObservation(hail_events=0, wind_events=1, heavy_rainfall_events=4,
                                      extreme_temperature_days=5),
        "flood": FloodObservation(flood_zone="X", distance_to_water_km=4.0),
        "climate": ClimateObservation(heat_trend_c_per_decade=0.3, precipitation_trend_pct_per_decade=3,
                                      wildfire_trend_index_pts_per_decade=2),
        "neighborhood": NeighborhoodObservation(population_trend_pct_5yr=1, crime_index=20,
                                                urban_density_per_sqmi=2000),
        "cost": CostObservation(construction_inflation_yoy_pct=3.0, labor_cost_index=110),
    }
    parts.update(overrides)
    return PropertyRecord(**parts)


def profile_for(record: PropertyRecord) -> PropertyProfile:
    return PropertyAgent().run(record.property)

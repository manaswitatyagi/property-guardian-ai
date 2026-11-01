from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from property_guardian_ai.models.schemas import (
    ClimateObservation,
    CostObservation,
    FloodObservation,
    MonthlyObservation,
    NeighborhoodObservation,
    PropertyInput,
    PropertyRecord,
    Provenance,
    WeatherObservation,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROPERTIES_FILE = "underwriting_properties_10.csv"
MONTHLY_FILE = "underwriting_monthly_risk_120.csv"


class PropertyNotFoundError(KeyError):
    pass


def _month_end_as_of(months: pd.Series) -> date:
    return pd.to_datetime(months, format="%Y-%m").max().date()


def _to_record(row: pd.Series, as_of: date) -> PropertyRecord:
    """Map one demo CSV row to typed observations. Provenance follows the field dictionary."""
    return PropertyRecord(
        property=PropertyInput(
            property_id=row["property_id"],
            address=row["address"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            year_built=int(row["year_built"]),
            home_value=float(row["home_value_usd"]),
            square_footage=int(row["square_footage"]),
            roof_age=int(row["roof_age_years"]),
            as_of=as_of,
        ),
        weather=WeatherObservation(
            hail_events=row["hail_events_12mo"],
            wind_events=row["wind_events_12mo"],
            heavy_rainfall_events=row["heavy_rain_events_12mo"],
            extreme_temperature_days=row["extreme_temperature_days_12mo"],
        ),
        # The demo dataset has no flood-claims history, so it stays unknown rather than assumed.
        flood=FloodObservation(
            flood_zone=row["flood_zone"],
            distance_to_water_km=row["distance_to_water_km"],
            historical_flooding=None,
        ),
        climate=ClimateObservation(
            heat_trend_c_per_decade=row["heat_trend_c_per_decade"],
            precipitation_trend_pct_per_decade=row["precipitation_trend_pct_per_decade"],
            wildfire_trend_index_pts_per_decade=row["wildfire_trend_index_pts_per_decade"],
        ),
        neighborhood=NeighborhoodObservation(
            population_trend_pct_5yr=row["population_trend_pct_5yr"],
            crime_index=row["crime_index_0_100"],
            urban_density_per_sqmi=row["urban_density_people_per_sqmi"],
        ),
        cost=CostObservation(
            construction_inflation_yoy_pct=row["construction_inflation_yoy_pct"],
            labor_cost_index=row["labor_cost_index_base100"],
        ),
    )


def _to_monthly(row: pd.Series) -> MonthlyObservation:
    return MonthlyObservation(
        property_id=row["property_id"],
        month=pd.to_datetime(row["month"], format="%Y-%m").date(),
        hail_events=row["hail_events"],
        wind_events=row["wind_events"],
        heavy_rain_events=row["heavy_rain_events"],
        extreme_temperature_days=row["extreme_temperature_days"],
        seasonal_climate_adjustment=row["seasonal_climate_adjustment_pts"],
        observed_risk_score=row["monthly_risk_score_0_100"],
    )


@dataclass(frozen=True)
class PropertyRepository:
    """Read-only access to property records and their monthly observations."""

    records: dict[str, PropertyRecord]
    monthly: dict[str, list[MonthlyObservation]]

    @classmethod
    def from_csv(cls, data_dir: Path = DEFAULT_DATA_DIR) -> PropertyRepository:
        properties = pd.read_csv(data_dir / PROPERTIES_FILE)
        monthly = pd.read_csv(data_dir / MONTHLY_FILE)
        as_of = _month_end_as_of(monthly["month"])

        records = {row["property_id"]: _to_record(row, as_of) for _, row in properties.iterrows()}
        observations: dict[str, list[MonthlyObservation]] = {pid: [] for pid in records}
        for _, row in monthly.sort_values(["property_id", "month"]).iterrows():
            observations.setdefault(row["property_id"], []).append(_to_monthly(row))
        return cls(records=records, monthly=observations)

    def property_ids(self) -> list[str]:
        return sorted(self.records)

    def get(self, property_id: str) -> PropertyRecord:
        if property_id not in self.records:
            raise PropertyNotFoundError(property_id)
        return self.records[property_id]

    def monthly_for(self, property_id: str) -> list[MonthlyObservation]:
        self.get(property_id)
        return self.monthly.get(property_id, [])

    def latest_month(self, property_id: str) -> MonthlyObservation | None:
        history = self.monthly_for(property_id)
        return history[-1] if history else None


def all_observations_simulated(record: PropertyRecord) -> bool:
    observations = [record.weather, record.flood, record.climate, record.neighborhood, record.cost]
    return all(obs.provenance == Provenance.SIMULATED for obs in observations)

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Provenance(StrEnum):
    """Where a value came from, so explanations never present simulated data as measured."""

    MEASURED = "measured"
    HISTORICAL_RECORD = "historical_record"
    SIMULATED = "simulated"
    DERIVED = "derived"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Agent inputs (observations)
# ---------------------------------------------------------------------------


class PropertyInput(FrozenModel):
    property_id: str
    address: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    year_built: int = Field(ge=1700)
    home_value: float = Field(ge=0)
    square_footage: int = Field(gt=0)
    roof_age: int | None = Field(default=None, ge=0)
    as_of: date
    home_value_provenance: Provenance = Provenance.SIMULATED
    roof_age_provenance: Provenance = Provenance.SIMULATED


class WeatherObservation(FrozenModel):
    """Event counts over a trailing observation window (NOAA Storm Events in production)."""

    hail_events: float = Field(ge=0)
    wind_events: float = Field(ge=0)
    heavy_rainfall_events: float = Field(ge=0)
    extreme_temperature_days: float = Field(ge=0)
    window_months: int = Field(default=12, gt=0)
    provenance: Provenance = Provenance.SIMULATED


class FloodObservation(FrozenModel):
    """FEMA-style flood inputs. `historical_flooding` is None when unknown."""

    flood_zone: str
    distance_to_water_km: float = Field(ge=0)
    historical_flooding: bool | None = None
    provenance: Provenance = Provenance.SIMULATED


class ClimateObservation(FrozenModel):
    heat_trend_c_per_decade: float
    precipitation_trend_pct_per_decade: float
    wildfire_trend_index_pts_per_decade: float
    provenance: Provenance = Provenance.SIMULATED


class NeighborhoodObservation(FrozenModel):
    population_trend_pct_5yr: float
    crime_index: float = Field(ge=0, le=100)
    urban_density_per_sqmi: float = Field(ge=0)
    provenance: Provenance = Provenance.SIMULATED


class CostObservation(FrozenModel):
    construction_inflation_yoy_pct: float
    labor_cost_index: float = Field(gt=0)
    provenance: Provenance = Provenance.SIMULATED


class PropertyRecord(FrozenModel):
    """Everything the system knows about one property at one point in time."""

    property: PropertyInput
    weather: WeatherObservation
    flood: FloodObservation
    climate: ClimateObservation
    neighborhood: NeighborhoodObservation
    cost: CostObservation


class MonthlyObservation(FrozenModel):
    property_id: str
    month: date
    hail_events: float = Field(ge=0)
    wind_events: float = Field(ge=0)
    heavy_rain_events: float = Field(ge=0)
    extreme_temperature_days: float = Field(ge=0)
    seasonal_climate_adjustment: float
    observed_risk_score: float = Field(ge=0, le=100)
    provenance: Provenance = Provenance.SIMULATED


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------


class PropertyProfile(FrozenModel):
    property_id: str
    address: str
    latitude: float
    longitude: float
    year_built: int
    property_age: int
    roof_age: int
    roof_age_estimated: bool
    home_value: float
    square_footage: int
    as_of: date


class WeatherRisk(FrozenModel):
    hail_frequency: float = Field(ge=0)
    wind_events: float = Field(ge=0)
    heavy_rainfall_events: float = Field(ge=0)
    extreme_temperature_days: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=100)
    provenance: Provenance


class FloodRisk(FrozenModel):
    flood_zone: str
    historical_flooding: bool | None
    distance_to_water_km: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=100)
    provenance: Provenance


class ClimateRisk(FrozenModel):
    heat_trend: float
    precipitation_trend: float
    wildfire_trend: float
    risk_score: float = Field(ge=0, le=100)
    provenance: Provenance


class NeighborhoodRisk(FrozenModel):
    population_trend: float
    crime_index: float = Field(ge=0)
    urban_density: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=100)
    provenance: Provenance


class ReplacementCostRisk(FrozenModel):
    construction_inflation: float
    labor_cost_index: float
    risk_score: float = Field(ge=0, le=100)
    provenance: Provenance


class FeatureContribution(FrozenModel):
    """One feature's SHAP contribution to a model score, in score points."""

    feature: str
    label: str
    domain: str
    value: float
    contribution: float
    provenance: Provenance


class ModelScore(FrozenModel):
    model_name: str
    score: float = Field(ge=0, le=100)
    base_value: float
    contributions: list[FeatureContribution]
    holdout_mae: float = Field(ge=0)


class ValidationReport(FrozenModel):
    """Grouped (by property) cross-validation results; nothing here is seen in training."""

    model_name: str
    n_rows: int
    n_properties: int
    n_folds: int
    holdout_mae: float
    holdout_r2: float
    high_risk_threshold: float
    roc_auc: float | None
    precision: float | None
    recall: float | None
    baseline_model_name: str
    baseline_holdout_mae: float
    trained_on_simulated_data: bool
    caveats: list[str]


class RiskDriver(FrozenModel):
    name: str
    score: float = Field(ge=0, le=100)
    explanation: str
    evidence: list[str] = Field(default_factory=list)


class Recommendation(FrozenModel):
    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    driver: str
    evidence: str


class FeatureChange(FrozenModel):
    feature: str
    label: str
    previous_value: float
    current_value: float
    contribution_change: float


class RiskChangeExplanation(FrozenModel):
    property_id: str
    previous_month: date
    current_month: date
    previous_score: float
    current_score: float
    score_change: float
    direction: str
    top_changes: list[FeatureChange]
    narrative: str


class ComponentRisk(FrozenModel):
    """Risk to one physical part of the home, with the evidence and actions that apply to it."""

    key: str
    name: str
    score: float = Field(ge=0, le=100)
    band: str
    explanation: str
    evidence: list[str]
    model_contributions: list[FeatureContribution] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


class Confidence(FrozenModel):
    score: float = Field(ge=0, le=100)
    level: ConfidenceLevel
    reasons: list[str]


class RiskAssessment(FrozenModel):
    property_profile: PropertyProfile
    weather_risk: WeatherRisk
    flood_risk: FloodRisk
    climate_risk: ClimateRisk
    neighborhood_risk: NeighborhoodRisk
    replacement_cost_risk: ReplacementCostRisk
    risk_score: float = Field(ge=0, le=100)
    risk_label: str
    scoring_method: str
    confidence: Confidence
    summary: str
    primary_drivers: list[RiskDriver]
    model_explanation: ModelScore | None = None
    risk_change: RiskChangeExplanation | None = None
    recommendations: list[Recommendation]
    component_risks: list[ComponentRisk] = Field(default_factory=list)


class RiskSnapshot(FrozenModel):
    """A point-in-time record kept by the Memory Agent."""

    property_id: str
    month: date
    risk_score: float = Field(ge=0, le=100)
    features: dict[str, float]
    contributions: dict[str, float]
    recommendations: list[Recommendation] = Field(default_factory=list)

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from property_guardian_ai.models.schemas import (
    FeatureContribution,
    ModelScore,
    MonthlyObservation,
    PropertyRecord,
    Provenance,
    ValidationReport,
)
from property_guardian_ai.services.property_repository import PropertyRepository

logger = logging.getLogger(__name__)

MODEL_NAME = "xgboost-monthly-risk-v1"
BASELINE_NAME = "random-forest-monthly-risk-v1"
HIGH_RISK_THRESHOLD = 60.0
N_FOLDS = 5
RANDOM_STATE = 42

FLOOD_ZONE_CODES: dict[str, float] = {"X": 0, "X_SHADED": 1, "A": 2, "AE": 3, "VE": 4}

Extractor = Callable[[PropertyRecord, MonthlyObservation], float]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    domain: str
    extract: Extractor
    provenance: Provenance = Provenance.SIMULATED


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("property_age", "Property age (years)", "property",
                lambda r, m: r.property.as_of.year - r.property.year_built, Provenance.HISTORICAL_RECORD),
    FeatureSpec("roof_age", "Roof age (years)", "property", lambda r, m: float(r.property.roof_age or 0)),
    FeatureSpec("flood_zone_code", "Flood zone (X=0 … VE=4)", "flood",
                lambda r, m: FLOOD_ZONE_CODES.get(r.flood.flood_zone, 2)),
    FeatureSpec("distance_to_water_km", "Distance to water (km)", "flood", lambda r, m: r.flood.distance_to_water_km),
    FeatureSpec("heat_trend", "Heat trend (°C/decade)", "climate", lambda r, m: r.climate.heat_trend_c_per_decade),
    FeatureSpec("precipitation_trend", "Precipitation trend (%/decade)", "climate",
                lambda r, m: r.climate.precipitation_trend_pct_per_decade),
    FeatureSpec("wildfire_trend", "Wildfire trend (index pts/decade)", "climate",
                lambda r, m: r.climate.wildfire_trend_index_pts_per_decade),
    FeatureSpec("population_trend", "Population trend (% / 5yr)", "neighborhood",
                lambda r, m: r.neighborhood.population_trend_pct_5yr),
    FeatureSpec("crime_index", "Crime index (0-100)", "neighborhood", lambda r, m: r.neighborhood.crime_index),
    FeatureSpec("urban_density", "Urban density (people/sq mi)", "neighborhood",
                lambda r, m: r.neighborhood.urban_density_per_sqmi),
    FeatureSpec("construction_inflation", "Construction inflation (% YoY)", "cost",
                lambda r, m: r.cost.construction_inflation_yoy_pct),
    FeatureSpec("labor_cost_index", "Labor cost index (base 100)", "cost", lambda r, m: r.cost.labor_cost_index),
    FeatureSpec("hail_events", "Hail events this month", "weather", lambda r, m: m.hail_events),
    FeatureSpec("wind_events", "Wind events this month", "weather", lambda r, m: m.wind_events),
    FeatureSpec("heavy_rain_events", "Heavy-rain events this month", "weather", lambda r, m: m.heavy_rain_events),
    FeatureSpec("extreme_temperature_days", "Extreme-temperature days this month", "weather",
                lambda r, m: m.extreme_temperature_days),
    FeatureSpec("seasonal_climate_adjustment", "Seasonal climate adjustment (pts)", "climate",
                lambda r, m: m.seasonal_climate_adjustment),
)
FEATURE_NAMES: list[str] = [spec.name for spec in FEATURES]
FEATURE_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURES}


def feature_vector(record: PropertyRecord, month: MonthlyObservation) -> dict[str, float]:
    return {spec.name: float(spec.extract(record, month)) for spec in FEATURES}


def build_training_frame(repository: PropertyRepository) -> pd.DataFrame:
    rows = [
        {"property_id": pid, "month": obs.month, "target": obs.observed_risk_score, **feature_vector(record, obs)}
        for pid, record in repository.records.items()
        for obs in repository.monthly_for(pid)
    ]
    return pd.DataFrame(rows)


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=1,
    )


def _new_baseline() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=1)


def _out_of_fold(make_model: Callable[[], object], x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    predictions = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(x, y, groups):
        model = make_model()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(x.iloc[test_idx])
    return np.clip(predictions, 0, 100)


def _classification_metrics(y: pd.Series, predicted: np.ndarray) -> tuple[float | None, float | None, float | None]:
    actual_high = (y >= HIGH_RISK_THRESHOLD).astype(int)
    if actual_high.nunique() < 2:
        return None, None, None
    predicted_high = (predicted >= HIGH_RISK_THRESHOLD).astype(int)
    return (
        round(float(roc_auc_score(actual_high, predicted)), 3),
        round(float(precision_score(actual_high, predicted_high, zero_division=0)), 3),
        round(float(recall_score(actual_high, predicted_high, zero_division=0)), 3),
    )


@dataclass(frozen=True)
class RiskModel:
    """XGBoost risk scorer with SHAP attributions and a grouped-holdout validation report."""

    model: XGBRegressor
    explainer: shap.TreeExplainer
    validation: ValidationReport

    @classmethod
    def train(cls, repository: PropertyRepository) -> RiskModel:
        frame = build_training_frame(repository)
        x, y, groups = frame[FEATURE_NAMES], frame["target"], frame["property_id"]

        oof = _out_of_fold(_new_model, x, y, groups)
        baseline_oof = _out_of_fold(_new_baseline, x, y, groups)
        roc_auc, precision, recall = _classification_metrics(y, oof)
        validation = ValidationReport(
            model_name=MODEL_NAME,
            n_rows=len(frame),
            n_properties=groups.nunique(),
            n_folds=N_FOLDS,
            holdout_mae=round(float(mean_absolute_error(y, oof)), 2),
            holdout_r2=round(float(r2_score(y, oof)), 3),
            high_risk_threshold=HIGH_RISK_THRESHOLD,
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            baseline_model_name=BASELINE_NAME,
            baseline_holdout_mae=round(float(mean_absolute_error(y, baseline_oof)), 2),
            trained_on_simulated_data=True,
            caveats=[
                "Target is the dataset's synthetic risk score, not observed losses or claims.",
                f"Only {groups.nunique()} properties: holdout metrics have very wide uncertainty.",
                "Folds are grouped by property so no property appears in both train and test.",
            ],
        )

        model = _new_model().fit(x, y)
        logger.info("risk_model_trained", extra={"model": MODEL_NAME, **validation.model_dump(exclude={"caveats"})})
        return cls(model=model, explainer=shap.TreeExplainer(model), validation=validation)

    def score(self, record: PropertyRecord, month: MonthlyObservation) -> ModelScore:
        features = feature_vector(record, month)
        x = pd.DataFrame([features], columns=FEATURE_NAMES)
        shap_values = self.explainer.shap_values(x)[0]
        prediction = float(self.model.predict(x)[0])

        contributions = sorted(
            (
                FeatureContribution(
                    feature=name,
                    label=FEATURE_BY_NAME[name].label,
                    domain=FEATURE_BY_NAME[name].domain,
                    value=features[name],
                    contribution=round(float(value), 2),
                    provenance=FEATURE_BY_NAME[name].provenance,
                )
                for name, value in zip(FEATURE_NAMES, shap_values, strict=True)
            ),
            key=lambda c: abs(c.contribution),
            reverse=True,
        )
        return ModelScore(
            model_name=MODEL_NAME,
            score=round(float(np.clip(prediction, 0, 100)), 2),
            base_value=round(float(np.ravel(self.explainer.expected_value)[0]), 2),
            contributions=contributions,
            holdout_mae=self.validation.holdout_mae,
        )

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _compute_monitoring_scores(properties: pd.DataFrame) -> pd.DataFrame:
    df = properties.copy()

    df["neighborhood_score"] = (
        (df["crime_index_0_100"] * 0.45)
        + (df["urban_density_people_per_sqmi"] / 100.0) * 1.4
        + (df["population_trend_pct_5yr"].abs() * 12.0)
        + (df["wildfire_trend_index_pts_per_decade"] * 0.9)
    )

    df["account_activity_score"] = (
        (df["risk_score_0_100"] * 0.55)
        + (df["roof_age_years"] * 0.8)
        + (df["home_value_usd"] / 10000.0) * 0.12
    )

    df["device_activity_score"] = (
        (df["hail_events_12mo"] * 8.0)
        + (df["wind_events_12mo"] * 6.0)
        + (df["heavy_rain_events_12mo"] * 7.0)
        + (df["extreme_temperature_days_12mo"] * 0.9)
        + (df["roof_age_years"] * 0.6)
    )

    df["financial_pressure_score"] = (
        (df["construction_inflation_yoy_pct"] * 6.4)
        + ((df["labor_cost_index_base100"] - 100.0) * 0.55)
        + (df["home_value_usd"] / 50000.0) * 0.5
    )

    df["monitoring_score"] = (
        df["neighborhood_score"] * 0.25
        + df["account_activity_score"] * 0.25
        + df["device_activity_score"] * 0.25
        + df["financial_pressure_score"] * 0.25
    )

    for col in [
        "neighborhood_score",
        "account_activity_score",
        "device_activity_score",
        "financial_pressure_score",
        "monitoring_score",
    ]:
        df[col] = df[col].apply(_clamp)

    df["monthly_change_pts"] = df["risk_score_0_100"] - df["risk_score_0_100"].mean()
    return df


def load_demo_data() -> dict[str, pd.DataFrame]:
    """Load the demo underwriting dataset shipped with the repository."""
    root = _project_root()
    data_dir = root / "data"

    properties = pd.read_csv(data_dir / "underwriting_properties_10.csv")
    monthly_risk = pd.read_csv(data_dir / "underwriting_monthly_risk_120.csv")

    properties["risk_score_0_100"] = pd.to_numeric(properties["risk_score_0_100"], errors="coerce")
    monthly_risk["month"] = pd.to_datetime(monthly_risk["month"], format="%Y-%m")
    monthly_risk["monthly_risk_score_0_100"] = pd.to_numeric(
        monthly_risk["monthly_risk_score_0_100"], errors="coerce"
    )

    properties = _compute_monitoring_scores(properties)
    return {"properties": properties, "monthly_risk": monthly_risk}


def _score_status(score: float) -> str:
    if score < 45:
        return "Low"
    if score < 75:
        return "Moderate"
    return "High"


def build_component_risk(property_row: pd.Series) -> dict[str, dict[str, object]]:
    flood_zone_multiplier = {"AE": 1.4, "X_SHADED": 0.8, "X": 0.4}
    zone_factor = flood_zone_multiplier.get(property_row["flood_zone"], 0.6)
    pool_present = property_row["square_footage"] > 3200 or property_row["home_value_usd"] > 1000000

    porch_score = _clamp(
        property_row["crime_index_0_100"] * 0.25
        + property_row["urban_density_people_per_sqmi"] / 80
        + property_row["heavy_rain_events_12mo"] * 3.5
    )
    roof_score = _clamp(
        property_row["roof_age_years"] * 2.8
        + property_row["hail_events_12mo"] * 5
        + property_row["wind_events_12mo"] * 4
    )
    plumbing_score = _clamp(
        property_row["heavy_rain_events_12mo"] * 4.2
        + property_row["distance_to_water_km"] * 18
        + property_row["roof_age_years"] * 0.8
    )
    structure_score = _clamp(
        (2026 - property_row["year_built"]) * 0.9
        + property_row["distance_to_water_km"] * 18
        + zone_factor * 12
    )
    living_score = _clamp(
        property_row["extreme_temperature_days_12mo"] * 0.9
        + abs(property_row["population_trend_pct_5yr"]) * 8
        + property_row["square_footage"] / 25
    )
    bathtub_score = _clamp(
        property_row["heavy_rain_events_12mo"] * 5.2
        + property_row["distance_to_water_km"] * 12
        + property_row["roof_age_years"] * 0.9
    )
    pool_score = 8 if not pool_present else _clamp(
        25 + property_row["wildfire_trend_index_pts_per_decade"] * 0.9 + property_row["distance_to_water_km"] * 10
    )

    component_map = {
        "Porch": {
            "score": porch_score,
            "status": _score_status(porch_score),
            "detail": "Review decking condition, drainage, and entry exposure to weather or intrusion.",
            "risk_factors": [
                f"Front exposure is elevated by a crime index of {property_row['crime_index_0_100']:.1f}.",
                f"Urban density contributes {property_row['urban_density_people_per_sqmi']:.0f} people per square mile to the risk profile.",
                f"Recent heavy rain events are driving water and wear on the porch area.",
            ],
            "mitigation": [
                "Seal or replace worn porch decking and railings.",
                "Improve drainage around the entry to prevent standing water.",
                "Add exterior lighting and deterrents for intrusion risk.",
            ],
        },
        "Roof": {
            "score": roof_score,
            "status": _score_status(roof_score),
            "detail": "Roof age and repeated wind or hail activity increase the chance of weather-related degradation.",
            "risk_factors": [
                f"Roof age is {property_row['roof_age_years']:.1f} years, which raises weather exposure risk.",
                f"Recent hail activity is adding {property_row['hail_events_12mo']:.0f} seasonal impact events.",
                f"Wind exposure is elevated with {property_row['wind_events_12mo']:.0f} significant wind events in the last year.",
            ],
            "mitigation": [
                "Inspect shingles or tiles for lifting, cracking, or flashing failures.",
                "Schedule a post-storm roof inspection after major wind or hail events.",
                "Replace aged roofing materials before leak-related damage increases.",
            ],
        },
        "Plumbing": {
            "score": plumbing_score,
            "status": _score_status(plumbing_score),
            "detail": "Water intrusion, flow stress, and older fixtures can increase leak and pressure risk.",
            "risk_factors": [
                f"Heavy rainfall exposure is occurring {property_row['heavy_rain_events_12mo']:.0f} times in the last year.",
                f"The property is about {property_row['distance_to_water_km']:.1f} km from water, which can raise moisture stress.",
                f"Roof age and broader moisture loading can increase leak pressure on plumbing infrastructure.",
            ],
            "mitigation": [
                "Check for hidden leaks under sinks, behind toilets, and in utility areas.",
                "Upgrade aging fixtures and pressure valves before seasonal storms hit.",
                "Add leak sensors to high-risk zones and water shutoff controls.",
            ],
        },
        "Structure": {
            "score": structure_score,
            "status": _score_status(structure_score),
            "detail": "Age, flood proximity, and long-term climate stress can pressure the structural envelope.",
            "risk_factors": [
                f"The home is {2026 - property_row['year_built']:.0f} years old, which increases maintenance demand.",
                f"Flood exposure is relevant at a distance of {property_row['distance_to_water_km']:.1f} km from water.",
                f"The flood-zone factor is {zone_factor:.1f}, increasing envelope stress under wet conditions.",
            ],
            "mitigation": [
                "Increase inspection frequency on foundation, framing, and drainage systems.",
                "Improve grading and water diversion away from the foundation.",
                "Prioritize preventive repairs before climate or flood stress compounds.",
            ],
        },
        "Living area": {
            "score": living_score,
            "status": _score_status(living_score),
            "detail": "Thermal stress and occupancy changes may increase wear on conditioned living zones.",
            "risk_factors": [
                f"The property experienced {property_row['extreme_temperature_days_12mo']:.0f} extreme-temperature days.",
                f"Population trend volatility is {abs(property_row['population_trend_pct_5yr']):.1f}% over five years.",
                f"The home footprint is {property_row['square_footage']:.0f} square feet, which increases climate load.",
            ],
            "mitigation": [
                "Review HVAC performance and insulation quality in major living zones.",
                "Tighten maintenance on windows, seals, and weather stripping.",
                "Improve thermal resilience with upgrades to climate control and ventilation.",
            ],
        },
        "Bathtub": {
            "score": bathtub_score,
            "status": _score_status(bathtub_score),
            "detail": "Water exposure and moisture persistence can produce slip, leakage, and finish deterioration risk.",
            "risk_factors": [
                f"Moisture pressure is elevated by {property_row['heavy_rain_events_12mo']:.0f} heavy rain events.",
                f"Nearby water proximity adds elevated dampness risk at {property_row['distance_to_water_km']:.1f} km.",
                f"Aging roof conditions can increase moisture intrusion behind wet areas.",
            ],
            "mitigation": [
                "Inspect caulking, grout, drainage, and finish integrity.",
                "Use anti-slip and water-resistant materials where moisture is chronic.",
                "Check for hidden penetration that could lead to subfloor or wall damage.",
            ],
        },
        "Swimming pool": {
            "score": pool_score,
            "status": _score_status(pool_score),
            "detail": "No pool was detected in this demo profile." if not pool_present else "The pool area should be checked for compliance, safety, and water management risk.",
            "risk_factors": [
                "The property does not present a pool risk in this demo because no pool is identified." if not pool_present else f"Pool-related exposure is amplified by wildfire risk index of {property_row['wildfire_trend_index_pts_per_decade']:.1f}.",
                "Water management and safety checks are important around the pool zone." if pool_present else "The pool asset remains low risk in this profile.",
            ],
            "mitigation": [
                "If present, verify fencing, safety barriers, and equipment maintenance.",
                "Inspect pool circulation and water chemistry to reduce damage and liability risk.",
                "Check drainage and hardscape conditions to prevent structural or slip issues.",
            ],
        },
    }

    return component_map


def build_monitoring_narrative(property_row: pd.Series, monthly_rows: pd.DataFrame) -> str:
    latest = monthly_rows.sort_values("month").iloc[-1]
    previous = monthly_rows.sort_values("month").iloc[-2]
    delta = latest["monthly_risk_score_0_100"] - previous["monthly_risk_score_0_100"]

    if delta > 5:
        trend = "increased"
    elif delta < -5:
        trend = "fell"
    else:
        trend = "remained stable"

    summary = (
        f"This property's monthly risk {trend} by {abs(delta):.1f} points over the last month, "
        f"with a neighborhood score of {property_row['neighborhood_score']:.1f}, account activity of {property_row['account_activity_score']:.1f}, "
        f"device exposure of {property_row['device_activity_score']:.1f}, and financial pressure of {property_row['financial_pressure_score']:.1f}. "
        f"The main pressure is concentrated around {latest['dominant_risk_driver']} and the property profile suggests a need for closer monitoring."
    )
    return summary


def get_property_summary(property_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    demo_data = load_demo_data()
    property_row = demo_data["properties"][demo_data["properties"]["property_id"] == property_id]
    monthly_rows = demo_data["monthly_risk"][demo_data["monthly_risk"]["property_id"] == property_id].sort_values("month")
    return property_row, monthly_rows

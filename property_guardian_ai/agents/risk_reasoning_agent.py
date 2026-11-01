from __future__ import annotations

from dataclasses import dataclass

from property_guardian_ai.agents.base import BaseAgent
from property_guardian_ai.models.schemas import (
    ClimateRisk,
    ComponentRisk,
    Confidence,
    ConfidenceLevel,
    FeatureContribution,
    FloodRisk,
    ModelScore,
    NeighborhoodRisk,
    Priority,
    PropertyProfile,
    Provenance,
    Recommendation,
    ReplacementCostRisk,
    RiskAssessment,
    RiskChangeExplanation,
    RiskDriver,
    WeatherRisk,
)

# Used only when no validated model is available.
HEURISTIC_WEIGHTS: dict[str, float] = {
    "weather": 0.30, "flood": 0.28, "climate": 0.17, "neighborhood": 0.10, "cost": 0.10, "property": 0.05,
}
DOMAIN_NAMES: dict[str, str] = {
    "weather": "Weather exposure",
    "flood": "Flood exposure",
    "climate": "Climate trend",
    "neighborhood": "Neighborhood conditions",
    "cost": "Replacement cost pressure",
    "property": "Property condition",
}
DOMAIN_EXPLANATIONS: dict[str, str] = {
    "weather": "Recent storm, rain and temperature events raise the chance of roof and envelope damage.",
    "flood": "Flood zone designation and proximity to water drive inundation exposure.",
    "climate": "Long-term heat, precipitation and wildfire trends increase loss frequency.",
    "neighborhood": "Population change, crime and density affect theft, vandalism and fire spread.",
    "cost": "Construction and labor inflation raise rebuild cost relative to coverage.",
    "property": "Building and roof age increase vulnerability to every peril.",
}
MAX_DRIVERS = 3


@dataclass(frozen=True)
class ReasoningInputs:
    profile: PropertyProfile
    weather: WeatherRisk
    flood: FloodRisk
    climate: ClimateRisk
    neighborhood: NeighborhoodRisk
    cost: ReplacementCostRisk
    recommendations: list[Recommendation]
    model_score: ModelScore | None = None
    risk_change: RiskChangeExplanation | None = None
    component_risks: tuple[ComponentRisk, ...] = ()

    def component_scores(self) -> dict[str, float]:
        age_score = min(100.0, self.profile.property_age * 0.6 + self.profile.roof_age * 2.0)
        return {
            "weather": self.weather.risk_score,
            "flood": self.flood.risk_score,
            "climate": self.climate.risk_score,
            "neighborhood": self.neighborhood.risk_score,
            "cost": self.cost.risk_score,
            "property": round(age_score, 2),
        }

    def provenances(self) -> list[Provenance]:
        return [self.weather.provenance, self.flood.provenance, self.climate.provenance,
                self.neighborhood.provenance, self.cost.provenance]


def risk_label(score: float) -> str:
    if score >= 80:
        return "Very High"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def _evidence(contribution: FeatureContribution) -> str:
    note = " (simulated)" if contribution.provenance == Provenance.SIMULATED else ""
    return f"{contribution.label} = {contribution.value:g}{note}: {contribution.contribution:+.1f} pts"


def _model_drivers(model_score: ModelScore, components: dict[str, float]) -> list[RiskDriver]:
    by_domain: dict[str, list[FeatureContribution]] = {}
    for contribution in model_score.contributions:
        by_domain.setdefault(contribution.domain, []).append(contribution)
    totals = {domain: sum(c.contribution for c in items) for domain, items in by_domain.items()}
    ranked = sorted((d for d in totals if totals[d] > 0), key=lambda d: totals[d], reverse=True)

    return [
        RiskDriver(
            name=DOMAIN_NAMES[domain],
            score=components[domain],
            explanation=f"{DOMAIN_EXPLANATIONS[domain]} Adds {totals[domain]:+.1f} pts vs the portfolio "
                        f"baseline of {model_score.base_value:.1f}.",
            evidence=[_evidence(c) for c in by_domain[domain] if abs(c.contribution) >= 0.1],
        )
        for domain in ranked[:MAX_DRIVERS]
    ]


def _heuristic_drivers(components: dict[str, float]) -> list[RiskDriver]:
    ranked = sorted(components, key=lambda d: components[d] * HEURISTIC_WEIGHTS[d], reverse=True)
    return [
        RiskDriver(
            name=DOMAIN_NAMES[domain],
            score=components[domain],
            explanation=DOMAIN_EXPLANATIONS[domain],
            evidence=[f"{DOMAIN_NAMES[domain]} index {components[domain]:.1f}/100 "
                      f"(weight {HEURISTIC_WEIGHTS[domain]:.0%})"],
        )
        for domain in ranked[:MAX_DRIVERS]
    ]


def _confidence(inputs: ReasoningInputs) -> Confidence:
    score, reasons = 90.0, []
    if all(p == Provenance.SIMULATED for p in inputs.provenances()):
        score -= 30
        reasons.append("All hazard inputs are simulated demo values, not measured data.")
    if inputs.flood.historical_flooding is None:
        score -= 5
        reasons.append("Flood claims history is unknown.")
    if inputs.profile.roof_age_estimated:
        score -= 5
        reasons.append("Roof age is estimated from property age.")
    if inputs.model_score is None:
        score -= 15
        reasons.append("Scored with unvalidated heuristic weights (no model available).")
    else:
        score -= min(20.0, inputs.model_score.holdout_mae)
        reasons.append(f"Model holdout error is ±{inputs.model_score.holdout_mae:.1f} pts (MAE).")

    score = max(0.0, score)
    level = ConfidenceLevel.HIGH if score >= 75 else ConfidenceLevel.MEDIUM if score >= 50 else ConfidenceLevel.LOW
    return Confidence(score=round(score, 1), level=level, reasons=reasons)


def _summary(inputs: ReasoningInputs, score: float, drivers: list[RiskDriver], confidence: Confidence) -> str:
    driver_text = ", ".join(d.name.lower() for d in drivers) or "no dominant driver"
    high_priority = sum(1 for r in inputs.recommendations if r.priority == Priority.HIGH)
    parts = [
        f"{inputs.profile.address}: risk {score:.0f}/100 ({risk_label(score)}), driven mainly by {driver_text}.",
    ]
    if inputs.risk_change is not None:
        parts.append(inputs.risk_change.narrative)
    parts.append(f"Confidence is {confidence.level.value}. "
                 f"{len(inputs.recommendations)} preventive actions identified, {high_priority} high priority.")
    return " ".join(parts)


class RiskReasoningAgent(BaseAgent):
    """Assembles a defensible assessment. It only restates values it was given; it never invents data."""

    name = "risk-reasoning-agent"

    def run(self, inputs: ReasoningInputs) -> RiskAssessment:
        components = inputs.component_scores()
        if inputs.model_score is not None:
            score = inputs.model_score.score
            method = inputs.model_score.model_name
            drivers = _model_drivers(inputs.model_score, components)
        else:
            score = round(sum(components[d] * w for d, w in HEURISTIC_WEIGHTS.items()), 2)
            method = "heuristic-weighted-sum"
            drivers = _heuristic_drivers(components)

        confidence = _confidence(inputs)
        return RiskAssessment(
            property_profile=inputs.profile,
            weather_risk=inputs.weather,
            flood_risk=inputs.flood,
            climate_risk=inputs.climate,
            neighborhood_risk=inputs.neighborhood,
            replacement_cost_risk=inputs.cost,
            risk_score=score,
            risk_label=risk_label(score),
            scoring_method=method,
            confidence=confidence,
            summary=_summary(inputs, score, drivers, confidence),
            primary_drivers=drivers,
            model_explanation=inputs.model_score,
            risk_change=inputs.risk_change,
            recommendations=inputs.recommendations,
            component_risks=list(inputs.component_risks),
        )

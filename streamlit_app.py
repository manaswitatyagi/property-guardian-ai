from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from property_guardian_ai.models.schemas import ModelScore, Priority, RiskAssessment, RiskSnapshot
from property_guardian_ai.services.container import Services, build_services

SERIES_BLUE = "#2a78d6"
INCREASE_RED = "#e34948"
DECREASE_BLUE = "#2a78d6"
BAND_LINES = ((35, "Moderate"), (60, "High"), (80, "Very High"))
PRIORITY_ICON = {Priority.HIGH: "🔴 High", Priority.MEDIUM: "🟠 Medium", Priority.LOW: "⚪ Low"}
TOP_FEATURES = 8


@st.cache_resource(show_spinner="Training risk model and seeding memory…")
def load_services() -> Services:
    return build_services()


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def trend_chart(history: list[RiskSnapshot]) -> go.Figure:
    months = [s.month for s in history]
    fig = go.Figure(go.Scatter(
        x=months, y=[s.risk_score for s in history], mode="lines+markers",
        line=dict(color=SERIES_BLUE, width=2), marker=dict(size=8),
        hovertemplate="%{x|%b %Y}<br>Risk score %{y:.1f}<extra></extra>",
    ))
    for threshold, label in BAND_LINES:
        fig.add_hline(y=threshold, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=11, color="gray"))
    fig.update_yaxes(range=[0, 100], title="Risk score")
    return _base_layout(fig, 320)


def shap_chart(model_score: ModelScore) -> go.Figure:
    top = sorted(model_score.contributions[:TOP_FEATURES], key=lambda c: c.contribution)
    fig = go.Figure(go.Bar(
        x=[c.contribution for c in top],
        y=[f"{c.label} = {c.value:g}" for c in top],
        orientation="h",
        marker=dict(color=[INCREASE_RED if c.contribution > 0 else DECREASE_BLUE for c in top],
                    line=dict(width=0)),
        hovertemplate="%{y}<br>%{x:+.2f} pts<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="gray", width=1))
    fig.update_xaxes(title="Contribution to score (pts vs portfolio baseline)")
    return _base_layout(fig, 340)


def render_header(assessment: RiskAssessment) -> None:
    change = assessment.risk_change
    cols = st.columns(4)
    cols[0].metric("Risk score", f"{assessment.risk_score:.0f} / 100",
                   delta=f"{change.score_change:+.1f} pts MoM" if change else None, delta_color="inverse")
    cols[1].metric("Risk band", assessment.risk_label)
    cols[2].metric("Confidence", f"{assessment.confidence.level.value.title()} ({assessment.confidence.score:.0f})")
    cols[3].metric("Preventive actions", len(assessment.recommendations))
    st.info(assessment.summary)


def render_drivers(assessment: RiskAssessment) -> None:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Why is the risk this high?")
        for driver in assessment.primary_drivers:
            with st.container(border=True):
                st.markdown(f"**{driver.name}** — component index {driver.score:.0f}/100")
                st.caption(driver.explanation)
                for item in driver.evidence:
                    st.markdown(f"- {item}")
        with st.expander("Confidence reasoning"):
            for reason in assessment.confidence.reasons:
                st.markdown(f"- {reason}")
    with right:
        st.subheader("Model explanation (SHAP)")
        if assessment.model_explanation:
            st.caption(f"{assessment.model_explanation.model_name} · portfolio baseline "
                       f"{assessment.model_explanation.base_value:.1f} pts · red raises risk, blue lowers it")
            st.plotly_chart(shap_chart(assessment.model_explanation), width="stretch")
        else:
            st.warning("No validated model available; scored with heuristic weights.")


def render_recommendations(assessment: RiskAssessment) -> None:
    st.subheader("Recommended preventive actions")
    if not assessment.recommendations:
        st.success("No preventive actions triggered for this property.")
        return
    for rec in assessment.recommendations:
        with st.container(border=True):
            st.markdown(f"**{rec.title}** · {PRIORITY_ICON[rec.priority]} · _{rec.driver}_")
            st.write(rec.description)
            st.caption(f"Evidence: {rec.evidence}")


def render_history(services: Services, assessment: RiskAssessment) -> None:
    property_id = assessment.property_profile.property_id
    history = services.memory.history(property_id)
    st.subheader("How has risk evolved?")
    st.plotly_chart(trend_chart(history), width="stretch")

    change = assessment.risk_change
    if change:
        st.markdown(f"**{change.previous_month:%b %Y} → {change.current_month:%b %Y}:** {change.narrative}")
        st.dataframe(pd.DataFrame([{
            "Feature": c.label, "Previous": c.previous_value, "Current": c.current_value,
            "Score impact (pts)": c.contribution_change,
        } for c in change.top_changes]), hide_index=True, width="stretch")

    with st.expander("Monthly history table"):
        st.dataframe(pd.DataFrame([{"Month": f"{s.month:%Y-%m}", "Risk score": s.risk_score} for s in history]),
                     hide_index=True, width="stretch")


def render_portfolio(services: Services) -> None:
    rows = []
    for pid in services.repository.property_ids():
        history = services.memory.history(pid)
        latest, previous = history[-1], history[-2]
        record = services.repository.get(pid)
        rows.append({
            "Property": pid,
            "City": record.property.address.split(",")[1].strip(),
            "Flood zone": record.flood.flood_zone,
            "Roof age": record.property.roof_age,
            "Risk score": round(latest.risk_score, 1),
            "MoM change": round(latest.risk_score - previous.risk_score, 1),
        })
    frame = pd.DataFrame(rows).sort_values("Risk score", ascending=False)
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
    })


def render_validation(services: Services) -> None:
    report = services.model.validation
    st.subheader("Model validation (grouped holdout)")
    st.caption(f"{report.n_folds}-fold cross-validation grouped by property: no property is in both train and "
               f"test. {report.n_rows} property-months across {report.n_properties} properties.")
    cols = st.columns(5)
    cols[0].metric("Holdout MAE", f"{report.holdout_mae:.1f} pts")
    cols[1].metric("Holdout R²", f"{report.holdout_r2:.2f}")
    cols[2].metric(f"ROC-AUC (≥{report.high_risk_threshold:.0f})", f"{report.roc_auc:.2f}" if report.roc_auc else "n/a")
    cols[3].metric("Precision / Recall",
                   f"{report.precision:.2f} / {report.recall:.2f}" if report.precision is not None else "n/a")
    cols[4].metric(f"Baseline MAE ({report.baseline_model_name.split('-')[0]})", f"{report.baseline_holdout_mae:.1f} pts")
    st.markdown("**Caveats**")
    for caveat in report.caveats:
        st.markdown(f"- {caveat}")


def main() -> None:
    st.set_page_config(page_title="Property Guardian AI", page_icon="🏠", layout="wide")
    st.title("Property Guardian AI")
    st.caption("Continuous, explainable property risk monitoring for underwriters — "
               "every score comes with drivers, evidence, confidence and preventive actions.")
    st.warning("Demo data: hazard, neighborhood and cost inputs are **simulated**; only coordinates, year built and "
               "square footage come from historical King County records. Do not read these as real hazards.",
               icon="⚠️")

    services = load_services()
    tab_property, tab_portfolio, tab_model = st.tabs(["Property assessment", "Portfolio", "Model governance"])

    with tab_property:
        ids = services.repository.property_ids()
        property_id = st.selectbox(
            "Property", ids, index=ids.index("KC-003"),
            format_func=lambda pid: f"{pid} — {services.repository.get(pid).property.address.split('(')[0].strip()}",
        )
        assessment = services.orchestrator.assess(property_id)
        render_header(assessment)
        render_drivers(assessment)
        render_recommendations(assessment)
        render_history(services, assessment)
        with st.expander("Raw RiskAssessment JSON (API response)"):
            st.json(assessment.model_dump(mode="json"), expanded=False)

    with tab_portfolio:
        st.subheader("Portfolio risk ranking")
        render_portfolio(services)

    with tab_model:
        render_validation(services)


if __name__ == "__main__":
    main()

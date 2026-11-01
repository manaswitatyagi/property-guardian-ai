from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from property_guardian_ai.services.data_loader import (
    build_component_risk,
    build_monitoring_narrative,
    load_demo_data,
)


def build_demo() -> None:
    st.set_page_config(page_title="Property Guardian AI Demo", layout="wide")
    st.title("Property Guardian AI")
    st.caption("House-by-house monitoring demo with visible asset-level risk checks for roof, plumbing, structure, and more.")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        .stImage > img { border-radius: 16px; box-shadow: 0 4px 18px rgba(0,0,0,0.08); }
        div[data-testid="stHorizontalBlock"] > div { align-items: stretch; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = load_demo_data()
    properties = data["properties"]
    monthly_risk = data["monthly_risk"]

    property_id = st.selectbox("Select property", properties["property_id"].tolist())
    selected_property = properties[properties["property_id"] == property_id].iloc[0]
    selected_monthly = monthly_risk[monthly_risk["property_id"] == property_id].sort_values("month")
    narrative = build_monitoring_narrative(selected_property, selected_monthly)
    component_risk = build_component_risk(selected_property)
    selected_component = str(st.query_params.get("component", st.session_state.get("selected_component", "Roof")))
    if selected_component not in component_risk:
        selected_component = "Roof"
    st.session_state["selected_component"] = selected_component

    latest = selected_monthly.iloc[-1]
    prior = selected_monthly.iloc[-2]
    delta = latest["monthly_risk_score_0_100"] - prior["monthly_risk_score_0_100"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall risk", f"{selected_property['risk_score_0_100']:.1f}")
    col2.metric("Neighborhood", f"{selected_property['neighborhood_score']:.1f}")
    col3.metric("Account activity", f"{selected_property['account_activity_score']:.1f}")
    col4.metric("Device activity", f"{selected_property['device_activity_score']:.1f}")
    col5.metric("Financial pressure", f"{selected_property['financial_pressure_score']:.1f}")

    st.subheader("Risk movement")
    trend_delta = f"{delta:+.1f} pts vs prior month"
    st.info(f"{narrative} | Change: {trend_delta}")

    left_col, right_col = st.columns([1.15, 1.85], gap="large")
    with left_col:
        house_html = f"""
        <style>
        .house-map {{
            position: relative;
            width: 100%;
            max-width: 520px;
            height: 370px;
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(49, 51, 63, 0.15);
            background-image: url('https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=1200&q=80');
            background-size: cover;
            background-position: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        }}
        .label {{
            position: absolute;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: white;
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px;
            padding: 6px 10px;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }}
        .label:hover {{ transform: translateY(-1px); }}
        .roof {{ left: 34%; top: 13%; }}
        .plumbing {{ left: 58%; top: 57%; }}
        .porch {{ left: 22%; bottom: 10%; }}
        </style>
        <div class="house-map">
            <div class="label roof" onclick="window.parent.location.search='?component=Roof';">Roof</div>
            <div class="label plumbing" onclick="window.parent.location.search='?component=Plumbing';">Plumbing</div>
            <div class="label porch" onclick="window.parent.location.search='?component=Porch';">Porch</div>
        </div>
        """
        components.html(house_html, height=430)
        st.caption(f"Selected area: {selected_component}")
    with right_col:
        st.subheader("Property profile")
        profile_cols = [
            "address",
            "city",
            "state",
            "year_built",
            "roof_age_years",
            "square_footage",
            "distance_to_water_km",
            "flood_zone",
            "home_value_usd",
        ]
        st.dataframe(selected_property[profile_cols].to_frame().T, width="stretch")

    st.subheader("Asset-level condition review")
    asset_order = ["Roof", "Porch", "Structure", "Plumbing", "Living area", "Bathtub", "Swimming pool"]

    button_cols = st.columns(len(asset_order))
    for idx, component_name in enumerate(asset_order):
        with button_cols[idx]:
            clicked = st.button(
                component_name,
                key=f"asset_{component_name}",
                width="stretch",
                type="secondary" if selected_component != component_name else "primary",
            )
            if clicked:
                selected_component = component_name
                st.session_state["selected_component"] = component_name
                st.query_params["component"] = component_name

    component_name = selected_component
    details = component_risk[component_name]
    score = float(details["score"])
    status = details["status"]

    detail_col, insight_col = st.columns([1.2, 1.8], gap="large")
    with detail_col:
        st.markdown(f"### {component_name}")
        st.metric("Risk score", f"{score:.0f}", delta=status)
        st.progress(min(score, 100) / 100)
        st.caption(details["detail"])
    with insight_col:
        st.markdown("**Why this is risky**")
        for item in details["risk_factors"]:
            st.write(f"• {item}")

        st.markdown("**What can change the score**")
        for item in details["mitigation"]:
            st.write(f"• {item}")

    tab_monitoring, tab_table = st.tabs(["Monitoring", "Portfolio"])

    with tab_monitoring:
        st.subheader("Monthly risk trend")
        st.line_chart(selected_monthly.set_index("month")["monthly_risk_score_0_100"])

        st.subheader("Monitor components")
        score_df = pd.DataFrame(
            {
                "Signal": [
                    "Neighborhood",
                    "Account activity",
                    "Device activity",
                    "Financial pressure",
                    "Overall monitoring",
                ],
                "Score": [
                    selected_property["neighborhood_score"],
                    selected_property["account_activity_score"],
                    selected_property["device_activity_score"],
                    selected_property["financial_pressure_score"],
                    selected_property["monitoring_score"],
                ],
            }
        )
        st.bar_chart(score_df.set_index("Signal")["Score"])

        st.subheader("Recent monthly pattern")
        st.dataframe(
            selected_monthly[["month", "monthly_risk_score_0_100", "dominant_risk_driver"]].tail(6),
            width="stretch",
        )

    with tab_table:
        st.subheader("Portfolio comparison")
        portfolio = properties[[
            "property_id",
            "address",
            "city",
            "risk_score_0_100",
            "neighborhood_score",
            "account_activity_score",
            "device_activity_score",
            "financial_pressure_score",
            "underwriting_risk_label",
        ]].copy()
        st.dataframe(portfolio, width="stretch")


if __name__ == "__main__":
    build_demo()

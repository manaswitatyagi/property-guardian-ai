from property_guardian_ai.services.data_loader import build_component_risk, load_demo_data


def test_load_demo_data():
    data = load_demo_data()

    assert "properties" in data
    assert "monthly_risk" in data
    assert len(data["properties"]) >= 10
    assert len(data["monthly_risk"]) >= 120
    assert "risk_score_0_100" in data["properties"].columns
    assert "monthly_risk_score_0_100" in data["monthly_risk"].columns

    component_risk = build_component_risk(data["properties"].iloc[0])
    assert set(component_risk) >= {
        "Porch",
        "Roof",
        "Plumbing",
        "Structure",
        "Living area",
        "Bathtub",
        "Swimming pool",
    }
    assert all(0 <= component_risk[name]["score"] <= 100 for name in component_risk)

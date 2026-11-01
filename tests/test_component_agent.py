from __future__ import annotations

from property_guardian_ai.dashboard import build_house_html

EXPECTED_KEYS = {"roof", "gutters", "foundation", "plumbing", "exterior", "openings", "yard"}


def test_every_component_is_scored_with_evidence(services) -> None:
    assessment = services.orchestrator.assess("KC-003")
    components = {c.key: c for c in assessment.component_risks}

    assert set(components) == EXPECTED_KEYS
    assert all(0 <= c.score <= 100 and c.evidence for c in components.values())
    assert components["foundation"].band in {"High", "Very High"}
    assert any(r.driver == "Flood exposure" for r in components["foundation"].recommendations)
    assert {m.feature for m in components["roof"].model_contributions} >= {"roof_age"}


def test_low_risk_home_scores_lower_than_flood_zone_home(services) -> None:
    risky = {c.key: c.score for c in services.orchestrator.assess("KC-003").component_risks}
    safe = {c.key: c.score for c in services.orchestrator.assess("KC-005").component_risks}

    assert safe["foundation"] < risky["foundation"]
    assert safe["roof"] < risky["roof"]


def test_house_view_embeds_every_component(services) -> None:
    assessment = services.orchestrator.assess("KC-001")
    html = build_house_html(assessment, services.repository.get("KC-001"))

    for key in EXPECTED_KEYS:
        assert f'data-key="{key}"' in html
    assert "Water · 0.6 km" in html
    assert "</script>" in html and html.count("</script>") == 1


def test_address_search(services) -> None:
    assert services.repository.search("kenmore 98028") == ["KC-003"]
    assert services.repository.search("Seattle") and all(
        "Seattle" in services.repository.get(pid).property.address for pid in services.repository.search("Seattle")
    )
    assert services.repository.search("1600 Pennsylvania") == []

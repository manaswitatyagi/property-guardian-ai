from __future__ import annotations

import pytest


def test_validation_is_grouped_by_property(services) -> None:
    report = services.model.validation

    assert report.n_properties == 10
    assert report.n_rows == 120
    assert report.trained_on_simulated_data
    assert report.roc_auc is not None
    assert report.holdout_mae < report.baseline_holdout_mae * 1.5


def test_shap_contributions_sum_to_score(services) -> None:
    record = services.repository.get("KC-001")
    scored = services.model.score(record, services.repository.latest_month("KC-001"))
    total = scored.base_value + sum(c.contribution for c in scored.contributions)

    assert total == pytest.approx(scored.score, abs=0.5)
    assert all(c.domain for c in scored.contributions)

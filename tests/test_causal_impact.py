"""Tests for Bayesian Structural Time Series (BSTS) implementation."""

import numpy as np
import pandas as pd
import pytest

from chilean_growth.causal_impact import (
    fit_bsts_model,
    fit_structural_time_series,
    summarize_impact,
    BSTSResult,
)


class TestBSTSResult:
    """Tests for BSTSResult namedtuple structure."""

    def test_bsts_result_fields(self):
        """BSTSResult should have all required fields."""
        fields = BSTSResult._fields
        expected = [
            "actual",
            "predicted",
            "predicted_lower",
            "predicted_upper",
            "pointwise_effect",
            "cumulative_effect",
            "effect_lower",
            "effect_upper",
            "posterior_mean",
            "posterior_std",
            "p_value",
        ]
        assert set(fields) == set(expected)


class TestFitBSTSModel:
    """Tests for the fit_bsts_model function."""

    def test_returns_bsts_result(self, sample_panel_data):
        """Should return BSTSResult namedtuple."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert isinstance(result, BSTSResult)

    def test_series_alignment(self, sample_panel_data):
        """All series should have same length."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert len(result.actual) == len(result.predicted)
        assert len(result.actual) == len(result.predicted_lower)
        assert len(result.actual) == len(result.predicted_upper)
        assert len(result.actual) == len(result.pointwise_effect)

    def test_effect_calculation(self, sample_panel_data):
        """Pointwise effect should equal actual minus predicted."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        expected_effect = result.actual - result.predicted
        pd.testing.assert_series_equal(
            result.pointwise_effect,
            expected_effect,
            check_names=False,
        )

    def test_cumulative_effect(self, sample_panel_data):
        """Cumulative effect should be cumsum of pointwise."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        expected_cum = result.pointwise_effect.cumsum()
        pd.testing.assert_series_equal(
            result.cumulative_effect,
            expected_cum,
            check_names=False,
        )

    def test_confidence_intervals(self, sample_panel_data):
        """Lower bound should be less than upper bound."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert all(result.predicted_lower <= result.predicted_upper)
        assert all(result.effect_lower <= result.effect_upper)

    def test_p_value_range(self, sample_panel_data):
        """P-value should be between 0 and 1."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert 0 <= result.p_value <= 1

    def test_custom_control_vars(self, sample_panel_data):
        """Should work with custom control variables."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            control_vars=["ARG", "BRA"],
        )
        assert isinstance(result, BSTSResult)


class TestFitStructuralTimeSeries:
    """Tests for the fit_structural_time_series function."""

    def test_returns_bsts_result(self, sample_panel_data):
        """Should return BSTSResult namedtuple."""
        result = fit_structural_time_series(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert isinstance(result, BSTSResult)

    def test_series_lengths(self, sample_panel_data):
        """All series should have same length."""
        result = fit_structural_time_series(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert len(result.actual) == len(result.predicted)

    def test_posterior_values(self, sample_panel_data):
        """Posterior mean and std should be finite."""
        result = fit_structural_time_series(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        assert np.isfinite(result.posterior_mean)
        assert np.isfinite(result.posterior_std)


class TestSummarizeImpact:
    """Tests for the summarize_impact function."""

    def test_returns_string(self, sample_panel_data):
        """Should return a formatted string."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        summary = summarize_impact(result, treatment_year=2014)
        assert isinstance(summary, str)

    def test_summary_contains_key_info(self, sample_panel_data):
        """Summary should contain key statistics."""
        result = fit_bsts_model(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
        )
        summary = summarize_impact(result, treatment_year=2014)
        assert "Actual" in summary
        assert "Predicted" in summary
        assert "Effect" in summary
        assert "P-value" in summary


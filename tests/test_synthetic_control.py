"""Tests for synthetic control method implementation."""

import numpy as np
import pandas as pd
import pytest

from chilean_growth.synthetic_control import (
    solve_weights,
    compute_rmspe,
    fit_synthetic_control,
    SCMResult,
)


class TestSolveWeights:
    """Tests for the solve_weights function."""

    def test_weights_sum_to_one(self, simple_matrices):
        """Weights should sum to 1."""
        X0, X1, V = simple_matrices
        W = solve_weights(X0, X1, V)
        assert np.isclose(np.sum(W), 1.0, atol=1e-6)

    def test_weights_non_negative(self, simple_matrices):
        """All weights should be non-negative."""
        X0, X1, V = simple_matrices
        W = solve_weights(X0, X1, V)
        assert np.all(W >= -1e-6)  # Allow small numerical errors

    def test_weights_shape(self, simple_matrices):
        """Weights should have correct shape."""
        X0, X1, V = simple_matrices
        W = solve_weights(X0, X1, V)
        assert W.shape == (X0.shape[1],)


class TestComputeRMSPE:
    """Tests for the compute_rmspe function."""

    def test_rmspe_positive(self, outcome_matrices):
        """RMSPE should be positive."""
        Y0, Y1, W = outcome_matrices
        rmspe = compute_rmspe(Y0, Y1, W)
        assert rmspe >= 0

    def test_rmspe_zero_perfect_fit(self):
        """RMSPE should be zero for perfect fit."""
        Y0 = np.array([[1, 2], [3, 4], [5, 6]])
        Y1 = np.array([1.5, 3.5, 5.5])
        W = np.array([0.5, 0.5])
        rmspe = compute_rmspe(Y0, Y1, W)
        assert np.isclose(rmspe, 0.0, atol=1e-10)

    def test_rmspe_known_value(self):
        """RMSPE should match known value."""
        Y0 = np.array([[1], [2], [3]])
        Y1 = np.array([2, 3, 4])
        W = np.array([1.0])
        rmspe = compute_rmspe(Y0, Y1, W)
        # Errors are [1, 1, 1], RMSPE = sqrt(mean([1, 1, 1])) = 1
        assert np.isclose(rmspe, 1.0, atol=1e-10)


class TestFitSyntheticControl:
    """Tests for the fit_synthetic_control function."""

    def test_returns_scm_result(self, sample_panel_data):
        """Should return SCMResult namedtuple."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        assert isinstance(result, SCMResult)

    def test_weights_in_result(self, sample_panel_data):
        """Result should contain valid weights."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        assert isinstance(result.weights, pd.Series)
        assert len(result.weights) > 0

    def test_series_lengths(self, sample_panel_data):
        """Actual and synthetic series should have same length."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        assert len(result.actual) == len(result.synthetic)
        assert len(result.actual) == len(result.gap)

    def test_gap_calculation(self, sample_panel_data):
        """Gap should equal actual minus synthetic."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        expected_gap = result.actual - result.synthetic
        pd.testing.assert_series_equal(
            result.gap,
            expected_gap,
            check_names=False,
        )

    def test_rmspe_positive_values(self, sample_panel_data):
        """RMSPE values should be positive."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        assert result.rmspe_pre >= 0
        assert result.rmspe_post >= 0
        assert result.rmspe_ratio >= 0

    def test_predictor_balance_shape(self, sample_panel_data):
        """Predictor balance table should have correct columns."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        assert "Actual" in result.predictor_balance.columns
        assert "Synthetic" in result.predictor_balance.columns
        assert "Sample Mean" in result.predictor_balance.columns

    def test_treatment_effect_direction(self, sample_panel_data):
        """Chile should show negative gap post-2014 due to simulated slowdown."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER", "COL"],
        )
        # Post-treatment gap should be predominantly negative
        post_gap = result.gap[result.gap.index >= 2014]
        assert post_gap.mean() < 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_donor(self, sample_panel_data):
        """Should work with a single donor."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG"],
        )
        assert np.isclose(result.weights.sum(), 1.0)

    def test_custom_outcome_lags(self, sample_panel_data):
        """Should work with custom outcome lags."""
        result = fit_synthetic_control(
            df=sample_panel_data,
            outcome_var="gdp_per_capita",
            treated_unit="CHL",
            treatment_year=2014,
            donor_pool=["ARG", "BRA", "PER"],
            outcome_lags=[1995, 2005, 2010],
        )
        assert isinstance(result, SCMResult)

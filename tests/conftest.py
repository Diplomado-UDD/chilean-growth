"""Pytest fixtures for Chilean Growth Slowdown tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    np.random.seed(42)

    countries = ["CHL", "ARG", "BRA", "PER", "COL"]
    years = list(range(1990, 2020))

    data = []
    for country in countries:
        base_gdp = 5000 if country == "CHL" else np.random.uniform(3000, 6000)
        growth_rate = 0.03 if country == "CHL" else np.random.uniform(0.01, 0.04)

        for i, year in enumerate(years):
            # Add slowdown for Chile after 2014
            if country == "CHL" and year >= 2014:
                growth_rate = 0.01

            gdp = base_gdp * (1 + growth_rate) ** i

            data.append({
                "country": country,
                "year": year,
                "gdp_per_capita": gdp + np.random.normal(0, 100),
                "population_growth": np.random.uniform(0.5, 2.0),
                "life_expectancy": np.random.uniform(70, 80),
                "adolescent_fertility": np.random.uniform(40, 80),
                "birth_rate": np.random.uniform(12, 20),
                "gov_consumption": np.random.uniform(0.10, 0.20),
                "gross_capital_formation": np.random.uniform(20, 30),
                "trade_openness": np.random.uniform(40, 80),
                "mean_years_schooling": np.random.uniform(7, 12),
            })

    return pd.DataFrame(data)


@pytest.fixture
def simple_matrices():
    """Create simple matrices for testing optimization."""
    np.random.seed(42)

    # K predictors, J donors
    K, J = 5, 4

    X0 = np.random.randn(K, J)  # Control units
    X1 = np.random.randn(K)     # Treated unit
    V = np.eye(K)               # Identity weights

    return X0, X1, V


@pytest.fixture
def outcome_matrices():
    """Create outcome matrices for RMSPE testing."""
    np.random.seed(42)

    T, J = 10, 4

    Y0 = np.random.randn(T, J) * 1000 + 5000  # Control outcomes
    Y1 = np.random.randn(T) * 1000 + 5000      # Treated outcome
    W = np.array([0.3, 0.3, 0.2, 0.2])         # Weights

    return Y0, Y1, W

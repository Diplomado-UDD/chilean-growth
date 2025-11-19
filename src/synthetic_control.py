"""
Synthetic Control Method implementation for Chilean Growth Slowdown replication.
Based on Abadie, Diamond & Hainmueller (2010, 2015) and Abadie (2021).
"""

from typing import NamedTuple

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from .logger import logger

from .config import (
    COUNTRY_NAMES,
    POST_TREATMENT_END,
    PRE_TREATMENT_START,
    TREATED_COUNTRY,
    TREATMENT_YEAR,
)


class SCMResult(NamedTuple):
    """Container for SCM results."""

    weights: pd.Series  # Optimal country weights (J x 1)
    v_weights: np.ndarray  # Predictor importance weights (K x 1)
    synthetic: pd.Series  # Synthetic control outcome series
    actual: pd.Series  # Actual treated unit outcome series
    gap: pd.Series  # Treatment effect (actual - synthetic)
    rmspe_pre: float  # Pre-treatment RMSPE
    rmspe_post: float  # Post-treatment RMSPE
    rmspe_ratio: float  # Ratio of post/pre RMSPE
    predictor_balance: pd.DataFrame  # Predictor balance table


def solve_weights(
    X0: np.ndarray,
    X1: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Solve for optimal synthetic control weights given predictor weights V.

    Minimizes: (X1 - X0 @ W)' V (X1 - X0 @ W)
    Subject to: W >= 0, sum(W) = 1

    Args:
        X0: Predictor matrix for control units (K x J)
        X1: Predictor vector for treated unit (K x 1)
        V: Diagonal weight matrix for predictors (K x K)

    Returns:
        Optimal weights W (J x 1)
    """
    J = X0.shape[1]

    W = cp.Variable(J)

    # Weighted distance
    diff = X1 - X0 @ W
    objective = cp.quad_form(diff, V)

    constraints = [
        W >= 0,
        cp.sum(W) == 1,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)

    return W.value


def compute_rmspe(
    Y0: np.ndarray,
    Y1: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Compute Root Mean Square Prediction Error.

    Args:
        Y0: Outcome matrix for control units (T x J)
        Y1: Outcome vector for treated unit (T x 1)
        W: Synthetic control weights (J x 1)

    Returns:
        RMSPE value
    """
    synthetic = Y0 @ W
    errors = Y1 - synthetic
    return np.sqrt(np.mean(errors**2))


def outer_objective(
    v_flat: np.ndarray,
    X0: np.ndarray,
    X1: np.ndarray,
    Y0_pre: np.ndarray,
    Y1_pre: np.ndarray,
) -> float:
    """
    Outer optimization objective: minimize pre-treatment RMSPE.

    Args:
        v_flat: Flattened predictor weights
        X0, X1: Predictor matrices
        Y0_pre, Y1_pre: Pre-treatment outcome matrices

    Returns:
        Pre-treatment RMSPE
    """
    K = len(v_flat)
    V = np.diag(v_flat)

    # Solve inner problem for W given V
    W = solve_weights(X0, X1, V)

    if W is None:
        return 1e10

    # Compute pre-treatment RMSPE
    return compute_rmspe(Y0_pre, Y1_pre, W)


def fit_synthetic_control(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    predictor_vars: list[str] | None = None,
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = TREATMENT_YEAR,
    donor_pool: list[str] | None = None,
    outcome_lags: list[int] | None = None,
) -> SCMResult:
    """
    Fit Synthetic Control Method.

    Args:
        df: Panel data with countries and years
        outcome_var: Name of outcome variable
        predictor_vars: List of predictor variable names
        treated_unit: ISO3 code of treated country
        treatment_year: Year of treatment
        donor_pool: List of donor country codes (if None, use all non-treated)
        outcome_lags: Years to include as outcome lags in predictors

    Returns:
        SCMResult with weights, synthetic series, and diagnostics
    """
    if predictor_vars is None:
        predictor_vars = [
            "population_growth",
            "life_expectancy",
            "adolescent_fertility",
            "birth_rate",
            "gov_consumption",
            "gross_capital_formation",
            "trade_openness",
            "mean_years_schooling",
        ]

    if outcome_lags is None:
        outcome_lags = [1990, 1995, 2000, 2005, 2010, 2013]

    # Filter to donor pool
    if donor_pool is not None:
        df = df[
            (df["country"] == treated_unit) | (df["country"].isin(donor_pool))
        ].copy()

    # Split data
    df_treated = df[df["country"] == treated_unit].set_index("year")
    df_control = df[df["country"] != treated_unit]

    control_countries = df_control["country"].unique()
    J = len(control_countries)

    # Build predictor matrices
    # X0: K x J, X1: K x 1 (vectors of pre-treatment averages)
    pre_mask = df_control["year"] < treatment_year

    predictor_list = []
    predictor_names = []

    # Add standard predictors (pre-treatment averages)
    for var in predictor_vars:
        if var in df.columns:
            # Treated unit
            x1_val = df_treated.loc[
                df_treated.index < treatment_year, var
            ].mean()

            # Control units
            x0_vals = (
                df_control[pre_mask]
                .groupby("country")[var]
                .mean()
                .reindex(control_countries)
            )

            predictor_list.append((x1_val, x0_vals.values))
            predictor_names.append(var)

    # Add outcome lags
    for lag_year in outcome_lags:
        if lag_year in df_treated.index:
            x1_val = df_treated.loc[lag_year, outcome_var]

            x0_vals = (
                df_control[df_control["year"] == lag_year]
                .set_index("country")[outcome_var]
                .reindex(control_countries)
            )

            predictor_list.append((x1_val, x0_vals.values))
            predictor_names.append(f"{outcome_var}_{lag_year}")

    # Assemble matrices
    K = len(predictor_list)
    X1 = np.array([p[0] for p in predictor_list])
    X0 = np.array([p[1] for p in predictor_list])

    # Handle missing values
    nan_mask = np.isnan(X0).any(axis=1) | np.isnan(X1)
    if nan_mask.any():
        print(f"Dropping {nan_mask.sum()} predictors with missing values")
        X0 = X0[~nan_mask]
        X1 = X1[~nan_mask]
        predictor_names = [p for p, m in zip(predictor_names, nan_mask) if not m]
        K = len(predictor_names)

    # Normalize predictors (important for optimization)
    X_mean = np.nanmean(np.hstack([X0, X1.reshape(-1, 1)]), axis=1)
    X_std = np.nanstd(np.hstack([X0, X1.reshape(-1, 1)]), axis=1)
    X_std[X_std == 0] = 1

    X0_norm = (X0 - X_mean.reshape(-1, 1)) / X_std.reshape(-1, 1)
    X1_norm = (X1 - X_mean) / X_std

    # Outcome matrices
    Y0_pre = (
        df_control[pre_mask]
        .pivot(index="year", columns="country", values=outcome_var)
        .reindex(columns=control_countries)
        .values
    )

    Y1_pre = df_treated.loc[
        df_treated.index < treatment_year, outcome_var
    ].values

    # Optimize V weights using nested optimization
    v0 = np.ones(K) / K

    result = minimize(
        outer_objective,
        v0,
        args=(X0_norm, X1_norm, Y0_pre, Y1_pre),
        method="L-BFGS-B",
        bounds=[(0, 1)] * K,
        options={"maxiter": 1000},
    )

    V_opt = np.diag(result.x / result.x.sum())

    # Solve for final weights
    W_opt = solve_weights(X0_norm, X1_norm, V_opt)

    # Create synthetic control series
    all_years = range(PRE_TREATMENT_START, POST_TREATMENT_END + 1)

    Y0_all = (
        df_control
        .pivot(index="year", columns="country", values=outcome_var)
        .reindex(index=all_years, columns=control_countries)
    )

    Y1_all = df_treated.reindex(all_years)[outcome_var]

    synthetic = pd.Series(
        Y0_all.values @ W_opt,
        index=all_years,
        name="synthetic",
    )

    actual = Y1_all.rename("actual")
    gap = actual - synthetic
    gap.name = "gap"

    # Compute RMSPE
    pre_years = [y for y in all_years if y < treatment_year]
    post_years = [y for y in all_years if y >= treatment_year]

    rmspe_pre = np.sqrt(np.mean(gap.loc[pre_years] ** 2))
    rmspe_post = np.sqrt(np.mean(gap.loc[post_years] ** 2))
    rmspe_ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf

    # Create weights series
    weights = pd.Series(W_opt, index=control_countries, name="weight")
    weights = weights[weights > 0.001].sort_values(ascending=False)

    # Predictor balance table
    balance_data = {
        "Actual": X1,
        "Synthetic": X0 @ W_opt,
        "Sample Mean": X0.mean(axis=1),
    }
    predictor_balance = pd.DataFrame(balance_data, index=predictor_names)

    return SCMResult(
        weights=weights,
        v_weights=result.x,
        synthetic=synthetic,
        actual=actual,
        gap=gap,
        rmspe_pre=rmspe_pre,
        rmspe_post=rmspe_post,
        rmspe_ratio=rmspe_ratio,
        predictor_balance=predictor_balance,
    )


def placebo_test(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    predictor_vars: list[str] | None = None,
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = TREATMENT_YEAR,
    donor_pool: list[str] | None = None,
) -> dict[str, SCMResult]:
    """
    Run placebo tests by applying SCM to each control unit.

    Args:
        Same as fit_synthetic_control

    Returns:
        Dictionary mapping country code to SCMResult
    """
    results = {}

    # First get the treated unit result
    results[treated_unit] = fit_synthetic_control(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        treated_unit=treated_unit,
        treatment_year=treatment_year,
        donor_pool=donor_pool,
    )

    # Get donor pool
    if donor_pool is None:
        donor_pool = df[df["country"] != treated_unit]["country"].unique()

    # Run for each placebo unit
    for placebo_unit in tqdm(donor_pool, desc="Placebo tests"):
        try:
            # Create new donor pool excluding placebo unit
            new_donors = [c for c in donor_pool if c != placebo_unit]

            result = fit_synthetic_control(
                df=df,
                outcome_var=outcome_var,
                predictor_vars=predictor_vars,
                treated_unit=placebo_unit,
                treatment_year=treatment_year,
                donor_pool=new_donors + [treated_unit],
            )

            results[placebo_unit] = result

        except Exception as e:
            logger.warning(f"Failed placebo for {placebo_unit}: {e}")

    return results


def compute_p_values(
    treated_result: SCMResult,
    placebo_results: dict[str, SCMResult],
    years: list[int] | None = None,
) -> pd.Series:
    """
    Compute p-values for treatment effect.

    P-value = proportion of placebo effects >= treated effect

    Args:
        treated_result: SCM result for treated unit
        placebo_results: Dictionary of placebo results
        years: Years to compute p-values for

    Returns:
        Series of p-values by year
    """
    if years is None:
        years = [y for y in treated_result.gap.index if y >= TREATMENT_YEAR]

    p_values = {}

    for year in years:
        treated_gap = abs(treated_result.gap.loc[year])

        # Count placebos with larger gaps
        n_larger = sum(
            1
            for result in placebo_results.values()
            if abs(result.gap.get(year, 0)) >= treated_gap
        )

        p_values[year] = n_larger / len(placebo_results)

    return pd.Series(p_values, name="p_value")


def in_time_placebo(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    predictor_vars: list[str] | None = None,
    treated_unit: str = TREATED_COUNTRY,
    placebo_year: int = 2006,
    donor_pool: list[str] | None = None,
) -> SCMResult:
    """
    Run in-time placebo test with alternative treatment year.

    Args:
        placebo_year: Year to use as placebo treatment

    Returns:
        SCMResult for placebo treatment
    """
    return fit_synthetic_control(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        treated_unit=treated_unit,
        treatment_year=placebo_year,
        donor_pool=donor_pool,
    )


def jackknife_test(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    predictor_vars: list[str] | None = None,
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = TREATMENT_YEAR,
    donor_pool: list[str] | None = None,
) -> dict[str, SCMResult]:
    """
    Run leave-one-out (jackknife) robustness test.

    Drops each donor with positive weight and re-estimates.

    Args:
        Same as fit_synthetic_control

    Returns:
        Dictionary mapping dropped country to SCMResult
    """
    # First run full model
    full_result = fit_synthetic_control(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        treated_unit=treated_unit,
        treatment_year=treatment_year,
        donor_pool=donor_pool,
    )

    # Get countries with positive weights
    positive_weight_countries = full_result.weights[
        full_result.weights > 0.001
    ].index.tolist()

    results = {"full": full_result}

    if donor_pool is None:
        donor_pool = list(df[df["country"] != treated_unit]["country"].unique())

    # Run for each leave-out
    for drop_country in tqdm(positive_weight_countries, desc="Jackknife tests"):
        try:
            new_donors = [c for c in donor_pool if c != drop_country]

            result = fit_synthetic_control(
                df=df,
                outcome_var=outcome_var,
                predictor_vars=predictor_vars,
                treated_unit=treated_unit,
                treatment_year=treatment_year,
                donor_pool=new_donors,
            )

            results[f"w/o_{drop_country}"] = result

        except Exception as e:
            logger.warning(f"Failed jackknife w/o {drop_country}: {e}")

    return results


if __name__ == "__main__":
    # Test with sample data
    from .data_loader import assemble_panel_data

    df = assemble_panel_data()

    result = fit_synthetic_control(df)

    print("\nOptimal Weights:")
    print(result.weights)

    print(f"\nPre-treatment RMSPE: {result.rmspe_pre:.2f}")
    print(f"Post-treatment RMSPE: {result.rmspe_post:.2f}")
    print(f"RMSPE Ratio: {result.rmspe_ratio:.2f}")

    print("\nPredictor Balance:")
    print(result.predictor_balance)

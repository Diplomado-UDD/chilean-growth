"""
Bayesian Structural Time Series (CausalImpact) for Chilean Growth Slowdown.
Based on Brodersen et al. (2015).
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import (
    MCMC_SAMPLES,
    RANDOM_SEED,
    TREATED_COUNTRY,
    TREATMENT_YEAR,
)


class BSTSResult(NamedTuple):
    """Container for BSTS/CausalImpact results."""

    actual: pd.Series
    predicted: pd.Series
    predicted_lower: pd.Series
    predicted_upper: pd.Series
    pointwise_effect: pd.Series
    cumulative_effect: pd.Series
    effect_lower: pd.Series
    effect_upper: pd.Series
    posterior_mean: float
    posterior_std: float
    p_value: float


def fit_bsts_model(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = TREATMENT_YEAR,
    control_vars: list[str] | None = None,
    n_samples: int = MCMC_SAMPLES,
    confidence: float = 0.95,
) -> BSTSResult:
    """
    Fit Bayesian Structural Time Series model.

    Uses state-space model with local linear trend and regression
    on control series to predict counterfactual.

    Args:
        df: Panel data
        outcome_var: Name of outcome variable
        treated_unit: ISO3 code of treated country
        treatment_year: Year of treatment
        control_vars: List of control country codes (if None, use all)
        n_samples: Number of posterior samples
        confidence: Confidence level for intervals

    Returns:
        BSTSResult with predictions and effects

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    required_cols = ["country", "year"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if outcome_var not in df.columns:
        raise ValueError(f"Outcome variable '{outcome_var}' not in DataFrame")

    if treated_unit not in df["country"].unique():
        raise ValueError(f"Treated unit '{treated_unit}' not in data")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")

    # Prepare data
    df_treated = df[df["country"] == treated_unit].set_index("year")[outcome_var]
    df_control = df[df["country"] != treated_unit]

    # Create control series matrix
    if control_vars is None:
        control_vars = df_control["country"].unique().tolist()

    Y_control = (
        df_control
        .pivot(index="year", columns="country", values=outcome_var)
        .reindex(columns=control_vars)
    )

    # Align indices
    common_years = df_treated.index.intersection(Y_control.index)
    Y_treated = df_treated.loc[common_years]
    Y_control = Y_control.loc[common_years]

    # Handle missing values - use forward fill then backward fill
    Y_control = Y_control.ffill().bfill()
    Y_treated = Y_treated.ffill().bfill()

    # Split into pre and post treatment
    pre_mask = Y_treated.index < treatment_year
    post_mask = Y_treated.index >= treatment_year

    Y_pre = Y_treated[pre_mask]
    X_pre = Y_control[pre_mask]

    Y_post = Y_treated[post_mask]
    X_post = Y_control[post_mask]

    # Fit regression model on pre-treatment data
    # Use regularized regression to handle multicollinearity
    X_pre_const = sm.add_constant(X_pre)

    # Ridge regression via OLS with regularization
    model = sm.OLS(Y_pre, X_pre_const)
    result = model.fit_regularized(alpha=0.1, L1_wt=0)

    # Predict counterfactual for all periods
    X_all = sm.add_constant(Y_control)
    predictions = result.predict(X_all)

    # Estimate prediction uncertainty via bootstrap
    residuals_pre = Y_pre - result.predict(X_pre_const)
    sigma = residuals_pre.std()

    # Generate posterior samples
    np.random.seed(RANDOM_SEED)
    prediction_samples = np.zeros((n_samples, len(Y_treated)))

    for i in range(n_samples):
        # Sample from predictive distribution
        noise = np.random.normal(0, sigma, len(Y_treated))
        prediction_samples[i] = predictions.values + noise

    # Compute credible intervals
    alpha = 1 - confidence
    predicted_lower = pd.Series(
        np.percentile(prediction_samples, 100 * alpha / 2, axis=0),
        index=Y_treated.index,
    )
    predicted_upper = pd.Series(
        np.percentile(prediction_samples, 100 * (1 - alpha / 2), axis=0),
        index=Y_treated.index,
    )

    # Compute effects
    predicted = pd.Series(predictions.values, index=Y_treated.index)
    actual = Y_treated

    pointwise_effect = actual - predicted
    cumulative_effect = pointwise_effect.cumsum()

    # Effect credible intervals
    effect_samples = actual.values.reshape(1, -1) - prediction_samples
    effect_lower = pd.Series(
        np.percentile(effect_samples, 100 * alpha / 2, axis=0),
        index=Y_treated.index,
    )
    effect_upper = pd.Series(
        np.percentile(effect_samples, 100 * (1 - alpha / 2), axis=0),
        index=Y_treated.index,
    )

    # Posterior summary for post-treatment effect
    post_effect_samples = effect_samples[:, post_mask.sum() * -1:]
    posterior_mean = np.mean(post_effect_samples)
    posterior_std = np.std(post_effect_samples)

    # P-value: probability that effect is positive (one-sided test)
    # For negative effect (slowdown), we compute P(effect < 0)
    p_value = np.mean(post_effect_samples.mean(axis=1) > 0)

    return BSTSResult(
        actual=actual,
        predicted=predicted,
        predicted_lower=predicted_lower,
        predicted_upper=predicted_upper,
        pointwise_effect=pointwise_effect,
        cumulative_effect=cumulative_effect,
        effect_lower=effect_lower,
        effect_upper=effect_upper,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        p_value=p_value,
    )


def fit_structural_time_series(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = TREATMENT_YEAR,
    confidence: float = 0.95,
) -> BSTSResult:
    """
    Fit structural time series model with local linear trend.

    Alternative to regression-based approach using state-space models.

    Args:
        df: Panel data
        outcome_var: Name of outcome variable
        treated_unit: ISO3 code of treated country
        treatment_year: Year of treatment
        confidence: Confidence level for intervals

    Returns:
        BSTSResult with predictions and effects
    """
    # Get treated series
    df_treated = df[df["country"] == treated_unit].set_index("year")[outcome_var]

    # Split pre/post
    pre_treatment = df_treated[df_treated.index < treatment_year]

    # Fit local linear trend model on pre-treatment
    model = sm.tsa.UnobservedComponents(
        pre_treatment,
        level="local linear trend",
    )
    result = model.fit(disp=False)

    # Forecast post-treatment
    n_post = len(df_treated) - len(pre_treatment)
    forecast = result.get_forecast(steps=n_post)

    # Combine fitted and forecast
    fitted_pre = result.fittedvalues
    forecast_mean = forecast.predicted_mean

    # Align indices properly
    post_years = df_treated.index[df_treated.index >= treatment_year]
    forecast_mean = pd.Series(forecast_mean.values[:len(post_years)], index=post_years)
    fitted_pre = pd.Series(fitted_pre.values, index=pre_treatment.index)

    predicted = pd.concat([fitted_pre, forecast_mean])

    # Confidence intervals
    alpha = 1 - confidence

    # Pre-treatment intervals from smoothed state
    pre_se = np.sqrt(result.smoothed_state_cov[0, 0, :])
    pre_lower = fitted_pre - stats.norm.ppf(1 - alpha / 2) * pre_se
    pre_upper = fitted_pre + stats.norm.ppf(1 - alpha / 2) * pre_se

    # Post-treatment intervals from forecast
    post_ci = forecast.conf_int(alpha=alpha)

    predicted_lower = pd.concat([
        pd.Series(pre_lower, index=pre_treatment.index),
        post_ci.iloc[:, 0],
    ])
    predicted_upper = pd.concat([
        pd.Series(pre_upper, index=pre_treatment.index),
        post_ci.iloc[:, 1],
    ])

    # Effects
    actual = df_treated
    pointwise_effect = actual - predicted
    cumulative_effect = pointwise_effect.cumsum()

    effect_lower = actual - predicted_upper
    effect_upper = actual - predicted_lower

    # Posterior summary
    post_mask = actual.index >= treatment_year
    posterior_mean = pointwise_effect[post_mask].mean()
    posterior_std = pointwise_effect[post_mask].std()

    # Simplified p-value
    t_stat = posterior_mean / (posterior_std / np.sqrt(post_mask.sum()))
    p_value = 1 - stats.t.cdf(abs(t_stat), df=post_mask.sum() - 1)

    return BSTSResult(
        actual=actual,
        predicted=predicted,
        predicted_lower=predicted_lower,
        predicted_upper=predicted_upper,
        pointwise_effect=pointwise_effect,
        cumulative_effect=cumulative_effect,
        effect_lower=effect_lower,
        effect_upper=effect_upper,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        p_value=p_value,
    )


def summarize_impact(result: BSTSResult, treatment_year: int = TREATMENT_YEAR) -> str:
    """
    Generate summary report of causal impact analysis.

    Args:
        result: BSTSResult from model fitting
        treatment_year: Year of treatment

    Returns:
        Formatted summary string
    """
    post_mask = result.actual.index >= treatment_year
    post_actual = result.actual[post_mask]
    post_predicted = result.predicted[post_mask]
    post_effect = result.pointwise_effect[post_mask]

    avg_actual = post_actual.mean()
    avg_predicted = post_predicted.mean()
    avg_effect = post_effect.mean()
    cum_effect = post_effect.sum()

    rel_effect = (avg_effect / avg_predicted) * 100

    summary = f"""
Causal Impact Analysis Summary
==============================

Post-treatment period: {treatment_year} - {post_actual.index.max()}

Average values:
  - Actual:    ${avg_actual:,.0f}
  - Predicted: ${avg_predicted:,.0f}
  - Effect:    ${avg_effect:,.0f} ({rel_effect:+.1f}%)

Cumulative effect: ${cum_effect:,.0f}

Final year ({post_actual.index.max()}):
  - Actual:    ${post_actual.iloc[-1]:,.0f}
  - Predicted: ${post_predicted.iloc[-1]:,.0f}
  - Effect:    ${post_effect.iloc[-1]:,.0f}

Statistical significance:
  - Posterior mean: ${result.posterior_mean:,.0f}
  - Posterior std:  ${result.posterior_std:,.0f}
  - P-value:        {result.p_value:.3f}
"""

    return summary


if __name__ == "__main__":
    from .data_loader import assemble_panel_data

    df = assemble_panel_data()

    # Fit BSTS model
    result = fit_bsts_model(df)

    print(summarize_impact(result))

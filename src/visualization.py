"""
Visualization functions for Chilean Growth Slowdown replication.
Reproduces figures from Toni, Paniagua & Ordenes (2023).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import filtfilt, butter

from .config import (
    TREATMENT_YEAR,
    PRE_TREATMENT_START,
    POST_TREATMENT_END,
    COUNTRY_NAMES,
    HP_LAMBDA,
)
from .synthetic_control import SCMResult
from .causal_impact import BSTSResult


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
    })


def dollar_formatter(x, pos):
    """Format axis ticks as dollars."""
    return f"${x:,.0f}"


def hp_filter(series: pd.Series, lamb: float = HP_LAMBDA) -> pd.Series:
    """
    Apply Hodrick-Prescott filter to extract trend.

    Args:
        series: Time series data
        lamb: Smoothing parameter (100 for annual data)

    Returns:
        Trend component
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, trend = hpfilter(series.dropna(), lamb=lamb)
    return trend


def plot_scm_main(
    result: SCMResult,
    title: str = "Per-capita income",
    save_path: str | None = None,
):
    """
    Plot main SCM result (Figure 8 left panel in paper).

    Args:
        result: SCMResult from fit_synthetic_control
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    years = result.actual.index

    # Plot actual and synthetic
    ax.plot(years, result.actual, "b-", label="Real Chile", linewidth=2.5)
    ax.plot(
        years, result.synthetic, "r--", label="Synthetic Chile", linewidth=2
    )

    # Treatment line
    ax.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)

    # Annotations for end values
    final_year = years[-1]
    ax.annotate(
        f"${result.synthetic.iloc[-1]:,.0f}",
        xy=(final_year, result.synthetic.iloc[-1]),
        xytext=(final_year + 0.5, result.synthetic.iloc[-1] + 300),
        fontsize=10,
    )
    ax.annotate(
        f"${result.actual.iloc[-1]:,.0f}",
        xy=(final_year, result.actual.iloc[-1]),
        xytext=(final_year + 0.5, result.actual.iloc[-1] - 500),
        fontsize=10,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Real GDP per capita constant 2015 US$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_scm_trend(
    result: SCMResult,
    title: str = "Per-capita income (HP filtered trends)",
    save_path: str | None = None,
):
    """
    Plot SCM result with HP filtered trends (Figure 8 right panel).

    Args:
        result: SCMResult from fit_synthetic_control
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Apply HP filter
    actual_trend = hp_filter(result.actual)
    synthetic_trend = hp_filter(result.synthetic)

    years = actual_trend.index

    ax.plot(years, actual_trend, "b-", label="Real Chile Trend", linewidth=2.5)
    ax.plot(
        years, synthetic_trend, "r--", label="Synthetic Chile Trend", linewidth=2
    )

    ax.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)

    # Annotations
    final_year = years[-1]
    ax.annotate(
        f"${synthetic_trend.iloc[-1]:,.0f}",
        xy=(final_year, synthetic_trend.iloc[-1]),
        xytext=(final_year + 0.5, synthetic_trend.iloc[-1] + 300),
        fontsize=10,
    )
    ax.annotate(
        f"${actual_trend.iloc[-1]:,.0f}",
        xy=(final_year, actual_trend.iloc[-1]),
        xytext=(final_year + 0.5, actual_trend.iloc[-1] - 500),
        fontsize=10,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Real GDP per capita constant 2015 US$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_decomposition(
    result: SCMResult,
    potential_gdp_growth: float = 0.045,  # 4.5% from PIB Tendencial
    title: str = "External and internal factors decomposition",
    save_path: str | None = None,
):
    """
    Plot decomposition of internal vs external factors (Figure 9).

    Args:
        result: SCMResult from fit_synthetic_control
        potential_gdp_growth: Annual potential GDP growth rate
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    years = result.actual.index

    # Calculate potential GDP (no shocks scenario)
    base_value = result.actual.loc[TREATMENT_YEAR - 1]
    potential_years = [y for y in years if y >= TREATMENT_YEAR - 1]
    potential_gdp = pd.Series(
        [base_value * (1 + potential_gdp_growth) ** (y - TREATMENT_YEAR + 1)
         for y in potential_years],
        index=potential_years,
    )

    # Left panel: full series
    ax1.plot(years, result.actual, "b-", label="Real Chile", linewidth=2.5)
    ax1.plot(
        years, result.synthetic, "r--", label="Synthetic Chile", linewidth=2
    )
    ax1.plot(
        potential_gdp.index,
        potential_gdp,
        "b:",
        label="Chile w/o Treatment & External shocks",
        linewidth=1.5,
    )

    ax1.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Real GDP per capita constant 2015 US$")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    # Right panel: zoomed post-treatment
    post_years = [y for y in years if y >= TREATMENT_YEAR]
    ax2.plot(
        post_years,
        result.actual.loc[post_years],
        "b-",
        label="Real Chile",
        linewidth=2.5,
    )
    ax2.plot(
        post_years,
        result.synthetic.loc[post_years],
        "r--",
        label="Synthetic Chile",
        linewidth=2,
    )
    ax2.plot(
        potential_gdp.loc[post_years].index,
        potential_gdp.loc[post_years],
        "b:",
        label="Chile w/o Treatment & External shocks",
        linewidth=1.5,
    )

    # Add decomposition annotations
    final = post_years[-1]
    total_gap = potential_gdp.loc[final] - result.actual.loc[final]
    internal_gap = result.synthetic.loc[final] - result.actual.loc[final]
    external_gap = potential_gdp.loc[final] - result.synthetic.loc[final]

    # Arrows showing decomposition
    y_mid_internal = (result.actual.loc[final] + result.synthetic.loc[final]) / 2
    y_mid_external = (result.synthetic.loc[final] + potential_gdp.loc[final]) / 2

    ax2.annotate(
        "2/3",
        xy=(final - 0.3, y_mid_internal),
        fontsize=12,
        fontweight="bold",
    )
    ax2.annotate(
        "1/3",
        xy=(final - 0.3, y_mid_external),
        fontsize=12,
        fontweight="bold",
    )

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Real GDP per capita constant 2015 US$")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_placebo_test(
    placebo_results: dict[str, SCMResult],
    treated_unit: str = "CHL",
    title: str = "Country placebo tests",
    save_path: str | None = None,
):
    """
    Plot country placebo test results (Figure B in appendix).

    Args:
        placebo_results: Dictionary of SCM results from placebo_test
        treated_unit: ISO3 code of treated country
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all placebos in gray
    for country, result in placebo_results.items():
        if country != treated_unit:
            ax.plot(
                result.gap.index,
                result.gap,
                color="gray",
                alpha=0.3,
                linewidth=1,
            )

    # Plot treated unit in bold
    treated_result = placebo_results[treated_unit]
    ax.plot(
        treated_result.gap.index,
        treated_result.gap,
        "b-",
        linewidth=2.5,
        label="Chile",
    )

    ax.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)
    ax.axhline(y=0, color="black", alpha=0.3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Deviations = Effective - Synthetic")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_p_values(
    p_values: pd.Series,
    treatment_effect: pd.Series,
    title: str = "Effects of treatment and p-values",
    save_path: str | None = None,
):
    """
    Plot treatment effects with p-values (Figure 11/C).

    Args:
        p_values: Series of p-values by year
        treatment_effect: Series of treatment effects (gaps)
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    years = treatment_effect.index
    post_years = [y for y in years if y >= TREATMENT_YEAR]

    # Bar plot of treatment effects
    colors = ["gray" if p > 0.1 else "darkblue" for p in p_values.loc[post_years]]
    bars = ax.bar(post_years, treatment_effect.loc[post_years], color=colors, alpha=0.8)

    # Add p-value annotations
    for year, bar in zip(post_years, bars):
        p_val = p_values.loc[year]
        ax.annotate(
            f"{p_val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", alpha=0.3)
    ax.set_xlabel("Post-Treatment Years")
    ax.set_ylabel("Difference from placebos' synthetic avg.")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_jackknife(
    jackknife_results: dict[str, SCMResult],
    title: str = "Jackknife permutation tests (leave-one-out)",
    save_path: str | None = None,
):
    """
    Plot leave-one-out robustness test results (Figure D).

    Args:
        jackknife_results: Dictionary of results from jackknife_test
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get full result
    full_result = jackknife_results.get("full")
    if full_result:
        ax.plot(
            full_result.actual.index,
            full_result.actual,
            "b-",
            label="Real Chile",
            linewidth=2.5,
        )

    # Plot each jackknife synthetic
    colors = plt.cm.Set2(np.linspace(0, 1, len(jackknife_results) - 1))

    for (key, result), color in zip(jackknife_results.items(), colors):
        if key != "full":
            country = key.replace("w/o_", "")
            label = f"w/o {COUNTRY_NAMES.get(country, country)}"
            ax.plot(
                result.synthetic.index,
                result.synthetic,
                "--",
                color=color,
                label=label,
                linewidth=1.5,
                alpha=0.8,
            )

    ax.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Real GDP per capita constant 2015 US$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_bsts_result(
    result: BSTSResult,
    title: str = "Bayesian Structural Time Series",
    save_path: str | None = None,
):
    """
    Plot BSTS/CausalImpact results (Figure 12).

    Args:
        result: BSTSResult from fit_bsts_model
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    years = result.actual.index

    # Left panel: actual vs predicted with CI
    ax1.plot(years, result.actual, "b-", label="Real Chile", linewidth=2)
    ax1.plot(years, result.predicted, "r--", label="Predicted Chile", linewidth=2)
    ax1.fill_between(
        years,
        result.predicted_lower,
        result.predicted_upper,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    ax1.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Real GDP per capita constant 2015 US$")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    # Right panel: treatment effect
    ax2.plot(years, result.pointwise_effect, "r--", label="Treatment Effect", linewidth=2)
    ax2.fill_between(
        years,
        result.effect_lower,
        result.effect_upper,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    ax2.axvline(x=TREATMENT_YEAR, color="black", linestyle=":", alpha=0.7)
    ax2.axhline(y=0, color="black", alpha=0.3)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Real GDP per capita constant 2015 US$")
    ax2.legend(loc="lower left")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_growth_rates(
    result: SCMResult,
    title: str = "GDP growth rates",
    save_path: str | None = None,
):
    """
    Plot actual vs synthetic growth rates (Figure 13).

    Args:
        result: SCMResult from fit_synthetic_control
        title: Plot title
        save_path: Path to save figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate growth rates
    actual_growth = result.actual.pct_change() * 100
    synthetic_growth = result.synthetic.pct_change() * 100

    post_years = [y for y in result.actual.index if y > TREATMENT_YEAR]

    x = np.arange(len(post_years))
    width = 0.35

    # Bar plot
    bars1 = ax.bar(
        x - width / 2,
        actual_growth.loc[post_years],
        width,
        label="Actual Chile",
        color="steelblue",
    )
    bars2 = ax.bar(
        x + width / 2,
        synthetic_growth.loc[post_years],
        width,
        label="Synthetic",
        color="indianred",
    )

    # Average lines
    actual_avg = actual_growth.loc[post_years].mean()
    synthetic_avg = synthetic_growth.loc[post_years].mean()

    ax.axhline(y=actual_avg, color="steelblue", linestyle="--", alpha=0.7)
    ax.axhline(y=synthetic_avg, color="black", linestyle="--", alpha=0.7)

    # Annotations
    ax.annotate(
        f"{actual_avg:.1f}%",
        xy=(len(post_years) - 0.5, actual_avg),
        fontsize=10,
    )
    ax.annotate(
        f"{synthetic_avg:.1f}%",
        xy=(len(post_years) - 0.5, synthetic_avg),
        fontsize=10,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(post_years)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    # Test visualizations
    setup_style()
    print("Visualization module loaded successfully")

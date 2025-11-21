#!/usr/bin/env python3
"""
Main script for replicating Chilean Growth Slowdown analysis.
Toni, Paniagua & Ordenes (2023)

Usage:
    uv run replicate.py              # Run full analysis
    uv run replicate.py --data-only  # Only fetch and save data
    uv run replicate.py --scm-only   # Only run SCM (requires data)
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from chilean_growth.logger import logger
from chilean_growth.config import (
    DONOR_POOL_GROUP_II,
    TREATMENT_YEAR,
    TREATED_COUNTRY,
    COUNTRY_NAMES,
    RANDOM_SEED,
)
from chilean_growth.data_loader import assemble_panel_data
from chilean_growth.synthetic_control import (
    fit_synthetic_control,
    placebo_test,
    jackknife_test,
    in_time_placebo,
    compute_p_values,
)
from chilean_growth.causal_impact import fit_bsts_model, summarize_impact
from chilean_growth.visualization import (
    plot_scm_main,
    plot_scm_trend,
    plot_decomposition,
    plot_placebo_test,
    plot_p_values,
    plot_jackknife,
    plot_bsts_result,
    plot_growth_rates,
)


def create_directories():
    """Create output directories."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def load_or_fetch_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load data from cache or fetch from sources.

    Args:
        force_refresh: If True, fetch fresh data even if cache exists

    Returns:
        Panel data with all countries and variables
    """
    data_path = Path("data/raw/panel_data.csv")

    if data_path.exists() and not force_refresh:
        logger.info(f"Loading cached data from {data_path}")
        return pd.read_csv(data_path)

    logger.info("Fetching data from sources...")
    df = assemble_panel_data(save_path=data_path)
    return df


def run_main_analysis(df: pd.DataFrame) -> dict:
    """
    Run main SCM analysis.

    Args:
        df: Panel data with countries and years

    Returns:
        Dictionary with SCM results keyed by 'main'
    """
    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC CONTROL METHOD ANALYSIS")
    logger.info("=" * 60)

    # Main SCM estimation
    logger.info("\nFitting main SCM model...")
    result = fit_synthetic_control(
        df=df,
        outcome_var="gdp_per_capita",
        treated_unit=TREATED_COUNTRY,
        treatment_year=TREATMENT_YEAR,
        donor_pool=DONOR_POOL_GROUP_II,
    )

    # Print results
    logger.info("\n" + "-" * 40)
    logger.info("OPTIMAL WEIGHTS")
    logger.info("-" * 40)
    for country, weight in result.weights.items():
        name = COUNTRY_NAMES.get(country, country)
        logger.info(f"  {name}: {weight:.3f}")

    logger.info("\n" + "-" * 40)
    logger.info("MODEL FIT")
    logger.info("-" * 40)
    logger.info(f"  Pre-treatment RMSPE:  {result.rmspe_pre:.2f}")
    logger.info(f"  Post-treatment RMSPE: {result.rmspe_post:.2f}")
    logger.info(f"  RMSPE Ratio:          {result.rmspe_ratio:.2f}")

    logger.info("\n" + "-" * 40)
    logger.info("TREATMENT EFFECT")
    logger.info("-" * 40)
    final_year = result.actual.index[-1]
    actual_final = result.actual.iloc[-1]
    synthetic_final = result.synthetic.iloc[-1]
    gap_final = result.gap.iloc[-1]
    pct_gap = (gap_final / synthetic_final) * 100

    logger.info(f"  Year {final_year}:")
    logger.info(f"    Actual Chile:    ${actual_final:,.0f}")
    logger.info(f"    Synthetic Chile: ${synthetic_final:,.0f}")
    logger.info(f"    Gap:             ${gap_final:,.0f} ({pct_gap:.1f}%)")

    # Growth rate comparison
    actual_growth = result.actual.pct_change() * 100
    synthetic_growth = result.synthetic.pct_change() * 100

    post_years = [y for y in result.actual.index if y > TREATMENT_YEAR]
    avg_actual_growth = actual_growth.loc[post_years].mean()
    avg_synthetic_growth = synthetic_growth.loc[post_years].mean()
    growth_gap = avg_synthetic_growth - avg_actual_growth

    logger.info(f"\n  Average GDP growth ({post_years[0]}-{post_years[-1]}):")
    logger.info(f"    Actual:    {avg_actual_growth:.1f}%")
    logger.info(f"    Synthetic: {avg_synthetic_growth:.1f}%")
    logger.info(f"    Gap:       {growth_gap:.1f}%")

    logger.info("\n" + "-" * 40)
    logger.info("PREDICTOR BALANCE")
    logger.info("-" * 40)
    logger.info(result.predictor_balance.to_string())

    return {"main": result}


def run_robustness_tests(df: pd.DataFrame, main_result) -> dict:
    """
    Run robustness tests.

    Args:
        df: Panel data with countries and years
        main_result: SCMResult from main analysis

    Returns:
        Dictionary with placebo, p-values, and jackknife results
    """
    results = {}

    logger.info("\n" + "=" * 60)
    logger.info("ROBUSTNESS TESTS")
    logger.info("=" * 60)

    # In-time placebo test (2006)
    logger.info("\n1. In-time placebo test (2006)...")
    placebo_2006 = in_time_placebo(
        df=df,
        placebo_year=2006,
        donor_pool=DONOR_POOL_GROUP_II,
    )
    results["placebo_2006"] = placebo_2006
    logger.info(f"   RMSPE ratio: {placebo_2006.rmspe_ratio:.2f}")

    # Country placebo tests
    logger.info("\n2. Country placebo tests...")
    placebo_results = placebo_test(
        df=df,
        donor_pool=DONOR_POOL_GROUP_II,
    )
    results["placebo_countries"] = placebo_results

    # P-values
    p_values = compute_p_values(main_result, placebo_results)
    results["p_values"] = p_values
    logger.info("   P-values by year:")
    for year, p in p_values.items():
        logger.info(f"     {year}: {p:.2f}")

    # Jackknife test
    logger.info("\n3. Jackknife (leave-one-out) tests...")
    jackknife_results = jackknife_test(
        df=df,
        donor_pool=DONOR_POOL_GROUP_II,
    )
    results["jackknife"] = jackknife_results

    return results


def run_bsts_analysis(df: pd.DataFrame) -> dict:
    """
    Run Bayesian Structural Time Series analysis.

    Args:
        df: Panel data with countries and years

    Returns:
        Dictionary with BSTS results keyed by 'bsts'
    """
    logger.info("\n" + "=" * 60)
    logger.info("BAYESIAN STRUCTURAL TIME SERIES (CAUSALIMPACT)")
    logger.info("=" * 60)

    result = fit_bsts_model(
        df=df,
        outcome_var="gdp_per_capita",
        treated_unit=TREATED_COUNTRY,
        treatment_year=TREATMENT_YEAR,
    )

    logger.info(summarize_impact(result))

    return {"bsts": result}


def generate_figures(results: dict):
    """
    Generate all figures.

    Args:
        results: Dictionary containing SCM, robustness, and BSTS results
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    main_result = results.get("main")

    if main_result:
        # Figure 8 (left): Main SCM result
        logger.info("  Figure 8 (left): Main SCM result...")
        plot_scm_main(main_result, save_path="reports/figures/fig8_left_scm_main.png")

        # Figure 8 (right): HP filtered trends
        logger.info("  Figure 8 (right): HP filtered trends...")
        plot_scm_trend(main_result, save_path="reports/figures/fig8_right_scm_trend.png")

        # Figure 9: Decomposition
        logger.info("  Figure 9: Decomposition...")
        plot_decomposition(main_result, save_path="reports/figures/fig9_decomposition.png")

        # Figure 13: Growth rates
        logger.info("  Figure 13: Growth rates...")
        plot_growth_rates(main_result, save_path="reports/figures/fig13_growth_rates.png")

    # Robustness figures
    placebo_results = results.get("placebo_countries")
    if placebo_results:
        logger.info("  Figure B: Country placebos...")
        plot_placebo_test(placebo_results, save_path="reports/figures/figB_placebos.png")

    p_values = results.get("p_values")
    if p_values is not None and main_result:
        logger.info("  Figure C: P-values...")
        plot_p_values(
            p_values,
            main_result.gap,
            save_path="reports/figures/figC_pvalues.png",
        )

    jackknife_results = results.get("jackknife")
    if jackknife_results:
        logger.info("  Figure D: Jackknife...")
        plot_jackknife(jackknife_results, save_path="reports/figures/figD_jackknife.png")

    # BSTS figure
    bsts_result = results.get("bsts")
    if bsts_result:
        logger.info("  Figure 12: BSTS result...")
        plot_bsts_result(bsts_result, save_path="reports/figures/fig12_bsts.png")

    logger.info("\nFigures saved to figures/")


def save_results(results: dict):
    """
    Save numerical results to CSV.

    Args:
        results: Dictionary containing SCM and BSTS results
    """
    main_result = results.get("main")

    if main_result:
        # Save synthetic control series
        series_df = pd.DataFrame({
            "year": main_result.actual.index,
            "actual": main_result.actual.values,
            "synthetic": main_result.synthetic.values,
            "gap": main_result.gap.values,
        })
        series_df.to_csv("results/scm_series.csv", index=False)

        # Save weights
        main_result.weights.to_csv("results/scm_weights.csv")

        # Save predictor balance
        main_result.predictor_balance.to_csv("results/predictor_balance.csv")

    logger.info("Results saved to results/")


def main():
    parser = argparse.ArgumentParser(
        description="Replicate Chilean Growth Slowdown analysis"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only fetch and save data",
    )
    parser.add_argument(
        "--scm-only",
        action="store_true",
        help="Only run SCM (skip BSTS)",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness tests",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force refresh data from sources",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    logger.info(f"Random seed set to {RANDOM_SEED}")

    # Setup
    create_directories()

    # Load data
    df = load_or_fetch_data(force_refresh=args.refresh_data)

    if args.data_only:
        logger.info("\nData fetched and saved. Exiting.")
        return

    # Run analyses
    all_results = {}

    # Main SCM
    scm_results = run_main_analysis(df)
    all_results.update(scm_results)

    # Robustness tests
    if not args.skip_robustness:
        robustness_results = run_robustness_tests(df, scm_results["main"])
        all_results.update(robustness_results)

    # BSTS
    if not args.scm_only:
        bsts_results = run_bsts_analysis(df)
        all_results.update(bsts_results)

    # Generate figures
    generate_figures(all_results)

    # Save results
    save_results(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

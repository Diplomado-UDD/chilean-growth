"""
Data loading utilities for Chilean Growth Slowdown replication.
Fetches data from World Bank API and other sources.
"""

from pathlib import Path

import pandas as pd
import wbgapi as wb

from .config import (
    COUNTRY_NAMES,
    DONOR_POOL_GROUP_II,
    POST_TREATMENT_END,
    PRE_TREATMENT_START,
    TREATED_COUNTRY,
    WB_INDICATORS,
)


def fetch_world_bank_data(
    countries: list[str] | None = None,
    start_year: int = PRE_TREATMENT_START,
    end_year: int = POST_TREATMENT_END,
) -> pd.DataFrame:
    """
    Fetch World Bank indicators for specified countries.

    Args:
        countries: List of ISO3 country codes
        start_year: Start year for data
        end_year: End year for data

    Returns:
        DataFrame with indicators as columns, (country, year) as index
    """
    if countries is None:
        countries = [TREATED_COUNTRY] + DONOR_POOL_GROUP_II

    all_data = []

    for var_name, indicator_code in WB_INDICATORS.items():
        try:
            df = wb.data.DataFrame(
                indicator_code,
                economy=countries,
                time=range(start_year, end_year + 1),
            )

            # Reshape from wide to long format
            # wbgapi returns country as index, years as columns (YR1990, YR1991, etc.)
            df = df.reset_index()
            df = df.melt(id_vars=["economy"], var_name="year", value_name=var_name)
            df.columns = ["country", "year", var_name]

            # Convert year from 'YR1990' format to int
            df["year"] = df["year"].str.replace("YR", "").astype(int)

            all_data.append(df)
            print(f"Fetched {var_name}: {len(df)} obs")

        except Exception as e:
            print(f"Error fetching {var_name}: {e}")

    # Merge all indicators
    result = all_data[0]
    for df in all_data[1:]:
        result = result.merge(df, on=["year", "country"], how="outer")

    return result.sort_values(["country", "year"]).reset_index(drop=True)


def load_penn_world_table(filepath: str | Path) -> pd.DataFrame:
    """
    Load Penn World Table data for gross capital formation.

    Download PWT from: https://www.rug.nl/ggdc/productivity/pwt/

    Args:
        filepath: Path to PWT Excel or CSV file

    Returns:
        DataFrame with gross capital formation data
    """
    filepath = Path(filepath)

    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath, sheet_name="Data")
    else:
        df = pd.read_csv(filepath)

    # Select relevant columns
    cols = ["countrycode", "year", "csh_i"]
    df = df[cols].copy()
    df.columns = ["country", "year", "gross_capital_formation"]

    # Convert share to percentage
    df["gross_capital_formation"] = df["gross_capital_formation"] * 100

    return df


def load_trade_openness(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load trade openness data from Our World in Data or generate proxy.

    Trade openness = (Exports + Imports) / GDP * 100

    Args:
        filepath: Path to trade data CSV

    Returns:
        DataFrame with trade openness data
    """
    if filepath is not None:
        return pd.read_csv(filepath)

    # Fetch from World Bank as proxy
    # NE.TRD.GNFS.ZS = Trade (% of GDP)
    countries = [TREATED_COUNTRY] + DONOR_POOL_GROUP_II

    try:
        df = wb.data.DataFrame(
            "NE.TRD.GNFS.ZS",
            economy=countries,
            time=range(PRE_TREATMENT_START, POST_TREATMENT_END + 1),
        )

        df = df.reset_index()
        df = df.melt(id_vars=["economy"], var_name="year", value_name="trade_openness")
        df.columns = ["country", "year", "trade_openness"]
        df["year"] = df["year"].str.replace("YR", "").astype(int)

        return df

    except Exception as e:
        print(f"Error fetching trade openness: {e}")
        return pd.DataFrame()


def load_schooling_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load mean years of schooling from UNDP HDI database.

    Download from: https://hdr.undp.org/data-center/documentation-and-downloads

    Args:
        filepath: Path to HDI data

    Returns:
        DataFrame with mean years of schooling
    """
    if filepath is not None:
        df = pd.read_csv(filepath)
        return df[["country", "year", "mean_years_schooling"]]

    # Generate proxy using World Bank secondary enrollment
    # SE.SEC.ENRR = School enrollment, secondary (% gross)
    countries = [TREATED_COUNTRY] + DONOR_POOL_GROUP_II

    try:
        df = wb.data.DataFrame(
            "SE.SEC.ENRR",
            economy=countries,
            time=range(PRE_TREATMENT_START, POST_TREATMENT_END + 1),
        )

        df = df.reset_index()
        df = df.melt(id_vars=["economy"], var_name="year", value_name="secondary_enrollment")
        df.columns = ["country", "year", "secondary_enrollment"]
        df["year"] = df["year"].str.replace("YR", "").astype(int)

        # Convert to proxy for mean years (rough approximation)
        # This is a simplified proxy - actual data should be used
        df["mean_years_schooling"] = df["secondary_enrollment"] / 10

        return df[["country", "year", "mean_years_schooling"]]

    except Exception as e:
        print(f"Error fetching schooling data: {e}")
        return pd.DataFrame()


def assemble_panel_data(
    pwt_path: str | Path | None = None,
    trade_path: str | Path | None = None,
    schooling_path: str | Path | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Assemble complete panel dataset for SCM analysis.

    Args:
        pwt_path: Path to Penn World Table data
        trade_path: Path to trade openness data
        schooling_path: Path to schooling data
        save_path: Path to save assembled data

    Returns:
        Complete panel DataFrame
    """
    print("Fetching World Bank data...")
    df = fetch_world_bank_data()

    print("Fetching trade openness...")
    trade_df = load_trade_openness(trade_path)
    if not trade_df.empty:
        df = df.merge(trade_df, on=["year", "country"], how="left")

    print("Fetching schooling data...")
    school_df = load_schooling_data(schooling_path)
    if not school_df.empty:
        df = df.merge(school_df, on=["year", "country"], how="left")

    # Load PWT if available
    if pwt_path is not None:
        print("Loading Penn World Table...")
        pwt_df = load_penn_world_table(pwt_path)
        df = df.merge(pwt_df, on=["year", "country"], how="left")
    else:
        # Use World Bank gross capital formation as proxy
        print("Using World Bank gross capital formation proxy...")
        try:
            countries = [TREATED_COUNTRY] + DONOR_POOL_GROUP_II
            gcf_df = wb.data.DataFrame(
                "NE.GDI.TOTL.ZS",  # Gross capital formation (% of GDP)
                economy=countries,
                time=range(PRE_TREATMENT_START, POST_TREATMENT_END + 1),
            )
            gcf_df = gcf_df.reset_index()
            gcf_df = gcf_df.melt(
                id_vars=["economy"], var_name="year", value_name="gross_capital_formation"
            )
            gcf_df.columns = ["country", "year", "gross_capital_formation"]
            gcf_df["year"] = gcf_df["year"].str.replace("YR", "").astype(int)

            df = df.merge(gcf_df, on=["year", "country"], how="left")
            print(f"Fetched gross_capital_formation: {len(gcf_df)} obs")
        except Exception as e:
            print(f"Error fetching GCF: {e}")

    # Add country names
    df["country_name"] = df["country"].map(COUNTRY_NAMES)

    # Sort and clean
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Save if path provided
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")

    print(f"Panel data assembled: {len(df)} observations")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")

    return df


def create_scm_matrices(
    df: pd.DataFrame,
    outcome_var: str = "gdp_per_capita",
    predictor_vars: list[str] | None = None,
    treated_unit: str = TREATED_COUNTRY,
    treatment_year: int = 2014,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create matrices for SCM optimization.

    Args:
        df: Panel data
        outcome_var: Name of outcome variable
        predictor_vars: List of predictor variable names
        treated_unit: ISO3 code of treated country
        treatment_year: Year of treatment

    Returns:
        X0: Predictor matrix for control units (K x J)
        X1: Predictor vector for treated unit (K x 1)
        Y0: Outcome matrix for control units (T0 x J)
        Y1: Outcome vector for treated unit (T0 x 1)
    """
    if predictor_vars is None:
        predictor_vars = [
            "gdp_per_capita",
            "population_growth",
            "life_expectancy",
            "adolescent_fertility",
            "birth_rate",
            "gov_consumption",
        ]

    # Split treated and control
    df_treated = df[df["country"] == treated_unit].copy()
    df_control = df[df["country"] != treated_unit].copy()

    # Pre-treatment period
    pre_treat = df_treated["year"] < treatment_year
    pre_years = df_treated[pre_treat]["year"].values

    # Create predictor matrices (pre-treatment averages)
    X1_dict = {}
    X0_dict = {}

    for var in predictor_vars:
        # Treated unit average
        X1_dict[var] = df_treated[pre_treat][var].mean()

        # Control units averages
        control_avgs = (
            df_control[df_control["year"] < treatment_year]
            .groupby("country")[var]
            .mean()
        )
        X0_dict[var] = control_avgs

    X1 = pd.Series(X1_dict)
    X0 = pd.DataFrame(X0_dict).T

    # Create outcome matrices
    Y1 = df_treated[pre_treat].set_index("year")[outcome_var]

    Y0 = (
        df_control[df_control["year"] < treatment_year]
        .pivot(index="year", columns="country", values=outcome_var)
    )

    return X0, X1, Y0, Y1


if __name__ == "__main__":
    # Test data loading
    df = assemble_panel_data(save_path="data/panel_data.csv")
    print(df.head())

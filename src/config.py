"""
Configuration parameters for the Chilean Growth Slowdown replication.
Based on Toni, Paniagua & Ordenes (2023).
"""

# Treatment configuration
TREATMENT_YEAR = 2014
PRE_TREATMENT_START = 1990
PRE_TREATMENT_END = 2013
POST_TREATMENT_END = 2019

# Treated unit
TREATED_COUNTRY = "CHL"  # Chile ISO3 code

# Donor pool - Group II (expanded pool with better pre-treatment fit)
DONOR_POOL_GROUP_II = [
    "ARG",  # Argentina
    "AUS",  # Australia
    "BOL",  # Bolivia
    "BRA",  # Brazil
    "CAN",  # Canada
    "CHN",  # China
    "COL",  # Colombia
    "CRI",  # Costa Rica
    "DOM",  # Dominican Republic
    "ECU",  # Ecuador
    "GTM",  # Guatemala
    "HND",  # Honduras
    "MEX",  # Mexico
    "NIC",  # Nicaragua
    "PAN",  # Panama
    "PER",  # Peru
    "PHL",  # Philippines
    "PRT",  # Portugal
    "ZAF",  # South Africa
    "ESP",  # Spain
    "USA",  # United States
    "URY",  # Uruguay
]

# Donor pool - Group I (Latin America only)
DONOR_POOL_GROUP_I = [
    "ARG",
    "BOL",
    "BRA",
    "COL",
    "CRI",
    "DOM",
    "ECU",
    "GTM",
    "HND",
    "NIC",
    "PAN",
    "PER",
    "URY",
]

# World Bank indicator codes
WB_INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.KD",  # GDP per capita (constant 2015 US$)
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",  # GDP per capita PPP (constant 2017 int $)
    "population_growth": "SP.POP.GROW",  # Population growth (annual %)
    "life_expectancy": "SP.DYN.LE00.IN",  # Life expectancy at birth
    "adolescent_fertility": "SP.ADO.TFRT",  # Adolescent fertility rate
    "birth_rate": "SP.DYN.CBRT.IN",  # Crude birth rate
    "gov_consumption": "NE.CON.GOVT.ZS",  # Government consumption (% of GDP)
}

# Penn World Table variables (to be loaded from CSV)
PWT_VARIABLES = {
    "gross_capital_formation": "csh_i",  # Share of gross capital formation
}

# Predictor variables for SCM (averages over pre-treatment period)
PREDICTOR_VARIABLES = [
    "gdp_per_capita",
    "population_growth",
    "life_expectancy",
    "adolescent_fertility",
    "birth_rate",
    "gov_consumption",
    "gross_capital_formation",
    "trade_openness",
    "mean_years_schooling",
]

# Special outcome lags to include as predictors
OUTCOME_LAGS = [1990, 1995, 2000, 2005, 2010, 2013]

# Country name mapping
COUNTRY_NAMES = {
    "ARG": "Argentina",
    "AUS": "Australia",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHN": "China",
    "CHL": "Chile",
    "COL": "Colombia",
    "CRI": "Costa Rica",
    "DOM": "Dominican Republic",
    "ECU": "Ecuador",
    "GTM": "Guatemala",
    "HND": "Honduras",
    "MEX": "Mexico",
    "NIC": "Nicaragua",
    "PAN": "Panama",
    "PER": "Peru",
    "PHL": "Philippines",
    "PRT": "Portugal",
    "ZAF": "South Africa",
    "ESP": "Spain",
    "USA": "United States",
    "URY": "Uruguay",
}

# Hodrick-Prescott filter parameter for yearly data
HP_LAMBDA = 100

# MCMC simulations for Bayesian estimation
MCMC_SAMPLES = 10000

# Random seed for reproducibility
RANDOM_SEED = 42

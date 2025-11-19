# Chilean Growth Slowdown Analysis

Replication code for **"Policy Changes and Growth Slowdown: Assessing the Lost Decade of the Latin American Miracle"** by Toni, Paniagua & Ordenes (2023).

## Overview

This project implements Synthetic Control Method (SCM) and Bayesian Structural Time Series (BSTS) analyses to quantify the impact of Chile's 2014 policy regime change on economic growth.

### Key Findings from the Paper

- **Two-thirds** of Chile's growth slowdown attributable to internal policy factors
- **One-third** attributable to external factors (end of commodity super-cycle)
- Nearly **10% reduction** in real GDP per capita over five years (2014-2019)
- **1.8% decline** in average GDP growth rates from 2015-2019

## Project Structure

```
chilean-growth/
    main.py                    # Entry point (placeholder)
    replicate.py               # Main replication script
    pyproject.toml             # Project dependencies
    T-P-O_Policy_Changes.pdf   # Original paper
    src/
        config.py              # Configuration parameters
        data_loader.py         # Data fetching utilities
        synthetic_control.py   # SCM implementation
        causal_impact.py       # BSTS/CausalImpact models
        visualization.py       # Figure generation
    data/                      # Cached panel data
    figures/                   # Output figures
    results/                   # Numerical results
```

## Methodology

### Synthetic Control Method

Constructs a counterfactual "Synthetic Chile" as a weighted combination of donor countries that best matches Chile's pre-treatment (1990-2013) economic trajectory. The treatment effect is the divergence between actual Chile and synthetic Chile post-2014.

**Donor Pool (Group II - Expanded)**:
- Latin America: Argentina, Bolivia, Brazil, Colombia, Costa Rica, Dominican Republic, Ecuador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Peru, Uruguay
- Commodity exporters: Australia, China, South Africa
- Colonial ties: Spain, Portugal
- Trade partners: Canada, Philippines, USA

**Predictor Variables**:
- GDP per capita (with lags: 1990, 1995, 2000, 2005, 2010, 2013)
- Population growth, life expectancy, adolescent fertility, birth rate
- Government consumption, gross capital formation
- Trade openness, mean years of schooling

### Bayesian Structural Time Series

Uses state-space models with control series regression to generate probabilistic counterfactual predictions with credible intervals (95% CI from 10,000 MCMC simulations).

## Installation

Requires Python 3.12+. Uses `uv` for package management.

```bash
# Clone repository
git clone <repository-url>
cd chilean-growth

# Install dependencies
uv sync

# Or if starting fresh
uv init
uv add cvxpy matplotlib numpy pandas scikit-learn scipy statsmodels wbgapi
```

## Usage

### Full Analysis

```bash
# Run complete analysis (SCM + robustness + BSTS + figures)
uv run replicate.py
```

### Selective Runs

```bash
# Only fetch and cache data
uv run replicate.py --data-only

# Only run SCM (skip BSTS)
uv run replicate.py --scm-only

# Skip robustness tests (faster)
uv run replicate.py --skip-robustness

# Force refresh data from sources
uv run replicate.py --refresh-data
```

## Output

### Figures

| Figure | Description |
|--------|-------------|
| `fig8_left_scm_main.png` | Main SCM result: Actual vs Synthetic Chile |
| `fig8_right_scm_trend.png` | HP-filtered trends |
| `fig9_decomposition.png` | Internal/external factors decomposition |
| `fig12_bsts.png` | BSTS counterfactual with confidence intervals |
| `fig13_growth_rates.png` | Actual vs synthetic GDP growth rates |
| `figB_placebos.png` | Country placebo tests |
| `figC_pvalues.png` | Treatment effects with p-values |
| `figD_jackknife.png` | Leave-one-out robustness tests |

### Results

- `results/scm_series.csv` - Time series of actual, synthetic, and gap
- `results/scm_weights.csv` - Optimal country weights
- `results/predictor_balance.csv` - Predictor balance table

## Robustness Tests

1. **In-time placebo** (2006): Tests for spurious effects at non-treatment dates
2. **Country placebo**: Applies SCM to each donor country
3. **P-values**: Statistical significance of treatment effects
4. **Jackknife**: Leave-one-out permutation tests

## Data Sources

- **World Bank WDI**: GDP per capita, demographic indicators
- **Penn World Table**: Gross capital formation
- **UNDP HDI**: Mean years of schooling
- **Our World in Data**: Trade openness

Data is fetched automatically via `wbgapi` and cached in `data/panel_data.csv`.

## Dependencies

- `cvxpy` - Convex optimization for SCM weights
- `matplotlib` - Visualization
- `numpy`, `pandas` - Data manipulation
- `scipy` - Optimization algorithms
- `statsmodels` - Time series models, HP filter
- `wbgapi` - World Bank data API

## References

- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391-425.
- Brodersen, K., et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1), 247-274.
- Toni, E., Paniagua, P., & Ordenes, P. (2023). Policy Changes and Growth Slowdown: Assessing the Lost Decade of the Latin American Miracle. *SSRN Electronic Journal*. DOI: 10.2139/ssrn.4640416

## License

Research replication code. See original paper for methodological details.

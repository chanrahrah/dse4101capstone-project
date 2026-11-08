# Sector ETF Volatility Forecasting

This project studies whether macro-financial predictors improve volatility forecasts
for XLB, XLE, XLF, XLI and XLY, and whether those forecasts improve volatility-managed
portfolios. The out-of-sample period is March 2008 to December 2025.

## Repository structure

```text
.
├── dataset/
│   ├── price/                 # Raw ETF data, return cleaning and monthly variance
│   └── macro_variables/       # Raw macro series, alignment and transformations
├── models/
│   ├── research_config.py     # Selected models, sample and comparison blocks
│   ├── forecasting/
│   │   ├── build/             # b0–b3: forecast estimation
│   │   ├── forecasts/         # Model forecast CSVs
│   │   ├── evaluation/        # e1–e6: forecast comparisons
│   │   ├── results/           # Forecast evaluation tables
│   │   └── robustness/        # Crisis and rolling-window analyses
│   ├── portfolio/             # p1–p3, portfolio settings and results
│   ├── utils/                 # Shared paths and portfolio functions
│   └── archive/               # Earlier experiments
├── requirements.txt
└── run_pipeline.py
```

## Data setup

The pipeline starts from the supplied raw CSVs. It does not download CRSP data.
The following files must be present:

```text
dataset/price/raw/sector_etf_prices_with_xlre.csv
dataset/price/raw/risk_free_rate_returns.csv
dataset/macro_variables/raw/   # Fourteen saved macro series
```

Price-cleaning scripts create daily excess returns and monthly realised variance.
Macro scripts align observations to month-end information and produce eleven predictors.

## Installation

From the repository root, install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run the analysis

Rebuild from the supplied raw data:

```bash
python3 run_pipeline.py --start raw
```

The execution order is:

**Data preparation → EDA → b0–b3 → e1–e6 → forecast robustness checks → p1–p3 → rp1–rp2**

b2 and b3 also generate the rolling-window forecasts. To re-estimate every model
without reusing saved GARCH fits, add `--no-fit-cache`.

To rerun data preparation and downstream analyses using existing forecasts:

```bash
python3 run_pipeline.py --start raw --reuse-forecasts
```

To download fresh macro observations first, use `--start fetch` instead. This requires
`FRED_API_KEY` in your environment or the repository-level `.env` file. Both modes
require the CRSP exports above.

## Outputs

- Main forecasts: `models/forecasting/forecasts/`
- Forecast evaluation tables: `models/forecasting/results/`
- Portfolio returns and weights: `models/portfolio/results/portfolios/`
- Portfolio performance tables: `models/portfolio/results/eval/`
- Research robustness results: `models/forecasting/robustness/results/` and `models/portfolio/robustness/results/`

Runs overwrite processed data and result CSVs. Source notebook displays can be older
snapshots; the runner prints where it saves fresh executed copies. Full model fitting can take
substantial time. Preview the execution order with `--dry-run`.

See [the model guide](models/README.md) and [the portfolio guide](models/portfolio/README.md)
for individual notebook descriptions and manual execution.

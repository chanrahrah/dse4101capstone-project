# Capstone Project

## Structure

- `dataset/` — raw and processed data files
- `models/` — forecasting and portfolio research notebooks, shared configuration, and results; see [the notebook map and run order](models/README.md)
- `models/archive/` — earlier model scripts, portfolio experiments, and forecast exports

## Getting Started

Clone the repo and run:

```
pip install -r requirements.txt
```

## Run the full pipeline

Fetch macro data and run data preparation, b0–b3, e1–e6, and p1–p3:

```bash
python3 run_pipeline.py --start fetch
```

Or rebuild everything from the existing raw data, without downloading:

```bash
python3 run_pipeline.py --start raw
```

Both modes require the existing CRSP exports `sector_etf_prices_with_xlre.csv` and
`risk_free_rate_returns.csv` in `dataset/price/raw/`. Fetch mode also requires
`FRED_API_KEY`. Raw mode requires the saved macro series in `dataset/macro_variables/raw/`.
The configured data dates are retained; fetch mode does not extend the research sample.

Outputs overwrite the existing data, forecast and result files. Forecasts are rebuilt,
with GARCH's existing fit cache retained. Each notebook runs in a fresh kernel using the
runner's Python environment; source notebooks are not modified. Notebook displays are
executed in memory rather than saved. The runner prints each file and stops on the first
error. Full model fitting can take substantial time.

## Refresh Macro Data

Set `FRED_API_KEY` in your environment or a repository-level `.env` file,
then run these commands from the repository root:

```bash
python3 dataset/macro_variables/fetch_fred_data.py
python3 dataset/macro_variables/transform_macro_data.py
```

The fetch script writes the combined monthly panel and a dropped-series report; duplicate
category panels are not exported. It writes API data to `dataset/macro_variables/raw/` and the
monthly panel to `dataset/macro_variables/clean/fred_monthly_panel.csv`.
The transform script uses that panel for all 11 predictors and writes
`final_macro_data.csv` and `adf_test_results.csv` to
`dataset/macro_variables/final_clean/`.

## Fast pipeline checks

```bash
python3 -m unittest discover -s tests -v
```

These checks validate saved data, model/ETF/month coverage, forecast targets, comparison blocks,
and the forecast-to-portfolio joins without downloading or fitting models. See
[the repository audit](models/AUDIT.md) for verification details and research limitations.

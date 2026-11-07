# Forecasting and portfolio research

Can macro-aware volatility forecasts improve out-of-sample accuracy for cyclical-sector ETFs
relative to historical-volatility benchmarks, and do these gains translate into better
risk-adjusted performance in volatility-managed portfolios?

## File map

```
models/
├── README.md
├── research_config.py
├── utils/
│   ├── paths.py
│   └── portfolio_utils.py
├── forecasting/
│   ├── build/
│   │   ├── b0_hist_rw.ipynb
│   │   ├── b1_garch_garchmidas.ipynb
│   │   ├── b2_elastic_net.ipynb
│   │   └── b3_xgboost.ipynb
│   ├── forecasts/                 tidy forecasts and GARCH-MIDAS coefficients
│   ├── evaluation/
│   │   ├── e1_garch_family.ipynb
│   │   ├── e2_macro_value.ipynb
│   │   ├── e3_vix_effect_en.ipynb
│   │   ├── e4_vix_effect_xgb.ipynb
│   │   ├── e5_linear_vs_nonlinear.ipynb
│   │   └── e6_forecast_summary.ipynb
│   └── results/                   e1–e6 losses and evaluation CSVs
├── portfolio/
│   ├── README.md
│   ├── config.py
│   ├── p1_build_portfolios.ipynb
│   ├── p2_portfolio_eval.ipynb
│   ├── p3_forecast_vs_portfolio.ipynb
│   └── results/
└── archive/
    ├── scripts/                   earlier model and portfolio work, with original outputs
    ├── forecasts/                 legacy root forecast exports
    └── dataset/price/             superseded price-cleaning pipeline
```

`research_config.py` defines the ETFs, sample window, selected models, benchmark and shared
comparison blocks. `portfolio/config.py` holds portfolio-only settings. `utils/paths.py` resolves
paths from the checkout; start notebook kernels anywhere inside this repository. Forecast tests
remain in their evaluation notebooks; no `forecast_eval.py` helper is required.

## Research questions and notebook map

| Question | Notebook | Shared block |
|---|---|---|
| Does macro information help within GARCH? | [e1](forecasting/evaluation/e1_garch_family.ipynb) | B1 |
| Does the full macro set improve Elastic Net over past volatility alone? | [e2](forecasting/evaluation/e2_macro_value.ipynb) | B2 |
| How much does VIX contribute in Elastic Net? | [e3](forecasting/evaluation/e3_vix_effect_en.ipynb) | B3 |
| Does the VIX finding hold in XGBoost? | [e4](forecasting/evaluation/e4_vix_effect_xgb.ipynb) | B4 |
| Does XGBoost outperform Elastic Net with matched macro inputs? | [e5](forecasting/evaluation/e5_linear_vs_nonlinear.ipynb) | B5 |
| Which models beat the historical-volatility benchmark? | [e6](forecasting/evaluation/e6_forecast_summary.ipynb) | Selected models vs benchmark |
| Do forecast gains improve portfolio performance? | [p1](portfolio/p1_build_portfolios.ipynb), [p2](portfolio/p2_portfolio_eval.ipynb), [p3](portfolio/p3_forecast_vs_portfolio.ipynb) | B0–B5 |

## Run order

To run the complete workflow automatically, use `python3 run_pipeline.py --start fetch`
from the repository root, or `--start raw` to rebuild from saved raw data without downloads.
See the [pipeline instructions](../README.md#run-the-full-pipeline) for prerequisites and outputs.

1. Install the dependencies in the repository's `requirements.txt` and prepare `dataset/`.
2. Review `research_config.py` and `portfolio/config.py`. Restart kernels after changing imported configuration.
3. Run **b0 → b1 → b2 → b3** to produce the benchmark, GARCH, Elastic Net and XGBoost forecasts.
4. Run **e1 → e2 → e3 → e4 → e5 → e6** for forecast evaluation. e6 is retained because p3 reads its tables.
5. Run **p1 → p2 → p3** to build portfolios, evaluate performance and connect it to forecast accuracy.

The active forecast files are `hist_rw_forecasts.csv`, `garch_forecasts.csv`, `en_forecasts.csv`
and `xgb_forecasts.csv`. Build notebooks write them directly to `forecasting/forecasts/`.
Evaluation notebooks write to `forecasting/results/`; existing table names and schemas are
preserved, including the `e6_losses.csv` and `e6_evaluation.csv` inputs to p3. Portfolio outputs stay
under `portfolio/results/portfolios/` and `portfolio/results/eval/`.

Archived files retain their historical contents and may use historical paths. They are excluded
from this run order. The notebook output cells are historical snapshots; rerun the relevant notebooks to refresh them.
See [AUDIT.md](AUDIT.md) for the pipeline checks and remaining research limitations.

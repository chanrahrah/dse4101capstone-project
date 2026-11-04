# Portfolio construction and evaluation

**Research question:** can macro-aware volatility forecasting improve out-of-sample forecast
accuracy for cyclical-sector ETFs relative to historical-volatility benchmarks, and do these gains
translate into better risk-adjusted performance in volatility-managed portfolios?

This folder turns each model's monthly variance forecasts into volatility-managed portfolios of
the five cyclical ETFs (XLB, XLE, XLF, XLI, XLY), following Moreira & Muir (2017). It then answers
the research question one comparison block at a time, on both the portfolio side and the forecast
side.

## Where things are

```
models/
├── research_config.py              shared: ETFs, sample, SELECTED models, BENCHMARK, BLOCKS
├── utils/portfolio_utils.py        every function: loading, construction, tests, plots
├── forecasting/
│   ├── build/b0_hist_rw.ipynb            builds the historical-volatility benchmark forecast
│   └── forecasts/                  tidy forecast files, one per model family
└── portfolio/
    ├── config.py                   portfolio settings: K_CAP, SPECS, MAIN_SPEC, FORECAST_COL
    ├── p1_build_portfolios.ipynb   build and save every portfolio, no evaluation
    ├── p2_portfolio_eval.ipynb    portfolio results, block by block
    ├── p3_forecast_vs_portfolio.ipynb  forecast results next to portfolio results
    └── results/
        ├── portfolios/             written by 01
        └── eval/                   written by 02 and 03
```

The active forecast files are in `models/forecasting/forecasts/`, written directly by the build
notebooks. Earlier scripts, portfolio work and root forecast exports are preserved under
`models/archive/` and are not part of the active run order.

## How to run

1. Edit `research_config.py` or `config.py` if needed.
2. Run the notebooks in order: **p1 → p2 → p3**, with the kernel started anywhere inside this checkout. 02 takes about a minute; the others take a few seconds.
3. 02 and 03 read the models, pairs and window from `results/portfolios/run_info.json`, so they
   always match the last run of 01. 03 also reads the tables saved by 02.

After editing a config file, restart the kernel of any open notebook, since Python caches imported
modules.

## The comparison blocks

Each block in `BLOCKS` asks one sub-question. 02 answers it with Ledoit-Wolf tests on the
portfolios, and e1–e5 evaluate the matching forecast pairs; p3 links e6 benchmark comparisons to portfolio results.

**Sign convention:** each pair is (a, b), with a the richer model. A positive difference always
means the addition helps: a has the higher Sharpe ratio, or the lower forecast loss.

| Block | Question | Pairs (a vs b) |
|---|---|---|
| B0 | Does simple volatility timing beat BH, and does fitting a historical model improve on it? | B0.1 `garch` vs `hist_rw` |
| B1 | Does macro help within GARCH? | B1.1 `garch_midas_baa10y` vs `garch` |
| B2 | Do macro + VIX help a linear model that uses only past volatility? | B2.1 `en_full_macro` vs `en_no_macro` |
| B3 | Effect of VIX in Elastic Net: is VIX doing all the work? | B3.1 `en_full_macro` vs `en_no_vix` · B3.2 `en_vix_only` vs `en_no_macro` · B3.3 `en_full_macro` vs `en_vix_only` |
| B4 | Effect of VIX in XGBoost (robustness check for B3) | B4.0 `xgb_full_macro` vs `xgb_no_macro` · B4.1 `xgb_full_macro` vs `xgb_no_vix` · B4.2 `xgb_vix_only` vs `xgb_no_macro` · B4.3 `xgb_full_macro` vs `xgb_vix_only` |
| B5 | With the same information, does XGBoost beat Elastic Net? | B5.1 no macro · B5.2 macro without VIX · B5.3 VIX only · B5.4 full macro |

A model missing from its forecast file is skipped with a notice by 01, together with the pairs that
use it.

## Method

- **Weight:** wᵢ,ₜ = min(σ̄²ᵢ / σ̂²ᵢ,ₜ₊₁, 2), where σ̂² is the model's forecast (`fc_level`) of next
  month's variance.
- **Variance target:** monthly RV is the sum of squared daily excess returns, matching every forecast family.
- **Reference σ̄², two specifications:**
  - **fixed** (main): the 2003-03 to 2008-02 mean, the same every month. Only the forecast moves
    the weights, so comparing portfolios across models tests forecast skill;
  - **expanding** (robustness): the mean up to month t. It adds re-levering after crises, which
    the **control** portfolio, min(σ̄²ₜ / σ̄²_fixed, 2) with no forecast, isolates.
- **Portfolio:** the 1/N average of the ETFs' managed excess returns, rescaled to buy-and-hold
  volatility. Each ETF's managed series is likewise rescaled to its own buy-and-hold volatility.
- **Benchmarks:** buy-and-hold (BH, w = 1) asks whether volatility timing works at all. `hist_rw`
  (σ̂²ₜ₊₁ = RVₜ, the Moreira & Muir specification) asks whether a forecast adds anything beyond the
  simplest volatility timing.
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months), the same for every model. Each
  forecast row dated t is the forecast for month t+1; 01 checks this every run.

## What each notebook does

| Notebook | Sections | Tests |
|---|---|---|
| 01 | Load forecasts and returns; check the date alignment; σ̄² under both specifications; build every portfolio and the control; weight diagnostics; save | none |
| 02 | **A** cyclical bucket: overview, one section per block (table against BH and `hist_rw`, pair tests, growth plot, verdicts), scoreboard. **B** per ETF: heatmaps against the benchmarks and for the pairs, full tables. **C** expanding σ̄²: do the pair results hold; timing vs re-levering (control regressions) | Ledoit-Wolf (2008) SR tests; α on BH (Newey-West) |
| 03 | Links e6 (forecast) and 02 (portfolio), each model against `hist_rw`: **1** pair-level link (forecast t against Ledoit-Wolf t); **2** ranks (forecast rank vs Sharpe ratio rank); **3** why gains do or do not carry through: scale, weight distance and cap, calm vs stress months | none new (reads e6 and 02); Spearman ρ |

The full forecast evaluation is in `models/forecasting/evaluation/`. Forecast accuracy against
`hist_rw` is `e6_forecast_summary.ipynb` (DM for the non-nested models, Clark-West
for Elastic Net, which nests `hist_rw`); run it before 03, which reads `forecasting/results/e6_evaluation.csv`.
03 runs no tests of its own.

## Outputs (`results/eval/`)

| File | Written by | Contents |
|---|---|---|
| `vs_benchmarks_bucket_<spec>.csv` | 02 | Every bucket portfolio against BH (performance, LW, α) and `hist_rw` |
| `vs_benchmarks_per_etf_<spec>.csv` | 02 | The same per ETF |
| `pairs.csv` | 02 | Every block-pair LW test (spec, block, pair, a, b, ETF) |
| `scoreboard_bucket.csv` | 02 | Portfolio answer per pair, both specifications |
| `extension.csv` | 02 | Control regressions under the expanding σ̄² |
| `forecast_portfolio_link.csv` | 03 | Forecast (e6) and portfolio (02) result of every model against `hist_rw`, per unit |
| `rank_summary_<spec>.csv` | 03 | Best models and Spearman ρ, with the scale-free version |
| `pair_diagnostics.csv` | 03 | Weight distance, cap and calm/stress results per model against `hist_rw` |

## Adding a model or a comparison

1. Put its tidy forecast file in `models/forecasting/forecasts/` (columns `model, etf, date, target_date,
   actual_logRV, actual_RV, fc_log, sigma2, fc_level`; a row dated t forecasts t+1). Its first
   forecast must be for 2008-03 or earlier.
2. If it is a new family, add the file to `FORECAST_FILES` in `utils/portfolio_utils.py`.
3. Add the variant to `SELECTED`, and any pairs to a block in `BLOCKS`, in `research_config.py`.
4. Rerun 01 → 03.

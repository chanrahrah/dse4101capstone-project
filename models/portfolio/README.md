# Portfolio construction and evaluation

**Research question:** can macro-aware volatility forecasting improve out-of-sample forecast
accuracy for cyclical-sector ETFs relative to historical-volatility benchmarks, and do these gains
translate into better risk-adjusted performance in volatility-managed portfolios?

This folder answers the second half. It turns each model's monthly variance forecasts into
volatility-managed portfolios of the five cyclical ETFs (XLB, XLE, XLF, XLI, XLY), following
Moreira & Muir (2017), evaluates them block by block, and sets the portfolio results next to the
forecast results.

## Where things are

```
models/
├── research_config.py                  ETFs, sample, SELECTED models, BENCHMARK, BLOCKS
├── utils/
│   ├── paths.py                        project paths, resolved from the checkout
│   ├── forecast_data.py                forecast files, coverage and target checks
│   └── portfolio_utils.py              loading, construction, tests and plots
├── forecasting/
│   ├── build/                          forecast notebooks (b0_hist_rw builds the benchmark)
│   ├── forecasts/                      tidy forecast files, one per model family (read by p1)
│   ├── evaluation/                     forecast evaluation, e1 to e6
│   └── results/                        e6_losses.csv and e6_evaluation.csv (read by p3)
└── portfolio/
    ├── config.py                       FORECAST_COL, K_CAP, SPECS, MAIN_SPEC
    ├── p1_build_portfolios.ipynb       builds and saves every portfolio
    ├── p2_portfolio_eval.ipynb         evaluates the portfolios, block by block
    ├── p3_forecast_vs_portfolio.ipynb  ranks the models by forecast accuracy and by portfolio performance
    ├── results/
    │   ├── portfolios/                 written by p1
    │   └── eval/                       written by p2 and p3
    └── robustness/
        ├── rp1_different_leverages.ipynb   the leverage cap k = 1.5, 3 and no cap
        ├── rp2_timing_vs_relevering.ipynb  expanding σ̄²: timing or re-levering?
        └── results/                        written by rp1 and rp2
```

Every path comes from `utils/paths.py`, so the notebooks run with the kernel started anywhere
inside the checkout and always write to `portfolio/results/` (the robustness notebooks to
`portfolio/robustness/results/`). Older scripts and portfolio work are kept in `models/archive/`
and are not part of the run order.

## How to run

For the complete data-to-portfolio workflow, use the root [notebook runner](../../run_all.ipynb).
It includes p1–p3 and rp1–rp2, with report
Table 6, Tables D1–D11 and Figures 2–4 mapped to their source sections and saved results.
Runs overwrite the normal research outputs.

1. Edit `research_config.py` or `config.py` if needed, then restart the kernel of any open
   notebook (Python caches imported modules).
2. Run **p1 → p2**. p1 takes a few seconds, p2 about a minute.
3. Run `models/forecasting/evaluation/e6_naive_vs_all_models.ipynb` if the forecasts have changed,
   then **p3**, which reads e6's results and p2's tables.
4. The root `run_pipeline.py` runs p1–p3 and both robustness notebooks automatically.
5. Robustness checks: **rp1** and **rp2** in `robustness/`, in any order after p1. Each needs
   only p1's saved portfolios and takes under half a minute.

p2 and p3 read the models, pairs and window from `results/portfolios/run_info.json`.
p3 checks model coverage in p2/e6; that check alone cannot detect stale numeric results.
Rerun the root pipeline after changing forecasts before using the portfolio tables.

## Method

- **Weight:** wᵢ,ₜ = min(σ̄²ᵢ / σ̂²ᵢ,ₜ₊₁, 2), where σ̂² is the model's forecast (`fc_level`) of next
  month's realised variance.
- **Realised variance:** the sum of squared daily excess returns in the month, the same target as
  every forecast family.
- **Reference σ̄², two specifications:**
  - **fixed** (main): the ETF's mean monthly RV over the training window, 2003-03 to 2008-02, the
    same every month. Only the forecast moves the weights, so comparing portfolios across models
    tests forecast skill.
  - **expanding** (robustness): the mean from 2003-03 up to month t. It also re-levers the
    portfolio after crises; the **control** portfolio, min(σ̄²ᵢ,ₜ / σ̄²ᵢ,fixed, 2) with no forecast,
    isolates that effect (rp2).
- **Leverage cap:** k = 2 in the main results; rp1 rebuilds the fixed-σ̄² portfolios with k = 1.5,
  k = 3 and no cap.
- **Portfolio:** the 1/N average of the ETFs' managed excess returns, rescaled to buy-and-hold
  volatility over the out-of-sample period. For the per-ETF results, each ETF's managed series is
  rescaled to its own buy-and-hold volatility.
- **Benchmarks:** buy-and-hold (BH, w = 1) asks whether volatility timing works at all. `hist_rw`
  (σ̂²ₜ₊₁ = RVₜ, the Moreira & Muir portfolio) asks whether a forecast adds anything beyond the
  simplest volatility timing.
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months), the same for every model. Each
  forecast row dated t is the forecast for month t+1; p1 checks this on every run.

## The comparison blocks

Each block in `BLOCKS` (`research_config.py`) asks one sub-question. p2 answers it on the portfolio
side with Ledoit-Wolf tests; e1 to e5 test the same pairs on the forecast side.

**Sign convention:** each pair is (a, b), with a the richer model. A positive difference means the
addition helps: a has the higher Sharpe ratio, or the lower forecast loss.

| Block | Question | Pairs (a vs b) |
|---|---|---|
| B0 | Does simple volatility timing beat BH, and does fitting a historical model improve on it? | B0.1 `garch` vs `hist_rw` |
| B1 | Does macro help within GARCH? | B1.1 `garch_midas_baa10y` vs `garch` |
| B2 | Do macro + VIX help a linear model that uses only past volatility? | B2.1 `en_full_macro` vs `en_no_macro` |
| B3 | Effect of VIX in Elastic Net: is VIX doing all the work? | B3.1 `en_full_macro` vs `en_no_vix` · B3.2 `en_vix_only` vs `en_no_macro` · B3.3 `en_full_macro` vs `en_vix_only` |
| B4 | Effect of VIX in XGBoost (robustness check for B3) | B4.0 `xgb_full_macro` vs `xgb_no_macro` · B4.1 `xgb_full_macro` vs `xgb_no_vix` · B4.2 `xgb_vix_only` vs `xgb_no_macro` · B4.3 `xgb_full_macro` vs `xgb_vix_only` |
| B5 | With matched macro inputs, does XGBoost beat Elastic Net? | B5.1 no macro · B5.2 macro without VIX · B5.3 VIX only · B5.4 full macro + VIX (`xgb_*` vs `en_*`) |

Every selected model must cover the full out-of-sample window for all five ETFs.
p1 stops if forecasts are missing or their realised values disagree with the price data.

## What each notebook does

| Notebook | Contents | Tests |
|---|---|---|
| p1 | Loads the forecasts and returns; checks the date alignment; computes σ̄² under both specifications; builds every portfolio and the control; weight diagnostics (per ETF and pooled); saves | none |
| p2 | **A** cyclical bucket: overview, one section per block (table against BH and `hist_rw`, pair tests, growth plot, verdicts), scoreboard. **B** per ETF: heatmaps against the benchmarks and for the pairs, full tables | Ledoit-Wolf (2008) Sharpe ratio tests; α on BH (Newey-West) |
| p3 | Ranks every model against `hist_rw`, pooled and per ETF: **1** forecast accuracy by QLIKE, with e6's log-RV MSE tests; **2** portfolio performance by Sharpe ratio, with p2's Ledoit-Wolf test; **3** forecast rank vs portfolio rank (slope chart, Spearman ρ) | none of its own: Diebold-Mariano and Clark-West from e6 (Clark-West for Elastic Net only, which nests `hist_rw`), Ledoit-Wolf from p2 |
| rp1 | Rebuilds every model's bucket under the fixed σ̄² with k = 1.5, 2, 3 and no cap (the k = 2 rebuild must match p1); ΔSR vs BH and α on BH per model and cap, each with its p-value | Ledoit-Wolf vs BH; α on BH (Newey-West) |
| rp2 | Expanding vs fixed σ̄² over time; control regressions: what the forecast adds beyond BH and re-levering under the expanding σ̄² | α on BH and the control (Newey-West) |

## Outputs

`results/portfolios/`, written by p1:

| File | Contents |
|---|---|
| `fixed/`, `expanding/` | Per specification: `returns.csv` (BH and each managed bucket), `positions.csv` and `weights.csv` (per ETF), `c.csv` (rescale factors), `sigma2_bar.csv`. `expanding/` also holds the control |
| `xret.csv` | Monthly excess returns of each ETF |
| `weight_diagnostics.csv` | Weight diagnostics per specification, model and ETF |
| `weight_diagnostics_pooled.csv` | The same, pooled across ETFs |
| `run_info.json` | Settings of the run (models, pairs, window), read by p2 and p3 |

`results/eval/`, written by p2 and p3:

| File | Written by | Contents |
|---|---|---|
| `vs_benchmarks_bucket_<spec>.csv` | p2 | Every bucket portfolio against BH (performance, Ledoit-Wolf, α) and against `hist_rw` |
| `vs_benchmarks_per_etf_<spec>.csv` | p2 | The same per ETF |
| `pairs.csv` | p2 | Every block-pair Ledoit-Wolf test (spec, block, pair, a, b, ETF) |
| `scoreboard_bucket.csv` | p2 | The portfolio answer per pair, under both specifications |
| `forecast_ranking.csv` | p3 | Models ranked by QLIKE per unit, with the QLIKE gain % vs `hist_rw` and e6's DM and Clark-West p-values on the log-RV MSE |
| `portfolio_ranking_<spec>.csv` | p3 | Models ranked by Sharpe ratio per unit, with ΔSR vs `hist_rw` and the Ledoit-Wolf p-value (main σ̄²) |
| `rank_summary_<spec>.csv` | p3 | Spearman ρ between accuracy and Sharpe ratio, with and without `hist_rw`, and `hist_rw`'s ranks (main σ̄²) |

`<spec>` is `fixed` or `expanding`; p3 writes the main specification (`fixed`) only. In p3 the unit
`Cyclical` is the pooled bucket.

`robustness/results/`, written by rp1 and rp2:

| File | Written by | Contents |
|---|---|---|
| `rp1_different_leverages.csv` | rp1 | Per cap and model: SR, ΔSR and Ledoit-Wolf p vs BH, α on BH, % capped, max weight, c |
| `rp2_timing_vs_relevering.csv` | rp2 | Control regressions under the expanding σ̄² |

## Adding a model or a comparison

1. Put its tidy forecast file in `models/forecasting/forecasts/`, with columns `model, etf, date,
   target_date, actual_logRV, actual_RV, fc_log, sigma2, fc_level`; a row dated t forecasts month
   t+1. Its first forecast must be for 2008-03 or earlier.
2. If it is a new model family, add the file to `FORECAST_FILES` in `utils/forecast_data.py`.
3. Add the variant to `SELECTED`, and any pairs to a block in `BLOCKS`, in `research_config.py`.
4. Rerun e6 → p1 → p2 → p3.

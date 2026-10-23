# Portfolio Construction (short guide)

Turns monthly volatility forecasts into volatility-managed portfolios of five cyclical ETFs
(XLB, XLE, XLF, XLI, XLY), following Moreira & Muir (2017). Full details: [README.md](README.md).

## Files

| File | Purpose |
|---|---|
| `config.py` | Settings: models (`SELECTED`), cap `K_CAP = 2`, training window, `SPECS`, model pairs (`COMPARISONS`) |
| `portfolio_utils.py` | Shared functions |
| `01_build_portfolios.ipynb` | Builds and saves every portfolio. No evaluation |
| `02_eval_vs_buy_and_hold.ipynb` | Does volatility timing beat buy-and-hold? |
| `03_eval_model_comparisons.ipynb` | Does one forecast give a better portfolio than another? (research question) |
| `04_forecast_vs_portfolio.ipynb` | Do more accurate forecasts give better portfolios? |

## How to run

1. Edit `config.py` if needed.
2. Run the notebooks in order: 01 → 02 → 03 → 04. Run them from this folder.
3. 01 writes `results/portfolios/`; 02–04 write their tables to `results/eval/`.

## Method

- **Weight:** w = min(σ̄² / σ̂², 2), where σ̂² is the model's forecast of next month's variance.
- **Two references σ̄²:**
  - **fixed** (main): the 2003-03 to 2008-02 mean, the same every month;
  - **expanding** (robustness): the mean up to the current month.
- **Portfolio:** the 1/N average of the ETFs' managed excess returns, rescaled to buy-and-hold
  volatility. Buy-and-hold uses w = 1.
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months), the same for every model.
- **Control (expanding only):** weight min(σ̄²_t / σ̄²_fixed, 2) with no forecast. It isolates the
  re-levering caused by the moving reference.

## Tests

| Notebook | Test |
|---|---|
| 02 | Ledoit-Wolf test of SR(model) = SR(BH); α on BH (Newey-West); drawdown, turnover, % at the cap. Extension: α on BH and the control |
| 03 | Ledoit-Wolf test of SR(a) = SR(b) for each pair in `COMPARISONS`, under both references |
| 04 | QLIKE and MSE forecast losses, ranks against Sharpe ranks, Spearman ρ |

## Adding a model

1. Put its tidy forecast file in `forecasts/`.
2. Add it to `FORECAST_FILES` in `portfolio_utils.py`.
3. Add it to `SELECTED`, and any pairs to `COMPARISONS`, in `config.py`.
4. Rerun 01–04.

Its first forecast must be for 2008-03 or earlier, otherwise 01 stops with an error.

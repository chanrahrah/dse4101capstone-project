# %% [markdown]
# ## Vintage variable start dates
#
# We use vintage variables; combined usable data range: **2001-03-31 to 2025-12-31**
#
# | Variable    | Vintage start date |
# |-------------|---------------------|
# | T10Y2Y      | 2000-01-31          |
# | DGS3MO      | 2000-01-31          |
# | BAA10Y      | 2000-01-31          |
# | VIXCLS      | 2000-01-31          |
# | UMCSENT     | 2000-01-31          |
# | DCOILWTICO  | 2000-01-31          |
# | CPIAUCSL    | 2000-02-29          |
# | M2SL        | 2000-02-29          |
# | INDPRO      | 2000-02-29          |
# | UNRATE      | 2000-02-29          |
# | PCEPI       | 2000-08-31          |
# | CFNAI       | 2001-03-31          |
# %% [markdown]
# ## Imports

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids the native-window crash entirely
import matplotlib.pyplot as plt

# %% [markdown]
# ## Paths & constants
#
# - Macro data now comes pre-cleaned and pre-transformed from the vintage pipeline
#   (`dataset/macro_variables/transform_macro_data.py`) — this file **is** `macro_transformed`.
# - Price data now comes pre-cleaned as a wide daily panel with both raw and excess returns.
# - Macro variable set changed: `NFCI`, `CFNAI`, `PCEPI`, `ICSA` are gone; `DGS3MO`,
#   `CPIAUCSL`, `T5YIE`, `T5YIE_MINUS_T10YIE`, `UMCSENT`, `NBUSBIS` are new.

# %%
PROJECT_ROOT = "/Users/lycfaith/Documents/GitHub/dse4101capstone-project"  # VERIFY: set to wherever this file sits relative to dataset/
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

MACRO_PATH = os.path.join(DATASET_DIR, "macro_variables", "final_clean", "final_macro_data.csv")
ADF_RESULTS_PATH = os.path.join(DATASET_DIR, "macro_variables", "final_clean", "adf_test_results.csv")
PRICE_PATH = os.path.join(DATASET_DIR, "price", "clean", "daily_returns_panel.csv")

FORECASTS_DIR = os.path.join(DATASET_DIR, "forecasts")
os.makedirs(FORECASTS_DIR, exist_ok=True)

CYCLICAL_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLY"]  # VERIFY: matches your forecasts/ folder

FINAL_MACRO_VARS = [
    "T10Y2Y", "DGS3MO", "BAA10Y", "CPIAUCSL", "T5YIE", "T5YIE_MINUS_T10YIE",
    "VIXCLS", "INDPRO", "UNRATE", "UMCSENT", "DCOILWTICO", "NBUSBIS",
]

SAMPLE_START = pd.Timestamp("2003-01-31")  # matches the vintage panel's own START_DATE
MIN_TRAIN = 60
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

# %% [markdown]
# ## Load vintage macro data
#
# Replaces the old raw macro load + ffill + `VAR_TRANSFORMS`/`apply_transform` + per-window
# ADF loop. Transformations are fixed ex ante upstream, per the professor's guidance: levels
# for `T10Y2Y`, `DGS3MO`, `BAA10Y`, `T5YIE`, `T5YIE_MINUS_T10YIE`, `VIXCLS`, `UNRATE`,
# `UMCSENT`; log-diff for `CPIAUCSL`, `INDPRO`, `NBUSBIS`, `DCOILWTICO`. No re-testing per
# rolling window — the ADF results below are loaded for documentation only.

# %%
macro_transformed = pd.read_csv(MACRO_PATH, index_col="date", parse_dates=True)
macro_transformed = macro_transformed[macro_transformed.index >= SAMPLE_START]

assert set(FINAL_MACRO_VARS) == set(macro_transformed.columns), (
    "FINAL_MACRO_VARS must match the columns actually present in final_macro_data.csv"
)

print(f"macro_transformed: {macro_transformed.index.min().date()} to {macro_transformed.index.max().date()}, "
      f"{macro_transformed.shape[0]} rows, {macro_transformed.isnull().sum().sum()} total nulls")

adf_reference = pd.read_csv(ADF_RESULTS_PATH, index_col="variable")
adf_reference

# %% [markdown]
# ## Load vintage price data
#
# Replaces the raw CRSP read + pivot. The panel is already a clean wide daily matrix with
# both raw (`ret_*`) and excess (`xret_*`) returns. We use excess returns for realized variance.

# %%
price_panel = pd.read_csv(PRICE_PATH, parse_dates=["date"]).set_index("date").sort_index()

xret_cols = {ticker: f"xret_{ticker}" for ticker in CYCLICAL_TICKERS}
missing_tickers = [t for t, col in xret_cols.items() if col not in price_panel.columns]
if missing_tickers:
    raise ValueError(f"Missing excess-return columns for: {missing_tickers}")

daily_returns = price_panel[[xret_cols[t] for t in CYCLICAL_TICKERS]].copy()
daily_returns.columns = CYCLICAL_TICKERS

print("daily_returns shape:", daily_returns.shape)
print("date range:", daily_returns.index.min().date(), "to", daily_returns.index.max().date())
daily_returns.head()

# %% [markdown]
# ## Monthly realized variance
#
# `min_count=1` matters here: without it, a month with no valid daily returns silently sums
# to `0.0` instead of `NaN`, and `log(0) = -inf` downstream. This turns a genuinely empty
# month into `NaN` instead, which is easy to spot and handle explicitly.

# %%
monthly_rv_squared = daily_returns.pow(2).resample("ME").sum(min_count=1)
log_rv_squared = np.log(monthly_rv_squared)

print("monthly_rv_squared shape:", monthly_rv_squared.shape)
print("date range:", monthly_rv_squared.index.min().date(), "to", monthly_rv_squared.index.max().date())
print()
print("--- monthly RV², first 5 months ---")
print(monthly_rv_squared.head())
print()
print("--- log(RV²), first 5 months ---")
print(log_rv_squared.head())
print()
print("--- log(RV²) summary stats per ticker ---")
print(log_rv_squared.describe())
print()
print("Months with no valid RV² per ticker (NaN, not silently 0):")
print(monthly_rv_squared.isnull().sum())

# %% [markdown]
# ## Align macro + own-lag features with target
#
# `macro_transformed`'s index and `log_rv_squared`'s index are both month-end now, so the
# inner join lines up correctly  this was a real risk with the old raw macro source, which
# used month-start dates.

# %%
aligned_data = {}
for ticker in CYCLICAL_TICKERS:
    own_lag = log_rv_squared[ticker].rename("own_log_rv_sq_lag0")
    y = log_rv_squared[ticker].shift(-1)
    y.name = "y_log_rv_squared_tplus1"
    df = macro_transformed.join(own_lag, how="inner").join(y, how="inner").dropna()
    aligned_data[ticker] = df
    print(f"{ticker}: {df.shape[0]} aligned rows, {df.index.min().date()} to {df.index.max().date()}")

ALL_FEATURE_COLS = FINAL_MACRO_VARS + ["own_log_rv_sq_lag0"]  # 13 features now (was 11)
print()
print(f"{len(ALL_FEATURE_COLS)} features per model:", ALL_FEATURE_COLS)

# %% [markdown]
# ## Walk-forward Elastic Net (annual refit)
#
# Refits once every 12 months on all data seen so far (with a fresh 5-fold `TimeSeriesSplit`
# hyperparameter search each time), then holds those coefficients and hyperparameters fixed
# while forecasting the next 12 months one at a time. Train 1–60 → forecast 61–72 (fixed
# coefficients) → retrain 1–72 → forecast 73–84 → ... and so on, with a shorter final block
# if the sample doesn't divide evenly by 12.

# %%
REFIT_INTERVAL = 12  # months between refits; hyperparameters + coefficients held fixed within each block

def walk_forward_forecast(df, feature_cols, min_train, refit_interval=REFIT_INTERVAL, window="expanding"):  ## CHANGED: added window param
    if window not in ("expanding", "rolling"):  ## CHANGED
        raise ValueError(f"window must be 'expanding' or 'rolling', got {window!r}")  ## CHANGED

    X = df[feature_cols].to_numpy()
    y = df["y_log_rv_squared_tplus1"].to_numpy()
    dates = df.index
    n = len(df)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        train_end = min_train
        while train_end < n:
            train_start = 0 if window == "expanding" else train_end - min_train  ## CHANGED

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    l1_ratio=L1_RATIOS,
                    cv=TimeSeriesSplit(n_splits=5),
                    max_iter=10000,
                    random_state=0,
                )),
            ])
            pipe.fit(X[train_start:train_end], y[train_start:train_end])  ## CHANGED: was X[:train_end], y[:train_end]
            enet = pipe.named_steps["enet"]

            block_end = min(train_end + refit_interval, n)
            for i in range(train_end, block_end):
                pred = pipe.predict(X[i:i + 1])[0]
                rows.append({
                    "date": dates[i],
                    "log_rv_sq_actual": y[i],
                    "log_rv_sq_pred": pred,
                    "alpha": enet.alpha_,
                    "l1_ratio": enet.l1_ratio_,
                    "refit_train_start": dates[train_start],  ## CHANGED: new field
                    "refit_train_end": dates[train_end - 1],
                })
            train_end = block_end
    return pd.DataFrame(rows).set_index("date")

t0 = time.time()
test_forecast = walk_forward_forecast(aligned_data["XLI"], ALL_FEATURE_COLS, MIN_TRAIN)
print(f"XLI: {len(test_forecast)} forecasts, {time.time()-t0:.1f}s")
test_forecast.head(15)

# %% [markdown]
# ## Run all tickers and save
#

# %%
forecasts = {"XLI": test_forecast}
for ticker in CYCLICAL_TICKERS:
    if ticker in forecasts:
        continue
    t0 = time.time()
    forecasts[ticker] = walk_forward_forecast(aligned_data[ticker], ALL_FEATURE_COLS, MIN_TRAIN)
    print(f"{ticker}: {len(forecasts[ticker])} forecasts, {time.time()-t0:.1f}s")

for ticker, df in forecasts.items():
    print(f"{ticker}: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")

for ticker, df in forecasts.items():
    out = df.copy()
    out["rv2_hat"] = np.exp(out["log_rv_sq_pred"])    # naive exp(); retransformation bias, see below
    out["rv2_actual"] = np.exp(out["log_rv_sq_actual"])
    out_path = os.path.join(FORECASTS_DIR, f"elastic_net_expanding_window_{ticker}_vintage_xret.csv")
    out.to_csv(out_path)
    forecasts[ticker] = out
    print(f"Saved {out_path} ({len(out)} rows)")

forecasts["XLI"].head()

# %% [markdown]
# ## Accuracy metrics
#
# `R2_OOS` follows the standard Campbell–Thompson out-of-sample R² definition: model squared
# error vs. an expanding historical-mean benchmark that uses the exact same information set.

# %%
def historical_mean_benchmark(aligned_df, min_train):
    y_full = aligned_df["y_log_rv_squared_tplus1"]
    n = len(y_full)
    benchmark = [y_full.iloc[:i].mean() for i in range(min_train, n)]
    return pd.Series(benchmark, index=y_full.index[min_train:n])

accuracy_rows = []
for ticker in CYCLICAL_TICKERS:
    df = forecasts[ticker]
    actual = df["log_rv_sq_actual"]
    pred = df["log_rv_sq_pred"]
    benchmark = historical_mean_benchmark(aligned_data[ticker], MIN_TRAIN)
    benchmark = benchmark.reindex(df.index)

    resid = actual - pred
    mse = (resid ** 2).mean()
    rmse = np.sqrt(mse)
    mae = resid.abs().mean()

    ss_res = (resid ** 2).sum()
    ss_tot_benchmark = ((actual - benchmark) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot_benchmark

    accuracy_rows.append({
        "ticker": ticker,
        "n_forecasts": len(df),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2_OOS": r2_oos,
    })

accuracy_df = pd.DataFrame(accuracy_rows).set_index("ticker")
print(accuracy_df)

# %% [markdown]
# ## Plots
#
# Actual vs. predicted log(RV²) per ticker, and hyperparameter stability (`alpha`, `l1_ratio`)
# across the walk-forward refits.

# %%
fig, axes = plt.subplots(len(CYCLICAL_TICKERS), 1, figsize=(12, 16), sharex=True)
for ax, ticker in zip(axes, CYCLICAL_TICKERS):
    df = forecasts[ticker]
    ax.plot(df.index, df["log_rv_sq_actual"], label="actual", marker="o", markersize=3)
    ax.plot(df.index, df["log_rv_sq_pred"], label="predicted", marker="o", markersize=3)
    ax.set_title(f"{ticker}: actual vs predicted log(RV\u00b2)")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ticker in CYCLICAL_TICKERS:
    df = forecasts[ticker]
    axes[0].plot(df.index, df["alpha"], label=ticker, alpha=0.7)
    axes[1].plot(df.index, df["l1_ratio"], label=ticker, alpha=0.7)
axes[0].set_ylabel("alpha")
axes[0].legend(fontsize=8, ncol=5)
axes[1].set_ylabel("l1_ratio")
axes[1].legend(fontsize=8, ncol=5)
axes[1].set_title("Hyperparameter stability across walk-forward refits")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Run rolling-window forecasts (all tickers)
#
# Same mechanics as the expanding-window run, just with a fixed-length training window.
# Saved under a different filename so it doesn't overwrite your expanding-window results.

# %%
forecasts_rolling = {}
for ticker in CYCLICAL_TICKERS:
    t0 = time.time()
    forecasts_rolling[ticker] = walk_forward_forecast(
        aligned_data[ticker], ALL_FEATURE_COLS, MIN_TRAIN, window="rolling"
    )
    print(f"{ticker} (rolling): {len(forecasts_rolling[ticker])} forecasts, {time.time()-t0:.1f}s")

for ticker, df in forecasts_rolling.items():
    out = df.copy()
    out["rv2_hat"] = np.exp(out["log_rv_sq_pred"])
    out["rv2_actual"] = np.exp(out["log_rv_sq_actual"])
    out_path = os.path.join(FORECASTS_DIR, f"elastic_net_rolling_window_{ticker}_vintage_xret.csv")
    out.to_csv(out_path)
    forecasts_rolling[ticker] = out
    print(f"Saved {out_path} ({len(out)} rows)")

forecasts_rolling["XLI"].head()

# %% [markdown]
# ## Rolling-window accuracy metrics
#
# Reuses the same `historical_mean_benchmark` (expanding historical mean) as the reference
# for both expanding- and rolling-window models, so `R2_OOS` is directly comparable between
# the two — only the model's own errors differ, the benchmark is held fixed.

# %%
accuracy_rows_rolling = []
for ticker in CYCLICAL_TICKERS:
    df = forecasts_rolling[ticker]
    actual = df["log_rv_sq_actual"]
    pred = df["log_rv_sq_pred"]
    benchmark = historical_mean_benchmark(aligned_data[ticker], MIN_TRAIN)
    benchmark = benchmark.reindex(df.index)

    resid = actual - pred
    mse = (resid ** 2).mean()
    rmse = np.sqrt(mse)
    mae = resid.abs().mean()

    ss_res = (resid ** 2).sum()
    ss_tot_benchmark = ((actual - benchmark) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot_benchmark

    accuracy_rows_rolling.append({
        "ticker": ticker, "n_forecasts": len(df),
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2_OOS": r2_oos,
    })

accuracy_df_rolling = pd.DataFrame(accuracy_rows_rolling).set_index("ticker")
print(accuracy_df_rolling)

# %% [markdown]
# ## Compare expanding vs. rolling window
#
# Positive `_diff` values mean expanding window did better for that ticker/metric;
# negative means rolling did better.

# %%
comparison = accuracy_df.add_suffix("_expanding").join(accuracy_df_rolling.add_suffix("_rolling"))
comparison["RMSE_diff_exp_minus_roll"] = comparison["RMSE_expanding"] - comparison["RMSE_rolling"]
comparison["R2_OOS_diff_exp_minus_roll"] = comparison["R2_OOS_expanding"] - comparison["R2_OOS_rolling"]
print(comparison)

# %%
fig, axes = plt.subplots(len(CYCLICAL_TICKERS), 1, figsize=(12, 16), sharex=True)
for ax, ticker in zip(axes, CYCLICAL_TICKERS):
    df_exp = forecasts[ticker]
    df_roll = forecasts_rolling[ticker]
    ax.plot(df_exp.index, df_exp["log_rv_sq_actual"], label="actual", color="black", linewidth=1)
    ax.plot(df_exp.index, df_exp["log_rv_sq_pred"], label="expanding", alpha=0.8)
    ax.plot(df_roll.index, df_roll["log_rv_sq_pred"], label="rolling", alpha=0.8)
    ax.set_title(f"{ticker}: expanding vs rolling window forecasts")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(CYCLICAL_TICKERS))
width = 0.35
ax.bar(x - width/2, accuracy_df.loc[CYCLICAL_TICKERS, "R2_OOS"], width, label="expanding")
ax.bar(x + width/2, accuracy_df_rolling.loc[CYCLICAL_TICKERS, "R2_OOS"], width, label="rolling")
ax.set_xticks(x)
ax.set_xticklabels(CYCLICAL_TICKERS)
ax.set_ylabel("R2_OOS")
ax.set_title("Out-of-sample R\u00b2: expanding vs rolling window")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
fig.tight_layout()
plt.show()

# %%

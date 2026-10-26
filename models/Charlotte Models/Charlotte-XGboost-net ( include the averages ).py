# XGBoost forecasts of next-month realized variance, per ETF, expanding window.
# Question: does macro information help, and does it still help once VIX is removed?
# Run it in VS Code cell by cell, or:  python Charlotte-XGboost-net.py
 
# %% 1. Settings  (the only cell you normally need to edit)
import itertools
import warnings
 
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, DMatrix
from sklearn.model_selection import KFold
 
warnings.filterwarnings("ignore")
 
DATA = "/Users/lycfaith/Documents/GitHub/dse4101capstone-project/dataset"
RETURNS_PATH = DATA + "/price/clean/daily_returns_panel_ver2.csv"
MACRO_PATH = DATA + "/macro_variables/final_clean/final_macro_data.csv"
 
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLY"]
FIRST_MONTH = "2003-02-28"     # first month used as a TARGET. Jan 2003 (one trading day) is never a target;
                               # it is only the own-lag input of the Feb 2003 row, as in Elastic Net.
                               # Use "2003-03-31" to drop it completely (214 forecasts instead of 215).
MIN_TRAIN = 60                 # months in the first training window
RETUNE_EVERY = 24              # re-pick the tuning settings every 24 months (refit every month)
DROP_ONE = False               # True = also drop each macro variable one at a time (run super slow)
 
MACRO = ["T10Y2Y", "DGS3MO", "BAA10Y", "CPIAUCSL", "T5YIE", "T5YIE_MINUS_T10YIE",
         "VIXCLS", "INDPRO", "UNRATE", "UMCSENT", "DCOILWTICO"]
 
# Shallow trees, strong shrinkage, small grid (8 combinations)
GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n)
        for d, lr, n in itertools.product([2, 3], [0.01, 0.03], [100, 300])]                    # 8 dictionaries
FIXED = dict(subsample=0.7, colsample_bytree=0.8, min_child_weight=3, reg_lambda=5,
             n_jobs=1, random_state=4101, verbosity=0)
 
# Past-volatility inputs (HAR style): last month, the average of the last 3 months, the average of the last 12 months
LAGS = ["own_lag", "avg3", "avg12"]                     #own lag is last month value of the thing im predicting
 
# The input sets we compare
no_vix = [m for m in MACRO if m != "VIXCLS"]
INPUT_SETS = {
    "1 Lags only":         LAGS,
    "2 + VIX only":        LAGS + ["VIXCLS"],
    "3 Macro without VIX": LAGS + no_vix,
    "4 All macro":         LAGS + MACRO,
}
if DROP_ONE:
    for m in MACRO:
        INPUT_SETS[f"All minus {m}"] = LAGS + [x for x in MACRO if x != m]
 
 
# %% 2. Build the monthly data: target = log realized variance of month t, inputs known at t-1
daily = pd.read_csv(RETURNS_PATH, parse_dates=["date"]).set_index("date")
month = daily.index.to_period("M").to_timestamp("M")
 
macro = pd.read_csv(MACRO_PATH, parse_dates=["date"]).set_index("date")
macro.index = macro.index + pd.offsets.MonthEnd(0)
macro = macro[MACRO].shift(1)                      # macro from month t-1 predicts month t, shift moves every value down one row
 
data = {}
for etf in ETFS:
    rv = (daily["xret_" + etf] ** 2).groupby(month).sum()                     # excess returns, all months
    df = pd.DataFrame({"log_rv": np.log(rv)})
    past = df["log_rv"].shift(1)                   # everything below uses months before t only
    df["own_lag"] = past                           # last month's log realized variance
    df["avg3"] = past.rolling(3, min_periods=1).mean()     # average of the last 3 months
    df["avg12"] = past.rolling(12, min_periods=1).mean()   # average of the last 12 months
    data[etf] = df.join(macro).dropna().loc[FIRST_MONTH:]  # attaches macro column, keep rows from feb 2003 onwards
print({etf: len(df) for etf, df in data.items()}, "monthly rows per ETF")
 
 
# %% 3. The model
def tune(X, y):
    """Pick the best grid setting with 5-fold CV (no shuffling) inside the training window."""
    best = None
    for params in GRID:
        pred = np.zeros(len(y))
        for train, valid in KFold(5).split(X):
            pred[valid] = XGBRegressor(**FIXED, **params).fit(X[train], y[train]).predict(X[valid])
        mse = np.mean((y - pred) ** 2)
        if best is None or mse < best[0]:
            best = (mse, params, np.var(y - pred, ddof=1)) # divide by n-1 to correct for bias
    return best[1], best[2]                        # best settings, CV residual variance (sigma2)
 
 
def walk_forward(df, inputs):
    """Expanding window: train on months before t, forecast month t, then move on one month."""
    X, y = df[inputs].to_numpy(float), df["log_rv"].to_numpy(float)
    rows = []
    for t in range(MIN_TRAIN, len(df)):
        if (t - MIN_TRAIN) % RETUNE_EVERY == 0:
            params, sigma2 = tune(X[:t], y[:t])
        model = XGBRegressor(**FIXED, **params).fit(X[:t], y[:t])
        pred = model.predict(X[t:t + 1])[0]
        # how much each input pushed this forecast up or down (SHAP values, built into xgboost)
        push = model.get_booster().predict(DMatrix(X[t:t + 1]), pred_contribs=True)[0][:-1]
        rows.append({"date": df.index[t], "actual": y[t], "pred": pred,
                     "level_pred": np.exp(pred + sigma2 / 2),     # log-normal correction
                     "sigma2": sigma2,
                     **{"push_" + c: v for c, v in zip(inputs, push)}})
    return pd.DataFrame(rows).set_index("date")
 
 
def walk_forward_ols(df, inputs):
    """Same expanding window, but plain OLS (no trees, no tuning). The simple HAR-style benchmark."""
    X, y = df[inputs].to_numpy(float), df["log_rv"].to_numpy(float)
    rows = []
    for t in range(MIN_TRAIN, len(df)):
        A = np.column_stack([np.ones(t), X[:t]])                  # add an intercept
        beta = np.linalg.lstsq(A, y[:t], rcond=None)[0]
        sigma2 = np.sum((y[:t] - A @ beta) ** 2) / (t - A.shape[1])   # residual variance of the training window
        pred = np.r_[1.0, X[t]] @ beta
        rows.append({"date": df.index[t], "actual": y[t], "pred": pred,
                     "level_pred": np.exp(pred + sigma2 / 2), "sigma2": sigma2})
    return pd.DataFrame(rows).set_index("date")
 
 
def score(fc, df):
    err = fc["actual"] - fc["pred"]
    benchmark = df["log_rv"].expanding().mean().shift(1).loc[fc.index]   # historical mean
    ratio = np.exp(fc["actual"]) / fc["level_pred"]
    return {"MSE": (err ** 2).mean(),                                    # log RV
            "R2_OOS": 1 - (err ** 2).sum() / ((fc["actual"] - benchmark) ** 2).sum(),
            "QLIKE": (ratio - np.log(ratio) - 1).mean()}                 # RV level
 
 
# %% 4. Run every input set for every ETF (takes several minutes)
forecasts, rows = {}, []
for etf in ETFS:
    for name, inputs in INPUT_SETS.items():
        forecasts[(name, etf)] = walk_forward(data[etf], inputs)
        rows.append({"etf": etf, "inputs": name, **score(forecasts[(name, etf)], data[etf])})
    # OLS benchmark: HAR lags only, no macro
    forecasts[("OLS HAR lags", etf)] = walk_forward_ols(data[etf], LAGS)
    rows.append({"etf": etf, "inputs": "OLS HAR lags", **score(forecasts[("OLS HAR lags", etf)], data[etf])})
    print(etf, "done")
results = pd.DataFrame(rows)
 
 
# %% 5. Results: one table per loss. Compare columns within a row.
for metric, better in [("MSE", "lower"), ("R2_OOS", "higher"), ("QLIKE", "lower")]:
    print(f"\n{metric} ({better} is better)")
    print(results.pivot(index="etf", columns="inputs", values=metric)
          .loc[ETFS, list(INPUT_SETS) + ["OLS HAR lags"]].round(3))
 
# What does VIX add on top of the other macro variables?  (4 minus 3)
cols = ["MSE", "R2_OOS", "QLIKE"]
with_vix = results[results.inputs == "4 All macro"].set_index("etf")[cols]
without_vix = results[results.inputs == "3 Macro without VIX"].set_index("etf")[cols]
print("\nAdding VIX to the other macro variables (set 4 minus set 3):")
print((with_vix - without_vix).loc[ETFS].round(3))
 
 
# %% 6. Save our forecasts in the same layout as EN
# Columns: model, etf, date, actual_logRV, actual_RV, fc_log, sigma2, fc_level
# date = forecast ORIGIN (the month before the month being forecast), as in her file.
# The file is saved automatically on your Desktop (found with ~), so no path to type.
import os
 
NAMES = {"OLS HAR lags": "ols_har_no_macro", "1 Lags only": "xgb_no_macro", "2 + VIX only": "xgb_vix_only",
         "3 Macro without VIX": "xgb_no_vix", "4 All macro": "xgb_full_macro"}
tidy = []
for (name, etf), fc in forecasts.items():
    tidy.append(pd.DataFrame({
        "model": NAMES.get(name, "xgb_" + name), "etf": etf,
        "date": fc.index - pd.offsets.MonthEnd(1),
        "actual_logRV": fc["actual"].values, "actual_RV": np.exp(fc["actual"].values),
        "fc_log": fc["pred"].values, "sigma2": fc["sigma2"].values, "fc_level": fc["level_pred"].values}))
tidy = pd.concat(tidy, ignore_index=True)
 
out_dir = os.path.expanduser("~/Desktop")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "xgb_forecasts_expanding_tidy.csv")
tidy.to_csv(out_file, index=False)
print("\nSaved", tidy.shape, "->", os.path.abspath(out_file))
 
 
# %% 7. How much does each input move the forecast?  (average size of its push, in log variance)
full = "4 All macro"
push = pd.DataFrame({etf: forecasts[(full, etf)].filter(like="push_").abs().mean() for etf in ETFS})
push.index = push.index.str.replace("push_", "")
push["average"] = push.mean(axis=1)
print("\nAverage push of each input on the forecast (set 4):")
print(push.sort_values("average", ascending=False).round(3))
print("\nShare of the total push:")
print((push / push.sum() * 100).sort_values("average", ascending=False).round(1))
 
 
# %% 8. Look at one tree (fit once on the full sample, only to look inside)
TREE_ETF = "XLF"
inputs = INPUT_SETS[full]
tree = XGBRegressor(**FIXED, max_depth=2, learning_rate=0.03, n_estimators=300)
tree.fit(data[TREE_ETF][inputs], data[TREE_ETF]["log_rv"])
print(f"\nFirst tree for {TREE_ETF}:")
print(tree.get_booster().get_dump()[0])            # e.g. [VIXCLS<20] yes=1,no=2 ...
try:                                               # picture needs: brew install graphviz
    import matplotlib.pyplot as plt                #                pip install graphviz
    from xgboost import plot_tree
    plot_tree(tree, num_trees=0)
    plt.show()
except Exception as e:
    print("No picture (Graphviz not installed). The text tree above has the same information.")
 
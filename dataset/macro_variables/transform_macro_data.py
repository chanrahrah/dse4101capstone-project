"""Narrow the leakage-safe FRED monthly panel to the final macro predictor set.

Starting point: clean/fred_monthly_panel.csv, produced by fetch_fred_data.py.
That script already drops series with no ALFRED/vintage history from 2000
onwards (NFCI, STLFSI4, ICSA, DTWEXBGS). This script performs a second,
economics-driven narrowing pass -- reducing collinearity within each FRED
channel -- per Dataset.pdf.

Kept (12 variables, 6 channels):
    Rates/yield curve    T10Y2Y, DGS3MO            -- level vs slope, not collinear
    Credit spreads       BAA10Y                     -- single series
    Inflation            CPIAUCSL, T5YIE,
                         T5YIE - T10YIE (engineered)
    Vol/stress           VIXCLS                     -- single series
    Real activity        INDPRO, UNRATE, UMCSENT
    Dollar/commodity     DCOILWTICO, NBUSBIS, DEXUSEU

Dropped (4 variables):
    PCEPI  -- redundant with CPIAUCSL (corr > 0.9); slower to publish and
              partly derived from CPI itself, so structurally laggier
    T10YIE -- redundant with T5YIE at the level; retained only long enough
              to engineer the T5YIE - T10YIE curve-stress spread, then dropped
    M2SL   -- inflation channel already covered by the rates/yield curve
              channel; M2 growth has weak/noisy short-horizon predictive power
    CFNAI  -- mechanically double-counts INDPRO/UNRATE/UMCSENT (it's a
              composite built from ~85 series including them) and carries a
              3-4 week publication lag, the worst of any real-activity candidate
"""

import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PANEL_PATH = "dataset/macro_variables/clean/fred_monthly_panel.csv"
OUTPUT_DIR = "dataset/macro_variables/final_clean"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_macro_data.csv")
ADF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "adf_test_results.csv")
ADF_SIGNIFICANCE = 0.05

DROP_NOW = ["PCEPI", "M2SL", "CFNAI"]
LOG_DIFF_VARS = ["CPIAUCSL", "INDPRO", "DCOILWTICO", "DEXUSEU"]
START_DATE = pd.Timestamp("2003-01-31") # to follow the first month-end with T5YIE/T10YIE vintage data

FINAL_COLUMNS = [
    "T10Y2Y",
    "DGS3MO",
    "BAA10Y",
    "CPIAUCSL",
    "T5YIE",
    "T5YIE_MINUS_T10YIE",
    "VIXCLS",
    "INDPRO",
    "UNRATE",
    "UMCSENT",
    "DCOILWTICO",
    "DEXUSEU",
    ]


def load_panel():
    return pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)


def drop_redundant_channels(panel):
    """Step 1: drop variables that are redundant within their channel.

    PCEPI, M2SL, and CFNAI are dropped outright (see module docstring).
    T10YIE is kept one more step, since it's needed to engineer the
    T5YIE - T10YIE inflation-curve spread before it's dropped.
    """
    return panel.drop(columns=DROP_NOW).copy()


def engineer_inflation_curve_spread(df):
    """Step 2: T5YIE - T10YIE, then drop T10YIE.

    Near-term minus long-term breakeven inflation expectations. An
    inverting spread (short-term jumping above long-term) signals an
    acute, near-term inflation shock (e.g. an energy spike) outrunning
    anchored long-run expectations -- a distinct signal from the level of
    T5YIE alone. T10YIE is dropped immediately after, since its only role
    here was constructing this spread.
    """
    df = df.copy()
    df["T5YIE_MINUS_T10YIE"] = df["T5YIE"] - df["T10YIE"]
    return df.drop(columns=["T10YIE"])


def apply_transformations(df):
    """Step 3: log-difference the asset-price-like / index-level series.

    Everything else (spreads, rates, VIX, UNRATE, UMCSENT) stays as a
    level -- see the table in Dataset.pdf for the full rationale per
    variable. Log-differences are computed on the full history (before
    slicing to START_DATE) so the first usable month doesn't lose its
    diff to a missing prior observation.
    """
    df = df.copy()
    for col in LOG_DIFF_VARS:
        df[col] = np.log(df[col]).diff()
    return df


def slice_to_start_date(df):
    """Step 4: slice to 2003-01-31, the first month-end with T5YIE/T10YIE vintage data."""
    final = df.loc[df.index >= START_DATE].copy()

    missing = final.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("Columns with remaining missing values after slicing to START_DATE:")
        print(missing)
    else:
        print("No missing values in the final panel.")

    return final


def save_final_panel(final):
    final = final[FINAL_COLUMNS]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final.to_csv(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} with columns: {list(final.columns)}")
    return final


def run_adf_tests(final):
    """Step 6: ADF stationarity check on every final variable.

    This is diagnostic, not a second pass at picking representations --
    We deliberately keeps several variables (T10Y2Y, DGS3MO,
    T5YIE, T5YIE_MINUS_T10YIE) as levels even though they may read as
    non-stationary, because the level *is* the economically meaningful
    object (see apply_transformations' docstring). A rejected null here
    just documents that regime-like persistence, it isn't a signal to
    re-difference those series.
    """
    rows = []
    for col in FINAL_COLUMNS:
        series = final[col].dropna()
        statistic, p_value, *_ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "variable": col,
                "adf_statistic": statistic,
                "p_value": p_value,
                "n_obs": len(series),
                "stationary_at_5pct": p_value < ADF_SIGNIFICANCE,
            }
        )

    results = pd.DataFrame(rows).set_index("variable")
    results.to_csv(ADF_OUTPUT_PATH)
    print(f"\nSaved {ADF_OUTPUT_PATH}")
    print("\n=== ADF stationarity test (H0: unit root / non-stationary) ===")
    print(results)
    return results


def main():
    panel = load_panel()
    df = drop_redundant_channels(panel)
    df = engineer_inflation_curve_spread(df)
    df = apply_transformations(df)
    final = slice_to_start_date(df)
    final = save_final_panel(final)
    print(final.describe())
    run_adf_tests(final)


if __name__ == "__main__":
    main()

"""Historical-volatility benchmark: the random-walk forecast of monthly realised variance.

The forecast made at the end of month t for month t+1 is month t's realised variance,
RV_hat(t+1) = RV(t). It has no estimated parameters. In a volatility-managed portfolio it gives the
weight sigma2_bar / RV(t), the specification of Moreira & Muir (2017).

Writes forecasts/hist_rw_forecasts_tidy.csv in the same format as the other tidy forecast files:
a row dated t (forecast origin) holds the forecast for target_date t+1. RV is `rv_1m` from
dataset/price/clean/realized_variance.csv, the shared panel the OLS and Elastic Net models use.

Run from anywhere: python models/forecasting/build_hist_rw.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RV_PATH = PROJECT_ROOT / "dataset" / "price" / "clean" / "realized_variance.csv"
OUT_PATH = Path(__file__).resolve().parent / "forecasts" / "hist_rw_forecasts.csv"

MODEL = "hist_rw"
# Same forecast origins as the other forecast files: 2008-02 (for holding month 2008-03) onwards
FIRST_ORIGIN = pd.Timestamp("2008-02-29")


def build_hist_rw(rv_path=RV_PATH):
    rv = pd.read_csv(rv_path, parse_dates=["date"])
    frames = []
    for etf, g in rv.sort_values("date").groupby("etf"):
        s = g.set_index("date")["rv_1m"]
        periods = pd.PeriodIndex(s.index, freq="M")
        assert np.all(np.diff(periods.asi8) == 1), f"{etf}: months are not consecutive"
        assert s.notna().all() and (s > 0).all(), f"{etf}: missing or non-positive RV"

        frames.append(pd.DataFrame({
            "model": MODEL,
            "etf": etf,
            "date": s.index[:-1],            # forecast origin t
            "target_date": s.index[1:],      # month being forecast, t+1
            "actual_logRV": np.log(s.to_numpy()[1:]),
            "actual_RV": s.to_numpy()[1:],
            "fc_log": np.log(s.to_numpy()[:-1]),
            "sigma2": np.nan,                # level forecast: no log-normal correction
            "fc_level": s.to_numpy()[:-1],
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[out["date"] >= FIRST_ORIGIN].reset_index(drop=True)


if __name__ == "__main__":
    out = build_hist_rw()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    span = out.groupby("etf")["date"].agg(["min", "max", "count"])
    print(span)
    print(f"Saved {len(out)} rows to {OUT_PATH}")

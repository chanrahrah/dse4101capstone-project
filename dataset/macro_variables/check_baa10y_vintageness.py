"""
Sanity check: does ALFRED vintage BAA - GS10 match FRED's BAA10Y series?

BAA10Y ("Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on
10-Year Treasury Constant Maturity") is DAILY, while FRED's BAA and GS10
series are MONTHLY -- so BAA10Y is not literally BAA - GS10 on matching
frequency. This script pulls the ALFRED first-release (as-originally-
published) vintage of monthly BAA and GS10, computes BAA - GS10, and
compares it against BAA10Y resampled to a monthly mean, to see how close
the monthly credit spread implied by BAA/GS10 is to BAA10Y.
"""

import os

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

START = "2000-01-01"
END = "2026-07-31"


def load_dotenv_fallback(path=None):
    """Minimal .env loader so FRED_API_KEY doesn't need to be exported manually."""
    if path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(repo_root, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_fred_api_key():
    load_dotenv_fallback()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing FRED_API_KEY. Set it in your environment (.env) before running this script."
        )
    return api_key


def fetch_alfred_series(series_id, api_key):
    """Fetch the ALFRED first-release (as-originally-published) vintage of a series.

    output_type=4 returns, for each observation date, only its initial
    release value -- the standard "vintage" read for a series.
    """
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "output_type": 4,
        "realtime_start": START,
        "realtime_end": END,
        "observation_start": START,
        "observation_end": END,
        "sort_order": "asc",
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "error_message" in payload:
        raise RuntimeError(f"FRED API error for {series_id}: {payload['error_message']}")

    df = pd.DataFrame(payload["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df.set_index("date")["value"].sort_index()


def fetch_fred_series(series_id, api_key):
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START,
        "observation_end": END,
        "sort_order": "asc",
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "error_message" in payload:
        raise RuntimeError(f"FRED API error for {series_id}: {payload['error_message']}")

    df = pd.DataFrame(payload["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df.set_index("date")["value"].sort_index()


def main():
    api_key = get_fred_api_key()

    print("Fetching ALFRED vintage BAA ...")
    baa = fetch_alfred_series("BAA", api_key)

    print("Fetching ALFRED vintage GS10 ...")
    gs10 = fetch_alfred_series("GS10", api_key)

    print("Fetching FRED BAA10Y (daily) ...")
    baa10y_daily = fetch_fred_series("BAA10Y", api_key)

    computed = (baa - gs10).dropna()
    computed.index = computed.index.to_period("M").to_timestamp("M")

    published = baa10y_daily.dropna().resample("ME").mean()
    published.index.name = "date"

    compare = pd.DataFrame(
        {"computed_BAA_minus_GS10": computed, "published_BAA10Y_monthly_mean": published}
    ).dropna()
    compare["diff"] = (
        compare["computed_BAA_minus_GS10"] - compare["published_BAA10Y_monthly_mean"]
    )

    print(f"\nCompared {len(compare)} overlapping months")
    print(f"Max abs diff:  {compare['diff'].abs().max():.4f}")
    print(f"Mean abs diff: {compare['diff'].abs().mean():.4f}")
    print(f"Correlation:   {compare['computed_BAA_minus_GS10'].corr(compare['published_BAA10Y_monthly_mean']):.6f}")

    TOLERANCE = 0.02  # pp; monthly BAA/GS10 vs monthly-mean-of-daily BAA10Y are not identical constructs
    close = compare["diff"].abs() < TOLERANCE
    print(f"Matching (|diff| < {TOLERANCE}pp): {close.sum()} / {len(compare)}")

    if not close.all():
        print("\nLargest mismatches:")
        print(compare.reindex(compare["diff"].abs().sort_values(ascending=False).index).head(25))

    if close.all():
        print(
            "\nRESULT: monthly BAA - GS10 tracks BAA10Y closely -- effectively the same spread."
        )
    else:
        print(
            "\nRESULT: BAA10Y is NOT identical to monthly BAA - GS10 -- see mismatches above "
            "(BAA10Y uses daily DBAA/DGS10, not the monthly BAA/GS10 series)."
        )


if __name__ == "__main__":
    main()

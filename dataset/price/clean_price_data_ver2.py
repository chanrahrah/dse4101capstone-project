"""
Clean CRSP daily sector ETF prices + risk-free rate into a daily panel of
raw and excess returns. Produces a data-quality report only -- no realized
variance, no macro merge, no portfolio weights (handled downstream).

XLRE handling: XLRE (inception 2015-10-07) is used ONLY to rebuild a
consistent XLF series across the Sept 2016 GICS reclassification. From the
ex-date (2016-09-19) onward, ret_XLF is the daily return of a buy-and-hold
bundle of 1 XLF share + 0.139146 XLRE shares (value-weighted using prior-close
position values). XLRE is then dropped from the output panel.

Inputs:
    raw/sector_etf_prices_with_xlre.csv       CRSP daily export (via WRDS)
    raw/risk_free_rate_returns.csv  daily risk-free rate series

Output:
    clean/daily_returns_panel.csv
"""

import os

import pandas as pd

RAW_DIR = "dataset/price/raw"
CLEAN_DIR = "dataset/price/clean"

PRICE_FILE = f"{RAW_DIR}/sector_etf_prices_with_xlre.csv"
RF_FILE = f"{RAW_DIR}/risk_free_rate_returns.csv"
OUT_FILE = f"{CLEAN_DIR}/daily_returns_panel.csv"

EXPECTED_TICKERS = ["XLI", "XLB", "XLE", "XLY", "XLF", "XLRE", "XLU", "XLP", "XLV"]

# Output panel is truncated to start here
SAMPLE_START_DATE = "2003-01-31"

CRSP_SENTINELS = [-66, -77, -88, -99]
IMPLAUSIBLE_ABS_RET_THRESHOLD = 0.5

# --- XLF / XLRE spin-off settings ---
XLRE_EX_DATE = "2016-09-19"        # ex-date of the XLRE distribution by XLF
XLRE_PER_XLF_SHARE = 0.139146      # XLRE shares distributed per XLF share
# Candidate CRSP price columns (first one found is used). Needed for the
# prior-close position values; negative CRSP prices (bid/ask midpoints) are
# converted with abs().
PRICE_COL_CANDIDATES = ["DlyPrc", "DlyClose", "PRC"]

# Assumption: risk_free_rate_returns.csv is already a daily decimal return
# (values like 0.0002), consistent with the price panel's return convention.
# No annualized -> daily conversion is applied.


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# PRICE DATA CLEANING
# ---------------------------------------------------------------------------
section("LOADING PRICE DATA")

price_raw = pd.read_csv(PRICE_FILE)
print(f"Loaded {len(price_raw):,} rows from {PRICE_FILE}")
print(f"Columns available: {list(price_raw.columns)}")

price_col = next((c for c in PRICE_COL_CANDIDATES if c in price_raw.columns), None)
if price_col is None:
    print(f"WARNING: none of {PRICE_COL_CANDIDATES} found in the file. The XLF/XLRE merge "
          "needs a price column for prior-close position values -- re-pull from WRDS "
          "with the price field included.")
else:
    print(f"Using price column '{price_col}' for XLF/XLRE position values.")

wanted_cols = ["PERMNO", "Ticker", "DlyCalDt", "DlyRet", "DlyVol"]
if price_col is not None:
    wanted_cols.append(price_col)
missing_cols = [c for c in wanted_cols if c not in price_raw.columns]
if missing_cols:
    print(f"NOTE: expected columns not present in file, skipping: {missing_cols}")
subset_cols = [c for c in wanted_cols if c in price_raw.columns]
price = price_raw[subset_cols].copy()
price["DlyCalDt"] = pd.to_datetime(price["DlyCalDt"])

tickers_present = sorted(price["Ticker"].unique())
missing_tickers = sorted(set(EXPECTED_TICKERS) - set(tickers_present))
extra_tickers = sorted(set(tickers_present) - set(EXPECTED_TICKERS))
print(f"\nTickers present in data ({len(tickers_present)}): {tickers_present}")
if missing_tickers:
    print(f"WARNING: expected tickers NOT found in the file at all: {missing_tickers}")
if extra_tickers:
    print(f"WARNING: unexpected tickers found: {extra_tickers}")

# --- PERMNO <-> Ticker stability check ---
section("PERMNO <-> TICKER STABILITY CHECK")

permnos_per_ticker = price.groupby("Ticker")["PERMNO"].nunique()
tickers_with_multiple_permnos = permnos_per_ticker[permnos_per_ticker > 1]
if len(tickers_with_multiple_permnos) > 0:
    print("FLAG: tickers mapping to more than one PERMNO:")
    for tkr in tickers_with_multiple_permnos.index:
        sub = price[price["Ticker"] == tkr]
        for permno, grp in sub.groupby("PERMNO"):
            print(f"  {tkr} -> PERMNO {permno}: {grp['DlyCalDt'].min().date()} to {grp['DlyCalDt'].max().date()} ({len(grp)} rows)")
else:
    print("OK: every ticker maps to exactly one PERMNO.")

tickers_per_permno = price.groupby("PERMNO")["Ticker"].nunique()
permnos_with_multiple_tickers = tickers_per_permno[tickers_per_permno > 1]
if len(permnos_with_multiple_tickers) > 0:
    print("FLAG: PERMNOs mapping to more than one ticker:")
    for permno in permnos_with_multiple_tickers.index:
        sub = price[price["PERMNO"] == permno]
        for tkr, grp in sub.groupby("Ticker"):
            print(f"  PERMNO {permno} -> {tkr}: {grp['DlyCalDt'].min().date()} to {grp['DlyCalDt'].max().date()} ({len(grp)} rows)")
else:
    print("OK: every PERMNO maps to exactly one ticker.")

# --- Coverage per ticker ---
section("COVERAGE PER TICKER (first/last available date)")

coverage = price.groupby("Ticker")["DlyCalDt"].agg(first_date="min", last_date="max", n_obs="count")
coverage = coverage.reindex(tickers_present)
print(coverage.to_string())
print("\nNo dates are dropped or truncated automatically -- XLRE's later inception")
print("(2015-10-07) is expected and handled in the XLF/XLRE merge below.")

# --- Missing values / sentinel codes / implausible returns ---
section("MISSING VALUES, CRSP SENTINEL CODES, AND IMPLAUSIBLE RETURNS")

n_missing_ret = price["DlyRet"].isna().sum()
print(f"Rows with missing DlyRet: {n_missing_ret}")
if n_missing_ret > 0:
    print(price.loc[price["DlyRet"].isna(), "Ticker"].value_counts().to_string())

print("\nCRSP sentinel code counts per ticker (DlyRet in {-66, -77, -88, -99}):")
sentinel_mask = price["DlyRet"].isin(CRSP_SENTINELS)
if sentinel_mask.any():
    print(price.loc[sentinel_mask].groupby("Ticker")["DlyRet"].value_counts().to_string())
else:
    print("None found.")

print(f"\nImplausible DlyRet values (== -1, or |DlyRet| > {IMPLAUSIBLE_ABS_RET_THRESHOLD}):")
implausible_mask = (price["DlyRet"] == -1) | (price["DlyRet"].abs() > IMPLAUSIBLE_ABS_RET_THRESHOLD)
implausible = price.loc[implausible_mask].sort_values("DlyCalDt")
if len(implausible) > 0:
    print(implausible.to_string(index=False))
else:
    print("None found.")

# --- Pivot to wide daily panel ---
section("PIVOT TO WIDE DAILY PANEL (outer join across tickers)")

ret_wide = price.pivot_table(index="DlyCalDt", columns="Ticker", values="DlyRet", aggfunc="first")
vol_wide = price.pivot_table(index="DlyCalDt", columns="Ticker", values="DlyVol", aggfunc="first")
ret_wide = ret_wide.sort_index()
vol_wide = vol_wide.sort_index()

# Wide price panel (used only for XLF/XLRE position values)
px_wide = None
if price_col is not None:
    px_wide = price.pivot_table(index="DlyCalDt", columns="Ticker", values=price_col, aggfunc="first")
    px_wide = px_wide.sort_index().abs()

print(f"Wide panel shape (outer join on trading dates): {ret_wide.shape[0]} dates x {ret_wide.shape[1]} tickers")
print("\nMissing-date counts per ticker after outer join (dates where OTHER tickers")
print("traded but this ticker has no return -- expected for ETFs with later")
print("inception, e.g. XLRE):")
print(ret_wide.isna().sum().to_string())

print("\nPairwise mismatched-date counts (dates where exactly one of the pair is")
print("missing a return -- large counts typically reflect different inception dates):")
pair_report = []
for i, t1 in enumerate(ret_wide.columns):
    for t2 in ret_wide.columns[i + 1:]:
        mismatch = ((ret_wide[t1].isna()) != (ret_wide[t2].isna())).sum()
        pair_report.append((t1, t2, mismatch))
pair_df = pd.DataFrame(pair_report, columns=["ticker_1", "ticker_2", "n_mismatched_dates"])
pair_df = pair_df.sort_values("n_mismatched_dates", ascending=False)
print(pair_df.to_string(index=False))

print("\nNo forward-fill or row-dropping has been applied -- the wide panel above")
print("keeps NaN for any ticker/date combination with no observed return.")

# --- Truncate to the requested sample start date ---
section(f"TRUNCATING TO SAMPLE_START_DATE = {SAMPLE_START_DATE}")

n_before = len(ret_wide)
ret_wide = ret_wide.loc[SAMPLE_START_DATE:]
vol_wide = vol_wide.loc[SAMPLE_START_DATE:]
n_after = len(ret_wide)
print(f"Dropped {n_before - n_after} dates before {SAMPLE_START_DATE} ({n_before} -> {n_after} rows).")

# ---------------------------------------------------------------------------
# XLF / XLRE MERGE (GICS reclassification, Sept 2016)
# ---------------------------------------------------------------------------
section(f"MERGING XLRE INTO XLF FROM EX-DATE {XLRE_EX_DATE}")

for needed in ("XLF", "XLRE"):
    if needed not in ret_wide.columns:
        raise ValueError(f"{needed} not found in the return panel; cannot build merged XLF series.")
if px_wide is None:
    raise ValueError(
        f"No price column found (tried {PRICE_COL_CANDIDATES}); prior-close position "
        "values cannot be computed. Re-pull the WRDS data with a price field."
    )

ex_date = pd.Timestamp(XLRE_EX_DATE)
if ex_date not in ret_wide.index:
    next_dates = ret_wide.index[ret_wide.index >= ex_date]
    print(f"NOTE: {ex_date.date()} is not a trading date in the panel; "
          f"using first available date on/after it: {next_dates[0].date()}")
    ex_date = next_dates[0]
post_mask = ret_wide.index >= ex_date

# Prior-close prices, aligned to the return panel's dates. Computed on the full
# price panel before reindexing so the first post-ex-date day sees the true prior close.
px_prev = px_wide.shift(1).reindex(ret_wide.index)

value_xlf_prev = px_prev["XLF"]                                # 1 share of XLF
value_xlre_prev = XLRE_PER_XLF_SHARE * px_prev["XLRE"]         # 0.139146 shares of XLRE
w_xlf_prev = value_xlf_prev / (value_xlf_prev + value_xlre_prev)
w_xlre_prev = 1.0 - w_xlf_prev

ret_combined = w_xlf_prev * ret_wide["XLF"] + w_xlre_prev * ret_wide["XLRE"]

xlf_original = ret_wide["XLF"].copy()
xlf_merged = xlf_original.copy()
xlf_merged[post_mask] = ret_combined[post_mask]

# Diagnostics: window around the ex-date
ex_pos = ret_wide.index.get_loc(ex_date)
lo, hi = max(ex_pos - 3, 0), min(ex_pos + 4, len(ret_wide))
diag = pd.DataFrame({
    "ret_XLF_raw": xlf_original,
    "ret_XLRE": ret_wide["XLRE"],
    "w_XLF_prev": w_xlf_prev,
    "w_XLRE_prev": w_xlre_prev,
    "ret_XLF_merged": xlf_merged,
}).iloc[lo:hi]
print("Window around the ex-date (raw XLF vs. merged bundle):")
print(diag.to_string())
print("\nCHECK: if ret_XLF_raw on the ex-date shows a large drop (roughly the value of the")
print("distributed XLRE shares as a % of XLF price), CRSP is NOT counting the distribution")
print("in DlyRet and the merged series fixes it. If ret_XLF_raw already looks normal on")
print("the ex-date, CRSP has already included the distribution in DlyRet -- applying the")
print("merge would then double count, so inspect before using.")

n_post = int(post_mask.sum())
n_post_nan = int(xlf_merged[post_mask].isna().sum())
print(f"\nPost-ex-date rows: {n_post}; merged XLF return missing on {n_post_nan} of them.")
if n_post_nan > 0:
    print("Dates with missing merged return (first 20 shown):")
    print(list(xlf_merged[post_mask][xlf_merged[post_mask].isna()].index[:20].strftime("%Y-%m-%d")))

print(f"\nPre-ex-date rows ({int((~post_mask).sum())}) keep the raw XLF return unchanged.")

# Overwrite XLF, then drop XLRE as a stand-alone column
ret_wide["XLF"] = xlf_merged
ret_wide = ret_wide.drop(columns="XLRE")
vol_wide = vol_wide.drop(columns="XLRE", errors="ignore")
output_tickers = list(ret_wide.columns)
print(f"\nXLRE dropped from the modelling panel. Remaining tickers: {output_tickers}")

# ---------------------------------------------------------------------------
# RISK-FREE RATE MERGE
# ---------------------------------------------------------------------------
section("LOADING RISK-FREE RATE DATA")

rf_raw = pd.read_csv(RF_FILE)
print(f"Loaded {len(rf_raw):,} rows from {RF_FILE}")
print(f"Columns: {list(rf_raw.columns)}")

date_col = "date" if "date" in rf_raw.columns else rf_raw.columns[0]
rate_col = "rf" if "rf" in rf_raw.columns else rf_raw.columns[1]

rf = rf_raw[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "rf_raw"})
rf["date"] = pd.to_datetime(rf["date"])

print("\nSample rows:")
print(rf.head(10).to_string(index=False))
print("\nSummary statistics of rf_raw:")
print(rf["rf_raw"].describe().to_string())

rf["rf_daily"] = rf["rf_raw"]

# --- Merge onto price panel: left join on price panel's trading dates ---
section("MERGING RISK-FREE RATE ONTO PRICE PANEL")

price_dates = pd.DataFrame({"date": ret_wide.index})
rf_merged = price_dates.merge(rf[["date", "rf_daily"]], on="date", how="left")
rf_merged = rf_merged.set_index("date")

n_missing_rf = rf_merged["rf_daily"].isna().sum()
print(f"Trading dates in price panel: {len(rf_merged)}")
print(f"Trading dates with NO matching rf rate after merge: {n_missing_rf}")
if n_missing_rf > 0:
    missing_rf_dates = rf_merged.loc[rf_merged["rf_daily"].isna()].index
    print("Dates missing rf (first 20 shown):")
    print(list(missing_rf_dates[:20].strftime("%Y-%m-%d")))

# ---------------------------------------------------------------------------
# OUTPUT: raw + excess return panel
# ---------------------------------------------------------------------------
section("BUILDING OUTPUT PANEL")

rf_daily_series = rf_merged["rf_daily"]

# ret_wide now holds the merged XLF series and no XLRE column, so xret_XLF
# is also built from the merged series.
raw_ret = ret_wide.add_prefix("ret_")
excess_ret = ret_wide.sub(rf_daily_series, axis=0).add_prefix("xret_")

panel = pd.concat([raw_ret, excess_ret], axis=1)
panel.insert(0, "rf_daily", rf_daily_series)
panel = panel.sort_index()
panel.index.name = "date"

os.makedirs(CLEAN_DIR, exist_ok=True)
panel.to_csv(OUT_FILE)

section("FINAL SUMMARY")
print(f"Output written to: {OUT_FILE}")
print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")
print(f"Number of tickers: {len(output_tickers)} -> {output_tickers}")
print(f"XLF is the merged XLF+XLRE series from {ex_date.date()} onward; XLRE not in output.")
print(f"Number of rows: {len(panel)}")
print("\nRemaining missingness by column:")
print(panel.isna().sum().to_string())

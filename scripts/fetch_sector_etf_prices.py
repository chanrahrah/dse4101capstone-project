import os
import yfinance as yf

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "sector_etf_prices.csv")

# 9 sector ETFs
tickers = ["XLI", "XLB", "XLE", "XLY", "XLF", "XLRE", "XLU", "XLP", "XLV"]

# pull adjusted close, daily, fixed date range for reproducibility
data = yf.download(tickers, start="2015-10-01", end="2026-07-31", auto_adjust=True)[["Close", "Volume"]]

# flatten (field, ticker) columns into e.g. "XLB_close", "XLB_volume"
data.columns = [f"{ticker}_{field.lower()}" for field, ticker in data.columns]

# turn the DatetimeIndex into a plain yyyy-mm-dd Date column
data = data.reset_index()
data["Date"] = data["Date"].dt.date

# save to csv
data.to_csv(OUTPUT_PATH, index=False)

print(data.head())
print(data.tail())
print(f"\nShape: {data.shape[0]} days x {data.shape[1]} tickers")

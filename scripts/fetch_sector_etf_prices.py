import os
import yfinance as yf

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "sector_etf_prices.xlsx")

# 9 sector ETFs
tickers = ["XLI", "XLB", "XLE", "XLY", "XLF", "XLRE", "XLU", "XLP", "XLV"]

# pull adjusted close, daily, fixed date range for reproducibility
data = yf.download(tickers, start="2015-10-01", end="2026-07-31", auto_adjust=True)["Close"]

# save to Excel
data.to_excel(OUTPUT_PATH)

print(data.head())
print(data.tail())
print(f"\nShape: {data.shape[0]} days x {data.shape[1]} tickers")

import os #to talk to the operating system, for file path access
import datetime as dt 
import pandas as pd
import pandas_datareader.data as fred #fetch financial and economic data from FRED without needing to know the URL format

START = dt.datetime(2000, 1, 1)
END = dt.datetime.today()
OUTPUT_DIR = "dataset"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")

test = fred.DataReader("VIXCLS", "fred", START, END)
print(test.tail())

CANDIDATE_SERIES = {
    "T10Y2Y": ("rates_yield_curve", "daily", "10Y-2Y Treasury spread", 0),
    "DGS3MO": ("rates_yield_curve", "daily", "3-Month Treasury rate", 0),
    "BAMLH0A0HYM2": ("credit_spreads", "daily", "ICE BofA US High Yield OAS", 0),
    "BAA10Y": ("credit_spreads", "daily", "Baa corporate yield minus 10Y Treasury", 0),
    "CPIAUCSL": ("inflation", "monthly", "CPI, all urban consumers", 1),
    "PCEPI": ("inflation", "monthly", "PCE price index", 1),
    "T5YIE": ("inflation", "daily", "5-Year breakeven inflation", 0),
    "T10YIE": ("inflation", "daily", "10-Year breakeven inflation", 0),
    "M2SL": ("inflation", "monthly", "M2 money stock", 1),
    "VIXCLS": ("volatility_stress", "daily", "CBOE VIX", 0),
    "NFCI": ("volatility_stress", "weekly", "Chicago Fed Financial Conditions Index", 0),
    "STLFSI4": ("volatility_stress", "weekly", "St. Louis Fed Financial Stress Index", 0),
    "INDPRO": ("real_activity_labour", "monthly", "Industrial Production Index", 1),
    "CFNAI": ("real_activity_labour", "monthly", "Chicago Fed National Activity Index", 1),
    "UNRATE": ("real_activity_labour", "monthly", "Unemployment rate", 1),
    "ICSA": ("real_activity_labour", "weekly", "Initial jobless claims", 0),
    "UMCSENT": ("real_activity_labour", "monthly", "U. Michigan Consumer Sentiment", 0),
    "DTWEXBGS": ("dollar_commodity", "daily", "Trade Weighted US Dollar Index", 0),
    "DCOILWTICO": ("dollar_commodity", "daily", "WTI Crude Oil Price", 0),
}
 # dictionary used. each line is "FRED_ID": (channel, frequency, description, lag)
 # based on verified BLS/BEA/Fed release schedules. 0 = same-day/real-time or released
# within the reference month itself. 1 = released sometime during the following month.


def fetch_series(series_id):
    df = fred.DataReader(series_id, "fred", START, END)
    s = df[series_id]
    s.index.name = "date"
    return s

# eg. fetch_series("VIXCLS"). fred data reader - actual call. 

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

channel_data = {}
all_monthly_data = {}
for series_id, (channel, freq, desc, lag_months) in CANDIDATE_SERIES.items():
    print(f"Fetching {series_id} ({channel}, {freq}, lag={lag_months}mo) ...")
    s = fetch_series(series_id)
    s.to_csv(os.path.join(RAW_DIR, f"{series_id}.csv"))
    monthly = s.resample("ME").mean().shift(lag_months)
    channel_data.setdefault(channel, {})[series_id] = monthly
    all_monthly_data[series_id] = monthly # put all the 6 channels tgt for easy viewing

for channel, series_dict in channel_data.items():
    panel = pd.DataFrame(series_dict)
    panel.index.name = "date"
    panel = panel.sort_index()
    panel.to_csv(os.path.join(CLEAN_DIR, f"{channel}.csv"))
    print(f"Saved {channel}.csv with columns: {list(series_dict.keys())}")


full_panel = pd.DataFrame(all_monthly_data)
full_panel.index.name = "date"
full_panel = full_panel.sort_index()
full_panel.to_csv(os.path.join(CLEAN_DIR, "fred_monthly_panel.csv"))
print(f"Saved fred_monthly_panel.csv with all {len(all_monthly_data)} series")
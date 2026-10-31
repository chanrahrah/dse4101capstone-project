# Capstone Project

## Structure

- `dataset/` — raw and processed data files
- `models/` — trained model artifacts and checkpoints
- `scripts/` — standalone scripts for data collection, preprocessing, and other one-off tasks

## Getting Started

Clone the repo and run:

```
pip install -r requirements.txt
```

## Refresh Macro Data

Set `FRED_API_KEY` in your environment or a repository-level `.env` file,
then run these commands from the repository root:

```bash
python3 dataset/macro_variables/fetch_fred_data.py
python3 dataset/macro_variables/transform_macro_data.py
```

The fetch script writes API data to `dataset/macro_variables/raw/` and the
monthly panel to `dataset/macro_variables/clean/fred_monthly_panel.csv`.
The transform script uses that panel for all 11 predictors and writes
`final_macro_data.csv` and `adf_test_results.csv` to
`dataset/macro_variables/final_clean/`.

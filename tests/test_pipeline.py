"""Fast checks of saved pipeline interfaces; no downloads or model fitting.

Run: python3 -m unittest discover -s tests -v
"""

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from research_config import BLOCKS, SELECTED, TICKERS, TRAIN_MONTHS, TRAIN_START
from utils.paths import FORECAST_DIR, FORECAST_RESULTS_DIR
from utils.portfolio_utils import (
    FORECAST_FILES, build_portfolio, forecast_wide, load_forecast_files,
    load_monthly_returns, load_portfolios, training_window,
)


class PipelineContracts(unittest.TestCase):
    def test_raw_macro_reconstruction_is_offline(self):
        spec = importlib.util.spec_from_file_location(
            "macro_fetch_audit", ROOT / "dataset/macro_variables/fetch_fred_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(module, "CLEAN_DIR", tmp), \
                patch.object(module, "get_fred_api_key", side_effect=AssertionError("API key requested")), \
                patch.object(module, "fetch_raw_series", side_effect=AssertionError("Download requested")), \
                patch.object(module.requests.sessions.Session, "request", side_effect=AssertionError("Network requested")), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main(from_raw=True)
            actual = pd.read_csv(Path(tmp) / "fred_monthly_panel.csv")
            expected = pd.read_csv(ROOT / "dataset/macro_variables/clean/fred_monthly_panel.csv")
            pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {"fred_monthly_panel.csv", "dropped_series.csv"})

    def test_active_notebooks_have_valid_metadata_and_python(self):
        for path in [*ROOT.joinpath("models").rglob("*.ipynb"),
                     *ROOT.joinpath("dataset").rglob("*.ipynb")]:
            if "archive" in path.parts:
                continue
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                nb = json.loads(path.read_text())
                ids = [c.get("id") for c in nb["cells"]]
                if nb.get("nbformat_minor", 0) >= 5:
                    self.assertTrue(all(ids))
                    self.assertEqual(len(ids), len(set(ids)))
                nbformat.validate(nb)
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))

    def test_variance_and_forecast_targets_match_price_data_exactly(self):
        _, rv = load_monthly_returns(TICKERS)
        source = pd.read_csv(ROOT / "dataset/price/clean/realized_variance.csv", parse_dates=["date"])
        truth = source.pivot(index="date", columns="etf", values="rv_1m")[TICKERS]
        np.testing.assert_allclose(rv.loc[truth.index, TICKERS], truth, rtol=1e-10, atol=1e-12)
        raw, available = load_forecast_files(SELECTED)
        _, start = training_window(TRAIN_START, TRAIN_MONTHS)
        expected_months = rv.index[rv.index >= start]
        for family, variants in SELECTED.items():
            self.assertEqual(available[family], variants)
            for variant in variants:
                with self.subTest(model=variant):
                    actual = forecast_wide(raw[family], variant, "actual_RV", TICKERS)
                    np.testing.assert_allclose(actual, rv.loc[actual.index, TICKERS], rtol=1e-10, atol=1e-12)
                    self.assertTrue(expected_months.isin(actual.index).all())

    def test_forecast_loader_rejects_corrupt_keys_values_and_dates(self):
        original = pd.read_csv(FORECAST_DIR / FORECAST_FILES["Historical"])
        cases = {}
        cases["duplicate"] = pd.concat([original, original.iloc[[0]]], ignore_index=True)
        cases["negative"] = original.copy()
        cases["negative"].loc[0, "fc_level"] = -1
        cases["target"] = original.copy()
        cases["target"].loc[0, "target_date"] = "2099-01-31"
        cases["origin"] = original.copy()
        cases["origin"].loc[0, "date"] = "2008-02-15"
        cases["missing column"] = original.drop(columns="fc_log")
        for name, frame in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                frame.to_csv(Path(tmp) / "input.csv", index=False)
                with self.assertRaises(ValueError):
                    load_forecast_files({"Historical": ["hist_rw"]},
                                        {"Historical": "input.csv"}, Path(tmp))

    def test_all_forecast_comparisons_match_shared_blocks(self):
        expected_files = {f"e{i}_{suffix}.csv" for i in range(1, 7)
                          for suffix in ("losses", "evaluation")}
        self.assertEqual({p.name for p in FORECAST_RESULTS_DIR.iterdir() if p.is_file()}, expected_files)
        for i in range(1, 6):
            with self.subTest(evaluation=i):
                frame = pd.read_csv(FORECAST_RESULTS_DIR / f"e{i}_evaluation.csv")
                expected = {(pair, a, b) for pair, a, b, _ in BLOCKS[f"B{i}"]["pairs"]}
                self.assertEqual(set(frame[["pair", "a", "b"]].itertuples(index=False, name=None)), expected)
                self.assertFalse(frame.duplicated(["pair", "ETF", "loss", "test"]).any())
                self.assertTrue(frame.p_value.between(0, 1).all())
                self.assertEqual(set(frame.ETF), {*TICKERS, "pooled"})

    def test_e6_covers_portfolio_models_and_loss_ratios(self):
        frame = pd.read_csv(FORECAST_RESULTS_DIR / "e6_evaluation.csv")
        losses = pd.read_csv(FORECAST_RESULTS_DIR / "e6_losses.csv").set_index(["model", "ETF"])
        selected = {v for variants in SELECTED.values() for v in variants}
        self.assertEqual(set(frame.a), selected - {"hist_rw"})
        self.assertEqual(set(frame.b), {"hist_rw"})
        self.assertFalse(frame.duplicated(["a", "b", "ETF", "loss", "test"]).any())
        expected = [losses.loc[(r.a, r.ETF), r.loss] / losses.loc[(r.b, r.ETF), r.loss]
                    for r in frame.itertuples()]
        np.testing.assert_allclose(frame.loss_ratio, expected)

    def test_saved_portfolios_match_the_corrected_variance_definition(self):
        _, rv = load_monthly_returns(TICKERS)
        train_idx, _ = training_window(TRAIN_START, TRAIN_MONTHS)
        raw, _ = load_forecast_files(SELECTED)
        portfolios, xret = load_portfolios("fixed")
        sigma_hat = forecast_wide(raw["Historical"], "hist_rw", "fc_level", TICKERS).loc[xret.index]
        rebuilt = build_portfolio(sigma_hat, xret, rv.loc[train_idx].mean(), 2)
        np.testing.assert_allclose(portfolios["hist_rw"]["managed"], rebuilt["managed"])
        np.testing.assert_allclose(portfolios["hist_rw"]["weights_base"], rebuilt["weights_base"])

    def test_portfolio_link_preserves_selected_e6_statistics(self):
        frame = pd.read_csv(FORECAST_RESULTS_DIR / "e6_evaluation.csv")
        frame["ETF"] = frame["ETF"].replace({"pooled": "Cyclical"})
        link = pd.read_csv(ROOT / "models/portfolio/results/eval/forecast_portfolio_link.csv")
        self.assertFalse(link.duplicated(["model", "ETF"]).any())
        for loss in ["MSE_log", "QLIKE"]:
            for row in link.to_dict(orient="records"):
                tests = frame[frame.a.eq(row["model"]) & frame.ETF.eq(row["ETF"]) & frame.loss.eq(loss)]
                primary = "Clark-West" if "Clark-West" in set(tests.test) else "DM"
                selected = tests[tests.test.eq(primary)].iloc[0]
                self.assertEqual(row[f"{loss} test"], primary)
                for suffix, column in [("ratio", "loss_ratio"), ("t", "gain_t"), ("p", "p_value")]:
                    self.assertAlmostEqual(row[f"{loss} {suffix}"], selected[column], places=12)

    def test_portfolio_rejects_misaligned_or_negative_forecasts(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        returns = pd.DataFrame({"XLB": [.01, -.02, .03]}, index=idx)
        forecast = pd.DataFrame({"XLB": [.01, .02, .01]}, index=idx)
        with self.assertRaises(ValueError):
            build_portfolio(forecast.iloc[::-1], returns, pd.Series({"XLB": .01}), 2)
        forecast.iloc[0, 0] = -1
        with self.assertRaises(ValueError):
            build_portfolio(forecast, returns, pd.Series({"XLB": .01}), 2)


if __name__ == "__main__":
    unittest.main()

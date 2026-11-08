"""Reproduce data preparation, EDA, forecasts, evaluations, portfolios and robustness checks."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DATA_STEPS = (
    "dataset/macro_variables/fetch_fred_data.py",
    "dataset/macro_variables/transform_macro_data.py",
    "dataset/price/clean_price_data_ver2.py",
    "dataset/price/calculate_realized_variance.py",
    "dataset/price/clean/eda_realized_volatility.ipynb",
)
BUILD_STEPS = (
    "models/forecasting/build/b0_hist_rw.ipynb",
    "models/forecasting/build/b1_garch_garchmidas.ipynb",
    "models/forecasting/build/b2_elastic_net.ipynb",
    "models/forecasting/build/b3_xgboost.ipynb",
)
ANALYSIS_STEPS = (
    "models/forecasting/evaluation/e1_garch_family.ipynb",
    "models/forecasting/evaluation/e2_macro_value.ipynb",
    "models/forecasting/evaluation/e3_vix_effect_en.ipynb",
    "models/forecasting/evaluation/e4_vix_effect_xgb.ipynb",
    "models/forecasting/evaluation/e5_linear_vs_nonlinear.ipynb",
    "models/forecasting/evaluation/e6_naive_vs_all_models.ipynb",
    "models/forecasting/robustness/r1_crisis_en.ipynb",
    "models/forecasting/robustness/r2_crisis_xgb.ipynb",
    "models/forecasting/robustness/r2_crisis_naive_vs_all_models.ipynb",
    "models/forecasting/robustness/r3_rolling_window_en.ipynb",
    "models/forecasting/robustness/r4_rolling_window_xgb.ipynb",
    "models/portfolio/p1_build_portfolios.ipynb",
    "models/portfolio/p2_portfolio_eval.ipynb",
    "models/portfolio/p3_forecast_vs_portfolio.ipynb",
    "models/portfolio/robustness/rp1_different_leverages.ipynb",
    "models/portfolio/robustness/rp2_timing_vs_relevering.ipynb",
)
SAVED_FORECASTS = (
    "models/forecasting/forecasts/hist_rw_forecasts.csv",
    "models/forecasting/forecasts/garch_forecasts.csv",
    "models/forecasting/forecasts/en_forecasts.csv",
    "models/forecasting/forecasts/xgb_forecasts.csv",
    "models/forecasting/robustness/forecasts/en_rolling60_forecasts.csv",
    "models/forecasting/robustness/forecasts/xgb_rolling60_forecasts.csv",
)


def pipeline_steps(reuse_forecasts=False):
    steps = list(DATA_STEPS)
    if not reuse_forecasts:
        steps.extend(BUILD_STEPS)
    return steps + list(ANALYSIS_STEPS)


def preflight(start, steps, reuse_forecasts=False):
    """Check inputs and dependencies before any research output is overwritten."""
    required = list(steps) + [
        "dataset/price/raw/sector_etf_prices_with_xlre.csv",
        "dataset/price/raw/risk_free_rate_returns.csv",
    ]
    spec = importlib.util.spec_from_file_location(
        "fred_pipeline", ROOT / "dataset/macro_variables/fetch_fred_data.py")
    fred = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fred)
    if start == "raw":
        required.extend(f"dataset/macro_variables/raw/{series}.csv"
                        for series in fred.FINAL_PANEL_COLUMNS)
    if reuse_forecasts:
        required.extend(SAVED_FORECASTS)
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))
    if start == "fetch":
        fred.get_fred_api_key()
    # Import compiled dependencies too: installed metadata alone does not establish usability.
    for module in ("nbformat", "nbclient", "ipykernel", "numba", "xgboost", "statsmodels.api"):
        __import__(module)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("fetch", "raw"), required=True)
    parser.add_argument("--reuse-forecasts", action="store_true",
                        help="Skip b0-b3 and evaluate saved expanding/rolling forecasts; not a fresh model fit.")
    parser.add_argument("--no-fit-cache", action="store_true",
                        help="Disable GARCH fit-cache reads and writes for a fresh fit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered files without executing or writing.")
    parser.add_argument("--run-dir", type=Path,
                        help="New directory for executed notebooks and script logs (default: supporting/pipeline_runs/<UTC>).")
    args = parser.parse_args(argv)
    if args.reuse_forecasts and args.no_fit_cache:
        parser.error("--no-fit-cache has no effect with --reuse-forecasts")
    steps = pipeline_steps(args.reuse_forecasts)
    if args.dry_run:
        for index, filename in enumerate(steps, 1):
            print(f"[{index}/{len(steps)}] {filename}")
        return

    preflight(args.start, steps, args.reuse_forecasts)
    import nbformat
    from nbclient import NotebookClient

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.run_dir or ROOT / "supporting/pipeline_runs" / stamp
    run_dir = run_dir.resolve()
    # Keep executed notebooks from separate runs in separate directories.
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DSE4101_RECOMPUTE_FORECASTS="1", MPLBACKEND="Agg",
               DSE4101_USE_FIT_CACHE="0" if args.no_fit_cache else "1")
    print(f"Executed notebooks and logs: {run_dir}", flush=True)
    if args.reuse_forecasts:
        print("Using saved forecasts; b0-b3 will NOT be refitted.", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="dse4101-kernel-") as temporary:
            spec = Path(temporary) / "kernels/dse4101-pipeline"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Pipeline Python", "language": "python",
            }))
            # Kernel discovery uses the parent process environment, not the kernel's env argument.
            previous_jupyter_path = os.environ.get("JUPYTER_PATH")
            os.environ["JUPYTER_PATH"] = temporary + os.pathsep + (previous_jupyter_path or "")
            try:
                for index, filename in enumerate(steps, 1):
                    print(f"[{index}/{len(steps)}] {filename}", flush=True)
                    notebook = None
                    try:
                        path = ROOT / filename
                        if path.suffix == ".py":
                            command = [sys.executable, "-u", str(path)]
                            if filename == DATA_STEPS[0] and args.start == "raw":
                                command.append("--from-raw")
                            log = run_dir / "logs" / (Path(filename).stem + ".log")
                            log.parent.mkdir(parents=True, exist_ok=True)
                            with log.open("w") as stream:
                                subprocess.run(command, cwd=ROOT, env=env, check=True,
                                               stdout=stream, stderr=subprocess.STDOUT)
                        else:
                            notebook = nbformat.read(path, as_version=4)
                            NotebookClient(
                                notebook, kernel_name="dse4101-pipeline", timeout=None,
                                allow_errors=False, ipython_hist_file=":memory:",
                            ).execute(cwd=str(ROOT), env=env)
                    finally:
                        if notebook is not None:
                            executed = run_dir / "notebooks" / filename
                            executed.parent.mkdir(parents=True, exist_ok=True)
                            nbformat.write(notebook, executed)
            finally:
                if previous_jupyter_path is None:
                    os.environ.pop("JUPYTER_PATH", None)
                else:
                    os.environ["JUPYTER_PATH"] = previous_jupyter_path
    except Exception as exc:
        failed = filename if "filename" in locals() else "kernel setup"
        raise SystemExit(f"Failed: {failed}\n{exc}\nOutputs: {run_dir}") from exc
    print(f"Pipeline complete. Outputs: {run_dir}")


if __name__ == "__main__":
    main()

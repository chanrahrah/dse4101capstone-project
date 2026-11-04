"""Run data preparation, forecasting, evaluation and portfolios in order."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("fetch", "raw"), required=True)
    args = parser.parse_args()

    import nbformat
    from nbclient import NotebookClient
    # Check the expensive stages' dependencies before refreshing any output files.
    import numba
    import xgboost

    root = Path(__file__).resolve().parent
    files = [
        "dataset/macro_variables/fetch_fred_data.py",
        "dataset/macro_variables/transform_macro_data.py",
        "dataset/price/clean_price_data_ver2.py",
        "dataset/price/calculate_realized_variance.py",
        "models/forecasting/build/b0_hist_rw.ipynb",
        "models/forecasting/build/b1_garch_garchmidas.ipynb",
        "models/forecasting/build/b2_elastic_net.ipynb",
        "models/forecasting/build/b3_xgboost.ipynb",
        "models/forecasting/evaluation/e1_garch_family.ipynb",
        "models/forecasting/evaluation/e2_macro_value.ipynb",
        "models/forecasting/evaluation/e3_vix_effect_en.ipynb",
        "models/forecasting/evaluation/e4_vix_effect_xgb.ipynb",
        "models/forecasting/evaluation/e5_linear_vs_nonlinear.ipynb",
        "models/forecasting/evaluation/e6_forecast_summary.ipynb",
        "models/portfolio/p1_build_portfolios.ipynb",
        "models/portfolio/p2_portfolio_eval.ipynb",
        "models/portfolio/p3_forecast_vs_portfolio.ipynb",
    ]
    env = dict(os.environ, DSE4101_RECOMPUTE_FORECASTS="1", MPLBACKEND="Agg")

    # A temporary kernel spec guarantees the same Python as this runner,
    # without installing a kernel or relying on notebook metadata.
    with tempfile.TemporaryDirectory(prefix="dse4101-kernel-") as temporary:
        spec = Path(temporary) / "kernels" / "dse4101-pipeline"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Pipeline Python",
            "language": "python",
        }))
        os.environ["JUPYTER_PATH"] = temporary + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        for index, filename in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {filename}", flush=True)
            try:
                path = root / filename
                if path.suffix == ".py":
                    command = [sys.executable, "-u", str(path)]
                    if index == 1 and args.start == "raw":
                        command.append("--from-raw")
                    subprocess.run(command, cwd=root, env=env, check=True)
                else:
                    notebook = nbformat.read(path, as_version=4)
                    NotebookClient(
                        notebook, kernel_name="dse4101-pipeline", timeout=None,
                        allow_errors=False, ipython_hist_file=":memory:",
                    ).execute(cwd=str(root), env=env)
            except Exception as exc:
                raise SystemExit(f"Failed: {filename}\n{exc}") from exc
    print("Pipeline complete.")


if __name__ == "__main__":
    main()

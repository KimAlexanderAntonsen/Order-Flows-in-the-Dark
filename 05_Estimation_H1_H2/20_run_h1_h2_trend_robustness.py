"""Run the family-level trend-controlled H1/H2 robustness pass.

Re-fits the H1/H2 estimation with a regime-specific linear 
``day_index`` covariate in the exogenous block. 
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


H1H2_DIR = Path(__file__).resolve().parent
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

_helpers = importlib.import_module("19_h1_h2_trend_robustness")


N_DRAWS = 5000
FAMILY_NAMES = _helpers.FAMILY_NAMES
OUTPUT_DIR = _helpers.OUTPUT_DIR
ensure_output_dirs = _helpers.ensure_output_dirs
run_family_trend_robustness = _helpers.run_family_trend_robustness


def _save_family_outputs(family_result: dict[str, object]) -> None:
    family = str(family_result["family"])
    pre_result = family_result["pre_result"]
    post_result = family_result["post_result"]

    pre_result.coefficient_table.to_csv(
        OUTPUT_DIR / f"h1h2_trend_{family}_pre_coefficients.csv", index=False
    )
    post_result.coefficient_table.to_csv(
        OUTPUT_DIR / f"h1h2_trend_{family}_post_coefficients.csv", index=False
    )

    for shock_output in family_result["shock_outputs"]:
        shock_name = str(shock_output["shock_name"])
        prefix = f"h1h2_trend_{family}_{shock_name}"
        shock_output["pre_bands"].to_csv(OUTPUT_DIR / f"{prefix}_pre_bands.csv", index=False)
        shock_output["post_bands"].to_csv(OUTPUT_DIR / f"{prefix}_post_bands.csv", index=False)
        shock_output["difference_bands"].to_csv(
            OUTPUT_DIR / f"{prefix}_post_minus_pre_bands.csv", index=False
        )


def main() -> None:
    ensure_output_dirs()
    print("Running family-level trend-controlled H1/H2 robustness...", flush=True)

    summary_rows: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  trend robustness: estimating {family_name}...", flush=True)
        family_result = run_family_trend_robustness(family_name, n_draws=N_DRAWS)
        _save_family_outputs(family_result)
        summary_rows.append(
            {
                "family": family_name,
                "p_lags": int(family_result["p_lags"]),
                "n_draws": int(N_DRAWS),
                "trend_col": _helpers.TREND_COL_NAME,
            }
        )

    summary_path = OUTPUT_DIR / "h1h2_trend_run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nTrend-controlled robustness outputs saved.", flush=True)
    print(f"  Run summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

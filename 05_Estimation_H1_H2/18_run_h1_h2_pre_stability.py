"""Run the family-level within-pre IRF stability test.

Splits the pre window at 2019-08-07 and computes the within-pre drift 
IRF on the full constant-membership panel for each family, useing 
5,000 simulation draws.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


H1H2_DIR = Path(__file__).resolve().parent
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

_helpers = importlib.import_module("17_h1_h2_pre_stability")


N_DRAWS = 5000
FAMILY_NAMES = _helpers.FAMILY_NAMES
OUTPUT_DIR = _helpers.OUTPUT_DIR

run_family_pre_stability = _helpers.run_family_pre_stability
build_key_summary = _helpers.build_key_summary
build_run_summary = _helpers.build_run_summary
load_sp500_universe = _helpers.load_sp500_universe
ensure_output_dirs = _helpers.ensure_output_dirs


def _save_family(family_result: dict[str, object]) -> None:
    family_name = str(family_result["family"])
    for placebo_output in family_result["placebo_outputs"]:
        shock_name = str(placebo_output["shock_name"])
        prefix = f"h1h2_pre_stability_{family_name}_{shock_name}"
        placebo_output["pre_a_bands"].to_csv(OUTPUT_DIR / f"{prefix}_pre_a_bands.csv", index=False)
        placebo_output["pre_b_bands"].to_csv(OUTPUT_DIR / f"{prefix}_pre_b_bands.csv", index=False)
        placebo_output["drift_bands"].to_csv(OUTPUT_DIR / f"{prefix}_drift_bands.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    tickers = load_sp500_universe()
    print(f"Running family-level within-pre stability on {len(tickers)} stocks...", flush=True)

    results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  pre-stability: estimating {family_name}...", flush=True)
        family_result = run_family_pre_stability(
            family_name,
            tickers=tickers,
            n_draws=N_DRAWS,
        )
        _save_family(family_result)
        results.append(family_result)

    key_summary = build_key_summary(results)
    key_path = OUTPUT_DIR / "h1h2_pre_stability_key_summary.csv"
    key_summary.to_csv(key_path, index=False)

    run_summary = build_run_summary(results, n_draws=N_DRAWS)
    run_path = OUTPUT_DIR / "h1h2_pre_stability_run_summary.csv"
    run_summary.to_csv(run_path, index=False)

    print("\nWithin-pre stability outputs saved.", flush=True)
    print(f"  Key summary: {key_path}", flush=True)
    print(f"  Run summary: {run_path}", flush=True)


if __name__ == "__main__":
    main()

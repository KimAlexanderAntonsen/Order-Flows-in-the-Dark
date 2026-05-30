"""Run the trend-controlled H3 robustness pass.

This is the third H3 robustness block (after p=3 and 
least-retail-reference control). It reruns the benchmark H3 
specification with a group-specific linear ``day_index`` trend 
absorbed into the VARX as an exogenous covariate. 
"""

from __future__ import annotations

import importlib


_helpers = importlib.import_module("18_h3_trend_robustness")


ensure_output_dirs = _helpers.ensure_output_dirs
run_trend_robustness = _helpers.run_trend_robustness
save_trend_outputs = _helpers.save_trend_outputs


def main() -> None:
    """Run the trend-controlled H3 robustness and save the outputs."""

    ensure_output_dirs()

    print("Running trend-controlled H3 robustness...")
    results = run_trend_robustness()

    paths = save_trend_outputs(results)

    print("\nSaved trend-robustness outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

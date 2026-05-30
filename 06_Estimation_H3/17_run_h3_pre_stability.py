"""Run the within-pre IRF stability test for H3.

This is a placebo check that targets the actual H3 counterfactual. For 
each family it splits the pre period at 2019-08-07 and estimates the 
VARX separately on the first and second pre halves for the treated 
group and the matched control group. The within-pre drift for each 
group is

    drift_group = IRF(pre_B)_group - IRF(pre_A)_group

and the placebo difference in difference is

    placebo_drift = drift_treated - drift_control

If `placebo_drift` is close to zero at the H3 key horizons, the 
IRF-level parallel-trends assumption is credible. A non-trivial 
`placebo_drift` at a horizon that also moves in the benchmark H3 would 
argue that some of the benchmark reading is pre-existing drift rather 
than a regime response.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


_config = importlib.import_module("01_h3_config")
_helpers = importlib.import_module("16_h3_pre_stability")


FAMILY_NAMES = _config.FAMILY_NAMES

run_family_pre_stability = _helpers.run_family_pre_stability
build_key_summary = _helpers.build_key_summary
build_run_summary = _helpers.build_run_summary


PRE_STABILITY_OUTPUT_DIR: Path = _config.H3_DIR / "output" / "pre_stability"

# The within-pre stability test is a robustness object, so we match 
# the draw count used by the other robustness passes instead of the 
# 10,000-draw benchmark.
N_DRAWS = 5000


def _save_family_outputs(family_result: dict[str, object]) -> None:
    """Save the drift and placebo bands for one family."""

    family_name = str(family_result["family"])
    for placebo_output in family_result["placebo_outputs"]:
        shock_name = str(placebo_output["shock_name"])
        placebo_output["treated_drift_bands"].to_csv(
            PRE_STABILITY_OUTPUT_DIR
            / f"h3_pre_stability_{family_name}_{shock_name}_treated_drift_bands.csv",
            index=False,
        )
        placebo_output["control_drift_bands"].to_csv(
            PRE_STABILITY_OUTPUT_DIR
            / f"h3_pre_stability_{family_name}_{shock_name}_control_drift_bands.csv",
            index=False,
        )
        placebo_output["placebo_bands"].to_csv(
            PRE_STABILITY_OUTPUT_DIR
            / f"h3_pre_stability_{family_name}_{shock_name}_placebo_bands.csv",
            index=False,
        )


def main() -> None:
    """Run the within-pre IRF stability test and save outputs."""

    PRE_STABILITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    family_results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"\nEstimating within-pre IRF stability for {family_name}...")
        family_result = run_family_pre_stability(family_name, n_draws=N_DRAWS)
        _save_family_outputs(family_result)
        family_results.append(family_result)

    run_summary = build_run_summary(family_results, n_draws=N_DRAWS)
    key_summary = build_key_summary(family_results)

    run_summary.to_csv(
        PRE_STABILITY_OUTPUT_DIR / "h3_pre_stability_run_summary.csv", index=False
    )
    key_summary.to_csv(
        PRE_STABILITY_OUTPUT_DIR / "h3_pre_stability_key_summary.csv", index=False
    )

    print("\nWithin-pre stability run summary")
    print(run_summary.to_string(index=False))


if __name__ == "__main__":
    main()

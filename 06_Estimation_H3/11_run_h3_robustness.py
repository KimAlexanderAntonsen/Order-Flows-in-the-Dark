"""Run the targeted robustness checks for H3.

This runner executes the two robustness checks that matter most for 
the H3 benchmark:

1. p = 3 instead of the Menkveld-style p = 2 benchmark.
2. The least-retail reference group instead of the matched control.

The point is not to replace the benchmark. It is to check whether the 
main H3 story survives sensible alternatives.
"""

from __future__ import annotations

import importlib


_robust = importlib.import_module("10_h3_robustness")


ensure_output_dirs = _robust.ensure_output_dirs
run_p3_robustness = _robust.run_p3_robustness
run_reference_control_robustness = _robust.run_reference_control_robustness
save_robustness_outputs = _robust.save_robustness_outputs
ROBUSTNESS_P_LAGS = _robust.ROBUSTNESS_P_LAGS


def main() -> None:
    """Run the H3 robustness block and save the comparison tables."""

    ensure_output_dirs()

    print(f"Running H3 p={ROBUSTNESS_P_LAGS} robustness...")
    p3_results = run_p3_robustness()

    print("\nRunning H3 alternative-control robustness...")
    reference_results = run_reference_control_robustness()

    paths = save_robustness_outputs(
        p3_results=p3_results,
        reference_results=reference_results,
    )

    print("\nSaved H3 robustness outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

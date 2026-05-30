"""Run the p=4 robustness pass for H3.

BIC prefers p=4 in every regime-by-family cell of the H1/H2 lag
selection. This runner produces the matching H3 difference in difference
artefacts so the lag-robustness battery can report both p=3 and p=4 
alongside the p=2 benchmark.
"""

from __future__ import annotations

import importlib


_robust = importlib.import_module("10_h3_robustness")


ensure_output_dirs = _robust.ensure_output_dirs
run_p4_robustness = _robust.run_p4_robustness
save_p4_outputs = _robust.save_p4_outputs
ROBUSTNESS_P_LAGS_P4 = _robust.ROBUSTNESS_P_LAGS_P4


def main() -> None:
    """Run the H3 p=4 robustness block and save the comparison tables."""

    ensure_output_dirs()

    print(f"Running H3 p={ROBUSTNESS_P_LAGS_P4} robustness...")
    p4_results = run_p4_robustness()

    paths = save_p4_outputs(p4_results)

    print("\nSaved H3 p=4 robustness outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

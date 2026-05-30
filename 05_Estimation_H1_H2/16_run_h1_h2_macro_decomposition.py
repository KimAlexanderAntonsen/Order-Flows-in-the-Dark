"""Run the FOMC-only and CPI/PPI-only macro decomposition.

The benchmark macro VARX in `03_run_h1_h2_estimation.py` pools all 22
macro events that survive the Oct 1-10 exclusion window (8 CPI + 8 PPI
+ 6 FOMC; the source list has 9 CPI and 9 PPI, but the Oct 10 CPI and
Oct 8 PPI releases fall inside the exclusion window and are dropped at
sample-load time) into a single shock family. This script re-estimates
the macro VARX on two disjoint event subsets:

1. `macro_fomc`: the 6 FOMC rate-decision events.
2. `macro_inflation`: the 16 CPI/PPI events.

The two refits use the same VARX engine, the same panel iterator 
structure, the same Monte Carlo confidence-band construction, and the 
same pre/post windows. Only the exogenous block differs. 

Outputs follow the same filename convention as the benchmark runner so 
the decomposition results live alongside the headline outputs and can 
be loaded by the same downstream presentation code.
"""

from __future__ import annotations

import importlib

_config = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")
_runner = importlib.import_module("03_run_h1_h2_estimation")

OUTPUT_DIR = _config.OUTPUT_DIR
N_SIMULATION_DRAWS = _config.N_SIMULATION_DRAWS

load_sp500_universe = _helpers.load_sp500_universe
run_family_h1_h2 = _helpers.run_family_h1_h2
build_run_summary = _helpers.build_run_summary
_save_family_outputs = _runner._save_family_outputs


# The two disjoint macro subsets that together recover the combined 
# macro benchmark. 
DECOMPOSITION_FAMILY_NAMES = ("macro_fomc", "macro_inflation")


def main() -> None:
    """Run the macro decomposition and save the IRF outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()

    print(
        "Running macro decomposition on the canonical universe "
        f"of {len(universe)} stocks..."
    )

    family_results: list[dict[str, object]] = []
    for family_name in DECOMPOSITION_FAMILY_NAMES:
        print(f"\nEstimating {family_name}...")
        family_result = run_family_h1_h2(
            family_name,
            tickers=universe,
            n_draws=N_SIMULATION_DRAWS,
        )
        _save_family_outputs(family_result)
        family_results.append(family_result)

    # Summary file is saved separately.
    summary = build_run_summary(family_results, n_draws=N_SIMULATION_DRAWS)
    summary.to_csv(OUTPUT_DIR / "h1h2_macro_decomposition_summary.csv", index=False)

    print("\nMacro decomposition summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

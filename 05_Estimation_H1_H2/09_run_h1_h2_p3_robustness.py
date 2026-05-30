"""Run the final p=3 robustness pass for H1/H2.

The benchmark H1/H2 layer follows Menkveld's p=2 specification. This 
runner adds one extra lag and saves the results separately so we can 
check whether the main conclusions survive a slightly richer dynamic 
structure.
"""

from __future__ import annotations

import importlib


_config = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")
_robust = importlib.import_module("08_h1_h2_robustness")


FAMILY_NAMES = _config.FAMILY_NAMES
N_SIMULATION_DRAWS = _config.N_SIMULATION_DRAWS
ROBUSTNESS_P_LAGS = _robust.ROBUSTNESS_P_LAGS

load_sp500_universe = _helpers.load_sp500_universe
run_family_h1_h2 = _helpers.run_family_h1_h2
build_run_summary = _helpers.build_run_summary

ensure_output_dir = _robust.ensure_output_dir
save_family_outputs = _robust.save_family_outputs
build_key_irf_summary_table = _robust.build_key_irf_summary_table
build_benchmark_comparison_table = _robust.build_benchmark_comparison_table
ROBUSTNESS_OUTPUT_DIR = _robust.ROBUSTNESS_OUTPUT_DIR


def main() -> None:
    """Run the split-sample H1/H2 robustness pass with p=3."""

    ensure_output_dir()
    universe = load_sp500_universe()

    print(
        "Running p=3 H1/H2 robustness on the canonical universe "
        f"of {len(universe)} stocks..."
    )

    family_results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"\nEstimating {family_name} with p={ROBUSTNESS_P_LAGS}...")
        family_result = run_family_h1_h2(
            family_name,
            tickers=universe,
            n_draws=N_SIMULATION_DRAWS,
            p_lags=ROBUSTNESS_P_LAGS,
        )
        save_family_outputs(family_result)
        family_results.append(family_result)

    summary = build_run_summary(
        family_results,
        n_draws=N_SIMULATION_DRAWS,
        p_lags=ROBUSTNESS_P_LAGS,
    )
    summary.to_csv(ROBUSTNESS_OUTPUT_DIR / "h1h2_p3_run_summary.csv", index=False)

    key_summary_path = build_key_irf_summary_table()
    comparison_path = build_benchmark_comparison_table()

    print("\np=3 robustness summary")
    print(summary.to_string(index=False))
    print(f"\nSaved key-horizon summary to {key_summary_path}")
    print(f"Saved p=3 versus p=2 comparison to {comparison_path}")


if __name__ == "__main__":
    main()

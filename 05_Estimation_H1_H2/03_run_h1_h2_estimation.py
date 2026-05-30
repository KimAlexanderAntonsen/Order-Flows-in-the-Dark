"""Run the H1/H2 estimation layer.

The goal of this script is:

1. Reuse the benchmark beta VARX engine.
2. Estimate the benchmark model separately in the pre and post 
   regimes.
3. Save regime-specific IRFs, confidence bands, and post-minus-pre
   comparison objects for VIX, macro, and earnings.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

_config = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")

OUTPUT_DIR = _config.OUTPUT_DIR
FAMILY_NAMES = _config.FAMILY_NAMES
N_SIMULATION_DRAWS = _config.N_SIMULATION_DRAWS

load_sp500_universe = _helpers.load_sp500_universe
run_family_h1_h2 = _helpers.run_family_h1_h2
build_run_summary = _helpers.build_run_summary


def _cluster_diagnostics_row(result, *, family: str, regime: str) -> dict[str, object]:
    """Flatten one regime's cluster diagnostics into a saveable row."""

    diagnostics = dict(result.cluster_diagnostics or {})
    diagnostics.update({"family": family, "regime": regime})
    return diagnostics


def _save_family_outputs(family_result: dict[str, object]) -> None:
    """Save one family's H1/H2 outputs with transparent filenames."""

    family = str(family_result["family"])
    pre_result = family_result["pre_result"]
    post_result = family_result["post_result"]

    pre_result.coefficient_table.to_csv(
        OUTPUT_DIR / f"h1h2_{family}_pre_coefficients.csv",
        index=False,
    )
    post_result.coefficient_table.to_csv(
        OUTPUT_DIR / f"h1h2_{family}_post_coefficients.csv",
        index=False,
    )

    cluster_rows = [
        _cluster_diagnostics_row(pre_result, family=family, regime="pre"),
        _cluster_diagnostics_row(post_result, family=family, regime="post"),
    ]
    pd.DataFrame(cluster_rows).to_csv(
        OUTPUT_DIR / f"h1h2_{family}_cluster_diagnostics.csv",
        index=False,
    )

    for shock_output in family_result["shock_outputs"]:
        shock_name = shock_output["shock_name"]

        shock_output["pre_bands"].to_csv(
            OUTPUT_DIR / f"h1h2_{family}_{shock_name}_pre_bands.csv",
            index=False,
        )
        shock_output["post_bands"].to_csv(
            OUTPUT_DIR / f"h1h2_{family}_{shock_name}_post_bands.csv",
            index=False,
        )
        shock_output["difference_bands"].to_csv(
            OUTPUT_DIR / f"h1h2_{family}_{shock_name}_post_minus_pre_bands.csv",
            index=False,
        )

        shock_output["pre_diagnostics"].to_csv(
            OUTPUT_DIR / f"h1h2_{family}_{shock_name}_pre_draw_diagnostics.csv",
            index=False,
        )
        shock_output["post_diagnostics"].to_csv(
            OUTPUT_DIR / f"h1h2_{family}_{shock_name}_post_draw_diagnostics.csv",
            index=False,
        )


def main() -> None:
    """Run the benchmark H1/H2 estimation layer and save the outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()

    print(
        "Running H1/H2 estimation on the canonical universe "
        f"of {len(universe)} stocks..."
    )

    family_results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"\nEstimating {family_name}...")
        family_result = run_family_h1_h2(
            family_name,
            tickers=universe,
            n_draws=N_SIMULATION_DRAWS,
        )
        _save_family_outputs(family_result)
        family_results.append(family_result)

    summary = build_run_summary(family_results, n_draws=N_SIMULATION_DRAWS)
    summary.to_csv(OUTPUT_DIR / "h1h2_run_summary.csv", index=False)

    print("\nH1/H2 estimation summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

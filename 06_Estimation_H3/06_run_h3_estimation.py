"""Run the benchmark H3 estimation layer.

This script is the H3 counterpart to the H1/H2 runner. It does four 
things:

1. Reuse the finished treated and matched-control groups.
2. Estimate the pre and post benchmark VARX separately for each group.
3. Construct the benchmark H3 object:

       (treated post - treated pre) - (control post - control pre)

4. Save the intermediate and final outputs with transparent filenames.
"""

from __future__ import annotations

import importlib

import pandas as pd


_config = importlib.import_module("01_h3_config")
_helpers = importlib.import_module("05_h3_estimation")


ESTIMATION_OUTPUT_DIR = _config.ESTIMATION_OUTPUT_DIR
FAMILY_NAMES = _config.FAMILY_NAMES
N_SIMULATION_DRAWS = _config.N_SIMULATION_DRAWS

run_family_h3 = _helpers.run_family_h3
build_run_summary = _helpers.build_run_summary
build_key_summary = _helpers.build_key_summary


def _cluster_rows(group_result: dict[str, object], family_name: str) -> list[dict[str, object]]:
    """Flatten pre/post cluster diagnostics for one group."""

    group_name = str(group_result["group"])
    rows: list[dict[str, object]] = []
    for regime_name, key in (("pre", "pre_result"), ("post", "post_result")):
        diagnostics = dict(getattr(group_result[key], "cluster_diagnostics", None) or {})
        diagnostics.update({"family": family_name, "group": group_name, "regime": regime_name})
        rows.append(diagnostics)
    return rows


def _save_group_outputs(group_result: dict[str, object], family_name: str) -> None:
    """Save one group's coefficients, bands, and diagnostics."""

    group_name = str(group_result["group"])
    group_result["pre_result"].coefficient_table.to_csv(
        ESTIMATION_OUTPUT_DIR / f"h3_{family_name}_{group_name}_pre_coefficients.csv",
        index=False,
    )
    group_result["post_result"].coefficient_table.to_csv(
        ESTIMATION_OUTPUT_DIR / f"h3_{family_name}_{group_name}_post_coefficients.csv",
        index=False,
    )

    for shock_output in group_result["shock_outputs"]:
        shock_name = str(shock_output["shock_name"])

        shock_output["pre_bands"].to_csv(
            ESTIMATION_OUTPUT_DIR / f"h3_{family_name}_{shock_name}_{group_name}_pre_bands.csv",
            index=False,
        )
        shock_output["post_bands"].to_csv(
            ESTIMATION_OUTPUT_DIR / f"h3_{family_name}_{shock_name}_{group_name}_post_bands.csv",
            index=False,
        )
        shock_output["change_bands"].to_csv(
            ESTIMATION_OUTPUT_DIR
            / f"h3_{family_name}_{shock_name}_{group_name}_post_minus_pre_bands.csv",
            index=False,
        )
        shock_output["pre_diagnostics"].to_csv(
            ESTIMATION_OUTPUT_DIR
            / f"h3_{family_name}_{shock_name}_{group_name}_pre_draw_diagnostics.csv",
            index=False,
        )
        shock_output["post_diagnostics"].to_csv(
            ESTIMATION_OUTPUT_DIR
            / f"h3_{family_name}_{shock_name}_{group_name}_post_draw_diagnostics.csv",
            index=False,
        )


def _save_h3_outputs(family_result: dict[str, object]) -> None:
    """Save the final treated-minus-control H3 bands and summaries."""

    family_name = str(family_result["family"])
    for h3_output in family_result["h3_outputs"]:
        shock_name = str(h3_output["shock_name"])
        h3_output["h3_bands"].to_csv(
            ESTIMATION_OUTPUT_DIR
            / f"h3_{family_name}_{shock_name}_treated_minus_control_post_minus_pre_bands.csv",
            index=False,
        )


def main() -> None:
    """Run the benchmark H3 estimation layer and save the outputs."""

    ESTIMATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    family_results: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"\nEstimating H3 benchmark for {family_name}...")
        family_result = run_family_h3(
            family_name,
            n_draws=N_SIMULATION_DRAWS,
        )
        _save_group_outputs(family_result["treated_result"], family_name)
        _save_group_outputs(family_result["control_result"], family_name)
        _save_h3_outputs(family_result)
        cluster_rows.extend(_cluster_rows(family_result["treated_result"], family_name))
        cluster_rows.extend(_cluster_rows(family_result["control_result"], family_name))
        family_results.append(family_result)

    run_summary = build_run_summary(family_results, n_draws=N_SIMULATION_DRAWS)
    key_summary = build_key_summary(family_results)

    run_summary.to_csv(ESTIMATION_OUTPUT_DIR / "h3_run_summary.csv", index=False)
    key_summary.to_csv(ESTIMATION_OUTPUT_DIR / "h3_key_triple_difference_summary.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        ESTIMATION_OUTPUT_DIR / "h3_cluster_diagnostics.csv",
        index=False,
    )

    print("\nH3 estimation summary")
    print(run_summary.to_string(index=False))


if __name__ == "__main__":
    main()

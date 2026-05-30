"""Targeted robustness checks for the H3 estimation layer.

This module keeps the H3 robustness block narrow and easy to follow.

The benchmark H3 result already exists, so the goal here is simply to 
ask whether the main H3 reading survives two natural stress tests:

1. A p=3 VARX specification instead of the Menkveld-style p=2 
   benchmark.
2. The least-retail reference group as an alternative control group.

The first check asks whether the dynamic lag choice matters. The 
second asks whether the benchmark findings are specific to the 
matched-control design or whether they also show up against the more 
extreme least-retail tail.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

_config = importlib.import_module("01_h3_config")
_helpers = importlib.import_module("05_h3_estimation")


ROBUSTNESS_P_LAGS = 3
ROBUSTNESS_P_LAGS_P4 = 4
ROBUSTNESS_N_DRAWS = 5000

ESTIMATION_OUTPUT_DIR = _config.ESTIMATION_OUTPUT_DIR
PRESENTATION_TABLE_DIR = _config.PRESENTATION_TABLE_DIR
ROBUSTNESS_P3_DIR = _config.ROBUSTNESS_P3_DIR
ROBUSTNESS_P4_DIR = _config.ROBUSTNESS_P4_DIR
ROBUSTNESS_REFERENCE_DIR = _config.ROBUSTNESS_REFERENCE_DIR

FAMILY_NAMES = _config.FAMILY_NAMES
run_family_h3 = _helpers.run_family_h3
build_run_summary = _helpers.build_run_summary
build_key_summary = _helpers.build_key_summary


def ensure_output_dirs() -> None:
    """Create the robustness folders if they do not already exist."""

    ROBUSTNESS_P3_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_P4_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save_group_outputs(
    family_result: dict[str, object],
    *,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save one robustness family's group-specific outputs.

    The file names deliberately mirror the benchmark H3 names so it 
    stays easy to compare the folders side by side.
    """

    family_name = str(family_result["family"])

    for group_key in ("treated_result", "control_result"):
        group_result = family_result[group_key]
        group_name = str(group_result["group"])

        group_result["pre_result"].coefficient_table.to_csv(
            output_dir / f"{prefix}_{family_name}_{group_name}_pre_coefficients.csv",
            index=False,
        )
        group_result["post_result"].coefficient_table.to_csv(
            output_dir / f"{prefix}_{family_name}_{group_name}_post_coefficients.csv",
            index=False,
        )

        for shock_output in group_result["shock_outputs"]:
            shock_name = str(shock_output["shock_name"])

            shock_output["pre_bands"].to_csv(
                output_dir / f"{prefix}_{family_name}_{shock_name}_{group_name}_pre_bands.csv",
                index=False,
            )
            shock_output["post_bands"].to_csv(
                output_dir / f"{prefix}_{family_name}_{shock_name}_{group_name}_post_bands.csv",
                index=False,
            )
            shock_output["change_bands"].to_csv(
                output_dir
                / f"{prefix}_{family_name}_{shock_name}_{group_name}_post_minus_pre_bands.csv",
                index=False,
            )
            shock_output["pre_diagnostics"].to_csv(
                output_dir
                / f"{prefix}_{family_name}_{shock_name}_{group_name}_pre_draw_diagnostics.csv",
                index=False,
            )
            shock_output["post_diagnostics"].to_csv(
                output_dir
                / f"{prefix}_{family_name}_{shock_name}_{group_name}_post_draw_diagnostics.csv",
                index=False,
            )


def _save_h3_outputs(
    family_result: dict[str, object],
    *,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save one family's final difference in difference bands tables."""

    family_name = str(family_result["family"])
    control_group = str(family_result["control_group"])
    triple_label = f"treated_minus_{control_group}"

    for h3_output in family_result["h3_outputs"]:
        shock_name = str(h3_output["shock_name"])
        h3_output["h3_bands"].to_csv(
            output_dir
            / f"{prefix}_{family_name}_{shock_name}_{triple_label}_post_minus_pre_bands.csv",
            index=False,
        )


def run_p3_robustness() -> list[dict[str, object]]:
    """Run the p=3 H3 robustness pass on the benchmark matched-control design."""

    results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  p=3 robustness: estimating {family_name}...", flush=True)
        family_result = run_family_h3(
            family_name,
            treated_group="treated",
            control_group="matched_control",
            n_draws=ROBUSTNESS_N_DRAWS,
            p_lags=ROBUSTNESS_P_LAGS,
        )
        _save_group_outputs(family_result, output_dir=ROBUSTNESS_P3_DIR, prefix="h3_p3")
        _save_h3_outputs(family_result, output_dir=ROBUSTNESS_P3_DIR, prefix="h3_p3")
        results.append(family_result)
    return results


def run_p4_robustness() -> list[dict[str, object]]:
    """Run the p=4 H3 robustness pass on the benchmark matched-control design."""

    results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  p=4 robustness: estimating {family_name}...", flush=True)
        family_result = run_family_h3(
            family_name,
            treated_group="treated",
            control_group="matched_control",
            n_draws=ROBUSTNESS_N_DRAWS,
            p_lags=ROBUSTNESS_P_LAGS_P4,
        )
        _save_group_outputs(family_result, output_dir=ROBUSTNESS_P4_DIR, prefix="h3_p4")
        _save_h3_outputs(family_result, output_dir=ROBUSTNESS_P4_DIR, prefix="h3_p4")
        results.append(family_result)
    return results


def run_reference_control_robustness() -> list[dict[str, object]]:
    """Run the alternative-control robustness pass with the least-retail group."""

    results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  reference-control robustness: estimating {family_name}...", flush=True)
        family_result = run_family_h3(
            family_name,
            treated_group="treated",
            control_group="least_retail_reference",
            n_draws=ROBUSTNESS_N_DRAWS,
            p_lags=_config.BENCHMARK_P_LAGS,
        )
        _save_group_outputs(
            family_result,
            output_dir=ROBUSTNESS_REFERENCE_DIR,
            prefix="h3_reference",
        )
        _save_h3_outputs(
            family_result,
            output_dir=ROBUSTNESS_REFERENCE_DIR,
            prefix="h3_reference",
        )
        results.append(family_result)
    return results


def _save_run_bundle(
    results: list[dict[str, object]],
    *,
    output_dir: Path,
    run_name: str,
) -> tuple[Path, Path]:
    """Save the compact run summary and key-horizon summary for one robustness."""

    run_summary = build_run_summary(results, n_draws=ROBUSTNESS_N_DRAWS)
    key_summary = build_key_summary(results)

    run_summary_path = output_dir / f"{run_name}_run_summary.csv"
    key_summary_path = output_dir / f"{run_name}_key_triple_difference_summary.csv"

    run_summary.to_csv(run_summary_path, index=False)
    key_summary.to_csv(key_summary_path, index=False)
    return run_summary_path, key_summary_path


def _merge_with_benchmark(
    robustness_key_summary: pd.DataFrame,
    *,
    robustness_label: str,
) -> pd.DataFrame:
    """Compare one robustness key-horizon table to the benchmark H3 summary."""

    benchmark = pd.read_csv(ESTIMATION_OUTPUT_DIR / "h3_key_triple_difference_summary.csv")
    join_cols = ["family", "shock_name", "horizon", "horizon_label"]
    compare_cols = [
        "dark_share_change_bps_point",
        "dark_share_change_bps_lower95",
        "dark_share_change_bps_upper95",
        "exclude_zero",
    ]

    merged = benchmark[join_cols + compare_cols].merge(
        robustness_key_summary[join_cols + compare_cols],
        on=join_cols,
        suffixes=("_benchmark", f"_{robustness_label}"),
    )
    merged["robustness_label"] = robustness_label
    merged["point_delta_vs_benchmark"] = (
        merged[f"dark_share_change_bps_point_{robustness_label}"]
        - merged["dark_share_change_bps_point_benchmark"]
    )
    merged["lower_delta_vs_benchmark"] = (
        merged[f"dark_share_change_bps_lower95_{robustness_label}"]
        - merged["dark_share_change_bps_lower95_benchmark"]
    )
    merged["upper_delta_vs_benchmark"] = (
        merged[f"dark_share_change_bps_upper95_{robustness_label}"]
        - merged["dark_share_change_bps_upper95_benchmark"]
    )
    return merged


def build_p3_vs_p2_comparison(p3_key_summary: pd.DataFrame) -> Path:
    """Save the benchmark-versus-p3 comparison table."""

    merged = _merge_with_benchmark(p3_key_summary, robustness_label="p3")
    path = ROBUSTNESS_P3_DIR / "h3_p3_vs_p2_comparison.csv"
    merged.to_csv(path, index=False)
    return path


def build_p4_vs_p2_comparison(p4_key_summary: pd.DataFrame) -> Path:
    """Save the benchmark-versus-p4 comparison table."""

    merged = _merge_with_benchmark(p4_key_summary, robustness_label="p4")
    path = ROBUSTNESS_P4_DIR / "h3_p4_vs_p2_comparison.csv"
    merged.to_csv(path, index=False)
    return path


def save_p4_outputs(p4_results: list[dict[str, object]]) -> dict[str, Path]:
    """Save the run summary, key-horizon summary, and comparison for p=4."""

    _, p4_key_path = _save_run_bundle(
        p4_results,
        output_dir=ROBUSTNESS_P4_DIR,
        run_name="h3_p4",
    )
    p4_key_summary = pd.read_csv(p4_key_path)
    p4_comparison_path = build_p4_vs_p2_comparison(p4_key_summary)
    return {
        "p4_key_summary": p4_key_path,
        "p4_comparison": p4_comparison_path,
    }


def build_reference_vs_matched_comparison(reference_key_summary: pd.DataFrame) -> Path:
    """Save the benchmark-versus-reference-control comparison table."""

    merged = _merge_with_benchmark(reference_key_summary, robustness_label="reference")
    path = ROBUSTNESS_REFERENCE_DIR / "h3_reference_vs_matched_comparison.csv"
    merged.to_csv(path, index=False)
    return path


def build_robustness_dashboard(
    *,
    p3_key_summary: pd.DataFrame,
    reference_key_summary: pd.DataFrame,
) -> Path:
    """Create one compact robustness dashboard for the appendix.

    This table is intentionally qualitative. It highlights whether the 
    benchmark reading appears materially changed once we:

    1. add one extra lag, or
    2. replace the matched control with the least-retail reference 
       group.
    """

    def _reading(frame: pd.DataFrame, family: str) -> str:
        family_frame = frame.loc[frame["family"] == family].copy()
        if family == "vix":
            pos = family_frame.loc[family_frame["shock_name"] == "dVIX_pos_inv"]
            sig = pos.loc[pos["exclude_zero"]]
            if sig.empty:
                return "No clear effect"
            horizons = ", ".join(sig["horizon_label"].astype(str).tolist())
            return f"Negative at {horizons}"
        if family == "macro":
            sig = family_frame.loc[family_frame["exclude_zero"]]
            if sig.empty:
                return "No clear effect"
            horizons = ", ".join(sig["horizon_label"].astype(str).tolist())
            return f"Negative at {horizons}"
        if family == "earnings":
            sig = family_frame.loc[family_frame["exclude_zero"]]
            if sig.empty:
                return "No clear effect"
            horizons = ", ".join(sig["horizon_label"].astype(str).tolist())
            return f"Negative at {horizons}"
        raise ValueError(f"Unknown family: {family}")

    benchmark = pd.read_csv(ESTIMATION_OUTPUT_DIR / "h3_key_triple_difference_summary.csv")
    rows = []
    family_labels = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}
    for family in FAMILY_NAMES:
        rows.append(
            {
                "family": family_labels[family],
                "benchmark_reading": _reading(benchmark, family),
                "p3_reading": _reading(p3_key_summary, family),
                "reference_control_reading": _reading(reference_key_summary, family),
            }
        )

    dashboard = pd.DataFrame(rows)
    path = PRESENTATION_TABLE_DIR / "h3_robustness_dashboard.csv"
    dashboard.to_csv(path, index=False)
    return path


def save_robustness_outputs(
    *,
    p3_results: list[dict[str, object]],
    reference_results: list[dict[str, object]],
) -> dict[str, Path]:
    """Save the summary outputs from the H3 robustness block."""

    _, p3_key_path = _save_run_bundle(
        p3_results,
        output_dir=ROBUSTNESS_P3_DIR,
        run_name="h3_p3",
    )
    _, reference_key_path = _save_run_bundle(
        reference_results,
        output_dir=ROBUSTNESS_REFERENCE_DIR,
        run_name="h3_reference",
    )

    p3_key_summary = pd.read_csv(p3_key_path)
    reference_key_summary = pd.read_csv(reference_key_path)

    p3_comparison_path = build_p3_vs_p2_comparison(p3_key_summary)
    reference_comparison_path = build_reference_vs_matched_comparison(reference_key_summary)
    dashboard_path = build_robustness_dashboard(
        p3_key_summary=p3_key_summary,
        reference_key_summary=reference_key_summary,
    )

    return {
        "p3_key_summary": p3_key_path,
        "reference_key_summary": reference_key_path,
        "p3_comparison": p3_comparison_path,
        "reference_comparison": reference_comparison_path,
        "dashboard": dashboard_path,
    }

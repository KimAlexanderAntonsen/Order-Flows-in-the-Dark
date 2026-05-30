"""Run the Step 2 IRF implementation on top of the Step 1 baseline.

Step 2 deliberately starts from the Menkveld benchmark choice of 
p = 2, even though our earlier diagnostic pass found that p = 3 is 
preferred by BIC. The reason is practical: before we compare 
specifications, we want a clean IRF implementation that mirrors the 
Menkveld setup as closely as possible.

This script does three things:

1. Re-fit the baseline VARX at p = 2 for each urgency family.
2. Construct the intended exogenous shock path for that family.
3. Verify that the direct recursion and companion-form simulation 
   agree.

The shock scaling is now aligned more closely with Menkveld:
- VIX headline IRFs use small innovation shocks,
- macro uses the natural event-dummy path,
- earnings uses a 1% EPS-surprise path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_model = importlib.import_module("06_beta_varx_model")
_panel = importlib.import_module("05_beta_varx_panel")
_irf = importlib.import_module("10_beta_varx_irf")

BETA_VARX_DIR = _config.BETA_VARX_DIR
DEFAULT_RIDGE = _config.DEFAULT_RIDGE
Y_COLS = _config.Y_COLS
VIX_X_COLS = _config.VIX_X_COLS
VIX_IRF_SHOCK_COLS = _config.VIX_IRF_SHOCK_COLS
VIX_IRF_SHOCK_SIZE = _config.VIX_IRF_SHOCK_SIZE
MACRO_X_COLS = _config.MACRO_X_COLS
MACRO_EVENT_SHOCK_SIZE = _config.MACRO_EVENT_SHOCK_SIZE
EARNINGS_X_COLS = _config.EARNINGS_X_COLS
EARNINGS_EVENT_SHOCK_SIZE = _config.EARNINGS_EVENT_SHOCK_SIZE

load_sp500_universe = _data.load_sp500_universe
BaselinePanelVARX = _model.BaselinePanelVARX
iter_vix_panel_pieces = _panel.iter_vix_panel_pieces
iter_macro_panel_pieces = _panel.iter_macro_panel_pieces
iter_earnings_panel_pieces = _panel.iter_earnings_panel_pieces

add_cumulative_columns = _irf.add_cumulative_columns
add_dark_share_response = _irf.add_dark_share_response
build_earnings_event_path = _irf.build_earnings_event_path
build_macro_event_path = _irf.build_macro_event_path
build_unit_shock_path = _irf.build_unit_shock_path
check_one_off_contemporaneous_response = _irf.check_one_off_contemporaneous_response
estimate_baseline_volume_means = _irf.estimate_baseline_volume_means
simulate_irf_companion = _irf.simulate_irf_companion
simulate_irf_recursive = _irf.simulate_irf_recursive
validate_irf_paths = _irf.validate_irf_paths


# Step 2 starts from the original Menkveld lag choice.
MENKVELD_P_LAGS = 2
STEP2_OUTPUT_DIR = BETA_VARX_DIR / "output" / "step2"


def _fit_step2_result(
    *,
    iterator_factory,
    common_x_cols: tuple[str, ...],
    panel_x_cols: tuple[str, ...],
) -> object:
    """Fit the Step 1 baseline model using the Step 2 lag choice."""

    model = BaselinePanelVARX(
        p_lags=MENKVELD_P_LAGS,
        ridge=DEFAULT_RIDGE,
        entity_fixed_effects=True,
    )
    return model.fit_from_iterator(
        iterator_factory(),
        entity_col="asset",
        time_col="timestamp",
        y_cols=Y_COLS,
        common_x_cols=common_x_cols,
        panel_x_cols=panel_x_cols,
    )


def _finalize_irf_output(irf: pd.DataFrame, *, volume_means: dict[str, float]) -> pd.DataFrame:
    """Add the derived dark-share path and cumulative columns."""

    irf = add_dark_share_response(
        irf,
        dark_volume_mean=volume_means["dark_volume_mean"],
        lit_volume_mean=volume_means["lit_volume_mean"],
    )
    return add_cumulative_columns(
        irf,
        value_cols=(
            "log_dark_volume_t",
            "log_lit_volume_t",
            "log_total_realized_variance_t",
            "dark_share_change",
            "dark_share_change_bps",
        ),
    )


def _run_vix_irfs(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one-off VIX IRFs and validate the mechanics."""

    result = _fit_step2_result(
        iterator_factory=lambda: iter_vix_panel_pieces(universe),
        common_x_cols=VIX_X_COLS,
        panel_x_cols=(),
    )
    result.coefficient_table.to_csv(STEP2_OUTPUT_DIR / "step2_vix_coefficients_p2.csv", index=False)

    volume_means = estimate_baseline_volume_means(iter_vix_panel_pieces(universe))
    response_tables: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    contemporaneous_rows: list[dict[str, object]] = []

    # Menkveld's headline VIX IRFs focus on urgency innovations rather 
    # than on the VIX level control. We therefore simulate the two 
    # innovation shocks and keep VIX_close in the fitted model as a 
    # control only.
    for shock_col in VIX_IRF_SHOCK_COLS:
        shock_path = build_unit_shock_path(VIX_X_COLS, shock_col=shock_col, shock_size=VIX_IRF_SHOCK_SIZE)
        recursive = simulate_irf_recursive(result, shock_path, horizon_end=60)
        companion = simulate_irf_companion(result, shock_path, horizon_end=60)

        validation = validate_irf_paths(recursive, companion, value_cols=Y_COLS)
        validation_rows.append({"spec": "vix", "shock_name": shock_col, **validation})

        contemporaneous = check_one_off_contemporaneous_response(
            result,
            recursive,
            shock_col=shock_col,
            shock_size=VIX_IRF_SHOCK_SIZE,
        )
        contemporaneous_rows.append({"spec": "vix", "shock_name": shock_col, **contemporaneous})

        final_irf = _finalize_irf_output(recursive, volume_means=volume_means)
        final_irf.insert(0, "shock_name", shock_col)
        final_irf.insert(0, "spec", "vix")
        response_tables.append(final_irf)

    return (
        pd.concat(response_tables, ignore_index=True),
        pd.DataFrame(validation_rows),
        pd.DataFrame(contemporaneous_rows),
    )


def _run_macro_irf(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the macro event-path IRF and validate it."""

    result = _fit_step2_result(
        iterator_factory=lambda: iter_macro_panel_pieces(universe),
        common_x_cols=MACRO_X_COLS,
        panel_x_cols=(),
    )
    result.coefficient_table.to_csv(STEP2_OUTPUT_DIR / "step2_macro_coefficients_p2.csv", index=False)

    volume_means = estimate_baseline_volume_means(iter_macro_panel_pieces(universe))
    shock_path = build_macro_event_path(shock_size=MACRO_EVENT_SHOCK_SIZE)
    recursive = simulate_irf_recursive(result, shock_path, horizon_end=60)
    companion = simulate_irf_companion(result, shock_path, horizon_end=60)

    validation = pd.DataFrame(
        [
            {
                "spec": "macro",
                "shock_name": "macro_event_path",
                **validate_irf_paths(recursive, companion, value_cols=Y_COLS),
            }
        ]
    )

    final_irf = _finalize_irf_output(recursive, volume_means=volume_means)
    final_irf.insert(0, "shock_name", "macro_event_path")
    final_irf.insert(0, "spec", "macro")
    return final_irf, validation


def _run_earnings_irf(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the earnings event-path IRF and validate it."""

    result = _fit_step2_result(
        iterator_factory=lambda: iter_earnings_panel_pieces(universe),
        common_x_cols=(),
        panel_x_cols=EARNINGS_X_COLS,
    )
    result.coefficient_table.to_csv(STEP2_OUTPUT_DIR / "step2_earnings_coefficients_p2.csv", index=False)

    volume_means = estimate_baseline_volume_means(iter_earnings_panel_pieces(universe))
    shock_path = build_earnings_event_path(
        shock_size=EARNINGS_EVENT_SHOCK_SIZE,
        block_minutes=30,
    )
    recursive = simulate_irf_recursive(result, shock_path, horizon_end=450)
    companion = simulate_irf_companion(result, shock_path, horizon_end=450)

    validation = pd.DataFrame(
        [
            {
                "spec": "earnings",
                "shock_name": "earnings_event_path",
                **validate_irf_paths(recursive, companion, value_cols=Y_COLS),
            }
        ]
    )

    final_irf = _finalize_irf_output(recursive, volume_means=volume_means)
    final_irf.insert(0, "shock_name", "earnings_event_path")
    final_irf.insert(0, "spec", "earnings")
    return final_irf, validation


def main() -> None:
    """Run the Step 2 IRF implementation and save the results."""

    STEP2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()

    print(f"Running Step 2 IRFs on the canonical universe of {len(universe)} stocks...")

    vix_irf, vix_validation, vix_contemporaneous = _run_vix_irfs(universe)
    macro_irf, macro_validation = _run_macro_irf(universe)
    earnings_irf, earnings_validation = _run_earnings_irf(universe)

    vix_irf.to_csv(STEP2_OUTPUT_DIR / "step2_vix_irf.csv", index=False)
    macro_irf.to_csv(STEP2_OUTPUT_DIR / "step2_macro_irf.csv", index=False)
    earnings_irf.to_csv(STEP2_OUTPUT_DIR / "step2_earnings_irf.csv", index=False)

    validation = pd.concat([vix_validation, macro_validation, earnings_validation], ignore_index=True)
    validation.to_csv(STEP2_OUTPUT_DIR / "step2_irf_validation.csv", index=False)
    vix_contemporaneous.to_csv(STEP2_OUTPUT_DIR / "step2_vix_h0_checks.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "spec": "vix",
                "p_lags": MENKVELD_P_LAGS,
                "shock_design": f"one_off_{VIX_IRF_SHOCK_SIZE:g}_innovation_shocks",
                "saved_rows": len(vix_irf),
                "max_abs_diff": float(vix_validation["max_abs_diff"].max()),
                "all_paths_validated": bool(vix_validation["passes_tolerance"].all()),
                "all_h0_checks_passed": bool(vix_contemporaneous["passes_tolerance"].all()),
            },
            {
                "spec": "macro",
                "p_lags": MENKVELD_P_LAGS,
                "shock_design": f"event_path_-1_to_4_x_{MACRO_EVENT_SHOCK_SIZE:g}",
                "saved_rows": len(macro_irf),
                "max_abs_diff": float(macro_validation["max_abs_diff"].max()),
                "all_paths_validated": bool(macro_validation["passes_tolerance"].all()),
                "all_h0_checks_passed": None,
            },
            {
                "spec": "earnings",
                "p_lags": MENKVELD_P_LAGS,
                "shock_design": f"event_path_13_half_hour_blocks_x_{EARNINGS_EVENT_SHOCK_SIZE:g}",
                "saved_rows": len(earnings_irf),
                "max_abs_diff": float(earnings_validation["max_abs_diff"].max()),
                "all_paths_validated": bool(earnings_validation["passes_tolerance"].all()),
                "all_h0_checks_passed": None,
            },
        ]
    )
    summary.to_csv(STEP2_OUTPUT_DIR / "step2_irf_run_summary.csv", index=False)

    print("\nStep 2 IRF validation")
    print(validation.to_string(index=False))
    print("\nVIX horizon-0 checks")
    print(vix_contemporaneous.to_string(index=False))
    print("\nStep 2 run summary")
    print(summary.to_string(index=False))

    if not validation["passes_tolerance"].all():
        raise RuntimeError("At least one IRF path failed the recursion-versus-companion validation.")
    if not vix_contemporaneous["passes_tolerance"].all():
        raise RuntimeError("At least one VIX horizon-0 IRF failed the contemporaneous-response check.")


if __name__ == "__main__":
    main()

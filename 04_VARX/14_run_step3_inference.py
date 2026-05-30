"""Run Step 3 confidence-band simulations for the beta VARX build.

This script keeps the economic setup from Step 2 and adds one new 
layer: uncertainty around the impulse responses. Following Menkveld, 
we draw parameter vectors from a multivariate normal distribution 
centered on the point estimate with covariance equal to the estimated 
parameter covariance matrix.

The Step 3 outputs are 95% confidence bands for the VIX, macro, and 
earnings IRFs, all based on the Menkveld benchmark lag choice p = 2 
and the calibrated Step 2 shock definitions.
"""

from __future__ import annotations

import importlib

import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_model = importlib.import_module("06_beta_varx_model")
_panel = importlib.import_module("05_beta_varx_panel")
_irf = importlib.import_module("10_beta_varx_irf")
_inference = importlib.import_module("13_beta_varx_inference")

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

build_unit_shock_path = _irf.build_unit_shock_path
build_macro_event_path = _irf.build_macro_event_path
build_earnings_event_path = _irf.build_earnings_event_path
estimate_baseline_volume_means = _irf.estimate_baseline_volume_means

simulate_irf_bands = _inference.simulate_irf_bands


MENKVELD_P_LAGS = 2
STEP3_OUTPUT_DIR = BETA_VARX_DIR / "output" / "step3"
N_SIMULATION_DRAWS = 10000
BASE_SEED = 2017


def _fit_step3_result(
    *,
    iterator_factory,
    common_x_cols: tuple[str, ...],
    panel_x_cols: tuple[str, ...],
) -> object:
    """Fit the baseline VARX using the Step 3 benchmark lag choice."""

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


def _run_vix_inference(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run confidence-band simulations for all three VIX shocks."""

    result = _fit_step3_result(
        iterator_factory=lambda: iter_vix_panel_pieces(universe),
        common_x_cols=VIX_X_COLS,
        panel_x_cols=(),
    )
    volume_means = estimate_baseline_volume_means(iter_vix_panel_pieces(universe))

    band_tables: list[pd.DataFrame] = []
    diagnostic_tables: list[pd.DataFrame] = []
    for shock_idx, shock_col in enumerate(VIX_IRF_SHOCK_COLS):
        shock_path = build_unit_shock_path(
            VIX_X_COLS,
            shock_col=shock_col,
            shock_size=VIX_IRF_SHOCK_SIZE,
        )
        bands, diagnostics = simulate_irf_bands(
            base_result=result,
            shock_path=shock_path,
            horizon_end=60,
            dark_volume_mean=volume_means["dark_volume_mean"],
            lit_volume_mean=volume_means["lit_volume_mean"],
            n_draws=N_SIMULATION_DRAWS,
            seed=BASE_SEED + shock_idx,
            require_stable=True,
        )
        bands.insert(0, "shock_name", shock_col)
        bands.insert(0, "spec", "vix")
        diagnostics.insert(0, "shock_name", shock_col)
        diagnostics.insert(0, "spec", "vix")
        band_tables.append(bands)
        diagnostic_tables.append(diagnostics)

    return pd.concat(band_tables, ignore_index=True), pd.concat(diagnostic_tables, ignore_index=True)


def _run_macro_inference(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run confidence-band simulations for the macro event path."""

    result = _fit_step3_result(
        iterator_factory=lambda: iter_macro_panel_pieces(universe),
        common_x_cols=MACRO_X_COLS,
        panel_x_cols=(),
    )
    volume_means = estimate_baseline_volume_means(iter_macro_panel_pieces(universe))

    bands, diagnostics = simulate_irf_bands(
        base_result=result,
        shock_path=build_macro_event_path(shock_size=MACRO_EVENT_SHOCK_SIZE),
        horizon_end=60,
        dark_volume_mean=volume_means["dark_volume_mean"],
        lit_volume_mean=volume_means["lit_volume_mean"],
        n_draws=N_SIMULATION_DRAWS,
        seed=BASE_SEED + 100,
        require_stable=True,
    )
    bands.insert(0, "shock_name", "macro_event_path")
    bands.insert(0, "spec", "macro")
    diagnostics.insert(0, "shock_name", "macro_event_path")
    diagnostics.insert(0, "spec", "macro")
    return bands, diagnostics


def _run_earnings_inference(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run confidence-band simulations for the earnings event path."""

    result = _fit_step3_result(
        iterator_factory=lambda: iter_earnings_panel_pieces(universe),
        common_x_cols=(),
        panel_x_cols=EARNINGS_X_COLS,
    )
    volume_means = estimate_baseline_volume_means(iter_earnings_panel_pieces(universe))

    bands, diagnostics = simulate_irf_bands(
        base_result=result,
        shock_path=build_earnings_event_path(
            shock_size=EARNINGS_EVENT_SHOCK_SIZE,
            block_minutes=30,
        ),
        horizon_end=450,
        dark_volume_mean=volume_means["dark_volume_mean"],
        lit_volume_mean=volume_means["lit_volume_mean"],
        n_draws=N_SIMULATION_DRAWS,
        seed=BASE_SEED + 200,
        require_stable=True,
    )
    bands.insert(0, "shock_name", "earnings_event_path")
    bands.insert(0, "spec", "earnings")
    diagnostics.insert(0, "shock_name", "earnings_event_path")
    diagnostics.insert(0, "spec", "earnings")
    return bands, diagnostics


def _summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Collapse draw diagnostics into one summary row per specification/shock."""

    rows: list[dict[str, object]] = []
    for (spec, shock_name), frame in diagnostics.groupby(["spec", "shock_name"], sort=False):
        attempts = int(frame["attempt_number"].max())
        accepted = int(frame["draw_number"].nunique())
        rows.append(
            {
                "spec": spec,
                "shock_name": shock_name,
                "p_lags": MENKVELD_P_LAGS,
                "n_draws_requested": N_SIMULATION_DRAWS,
                "n_draws_accepted": accepted,
                "attempts": attempts,
                "unstable_rejections": attempts - accepted,
                "acceptance_rate": accepted / float(attempts),
                "max_spectral_radius": float(frame["spectral_radius"].max()),
                "mean_spectral_radius": float(frame["spectral_radius"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run Step 3 confidence-band simulations and save the results."""

    STEP3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()

    print(
        "Running Step 3 confidence-band simulations on the canonical universe "
        f"of {len(universe)} stocks..."
    )

    vix_bands, vix_diagnostics = _run_vix_inference(universe)
    macro_bands, macro_diagnostics = _run_macro_inference(universe)
    earnings_bands, earnings_diagnostics = _run_earnings_inference(universe)

    vix_bands.to_csv(STEP3_OUTPUT_DIR / "step3_vix_irf_bands.csv", index=False)
    macro_bands.to_csv(STEP3_OUTPUT_DIR / "step3_macro_irf_bands.csv", index=False)
    earnings_bands.to_csv(STEP3_OUTPUT_DIR / "step3_earnings_irf_bands.csv", index=False)

    diagnostics = pd.concat([vix_diagnostics, macro_diagnostics, earnings_diagnostics], ignore_index=True)
    diagnostics.to_csv(STEP3_OUTPUT_DIR / "step3_draw_diagnostics.csv", index=False)

    summary = _summarize_diagnostics(diagnostics)
    summary.to_csv(STEP3_OUTPUT_DIR / "step3_inference_run_summary.csv", index=False)

    print("\nStep 3 inference summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

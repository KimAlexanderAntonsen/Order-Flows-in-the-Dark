"""Run a compact diagnostic pass for the Step 1 beta VARX baseline.

The goal is to answer two practical questions before Step 2:

1. Does the Menkveld-style choice of p = 2 look reasonable on our 
   data?
2. Is the estimated autoregressive system stable enough to support 
   IRFs?

To keep the pass lightweight and interpretable, we estimate the 
baseline for lag orders p = 1, 2, 3 and compare information criteria. 
We then check the companion-matrix spectral radius for each fitted 
system.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_model = importlib.import_module("06_beta_varx_model")
_panel = importlib.import_module("05_beta_varx_panel")

BASELINE_MODEL = _config.BASELINE_MODEL
BETA_VARX_DIR = _config.BETA_VARX_DIR
load_sp500_universe = _data.load_sp500_universe
BaselinePanelVARX = _model.BaselinePanelVARX
iter_earnings_panel_pieces = _panel.iter_earnings_panel_pieces
iter_macro_panel_pieces = _panel.iter_macro_panel_pieces
iter_vix_panel_pieces = _panel.iter_vix_panel_pieces


OUTPUT_DIR = BETA_VARX_DIR / "output"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
P_GRID = (1, 2, 3)


def _run_spec_diagnostics(
    *,
    spec_name: str,
    iterator_factory,
    common_x_cols: tuple[str, ...],
    panel_x_cols: tuple[str, ...],
    y_cols: tuple[str, ...],
) -> list[dict[str, object]]:
    """Estimate one specification across the lag grid and collect diagnostics."""

    rows: list[dict[str, object]] = []
    for p_lags in P_GRID:
        model = BaselinePanelVARX(
            p_lags=p_lags,
            ridge=BASELINE_MODEL.ridge,
            entity_fixed_effects=BASELINE_MODEL.entity_fixed_effects,
        )
        result = model.fit_from_iterator(
            iterator_factory(),
            entity_col="asset",
            time_col="timestamp",
            y_cols=y_cols,
            common_x_cols=common_x_cols,
            panel_x_cols=panel_x_cols,
        )
        diagnostics = model.diagnostics_from_iterator(
            iterator_factory(),
            result=result,
            entity_col="asset",
            time_col="timestamp",
            y_cols=y_cols,
            common_x_cols=common_x_cols,
            panel_x_cols=panel_x_cols,
        )
        rows.append(
            {
                "spec": spec_name,
                "p_lags": p_lags,
                "nobs": result.design_info["nobs"],
                "n_entities": result.design_info["n_entities"],
                "k_regressors": result.design_info["k_regressors"],
                "sample_start": result.design_info["sample_start"],
                "sample_end": result.design_info["sample_end"],
                **diagnostics,
            }
        )
    return rows


def main() -> None:
    """Run lag-selection and stability diagnostics for Step 1."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_sp500_universe()
    y_cols = BASELINE_MODEL.y_cols

    rows: list[dict[str, object]] = []
    rows.extend(
        _run_spec_diagnostics(
            spec_name="vix",
            iterator_factory=lambda: iter_vix_panel_pieces(universe),
            common_x_cols=("dVIX_pos_inv", "dVIX_neg_inv", "VIX_close"),
            panel_x_cols=(),
            y_cols=y_cols,
        )
    )
    rows.extend(
        _run_spec_diagnostics(
            spec_name="macro",
            iterator_factory=lambda: iter_macro_panel_pieces(universe),
            common_x_cols=(
                "pre_news_1min",
                "post_news_0min",
                "post_news_1min",
                "post_news_2min",
                "post_news_3min",
                "post_news_4min",
            ),
            panel_x_cols=(),
            y_cols=y_cols,
        )
    )
    rows.extend(
        _run_spec_diagnostics(
            spec_name="earnings",
            iterator_factory=lambda: iter_earnings_panel_pieces(universe),
            common_x_cols=(),
            panel_x_cols=tuple(f"post_ea_{k}" for k in range(1, 14)),
            y_cols=y_cols,
        )
    )

    diagnostics = pd.DataFrame(rows).sort_values(["spec", "p_lags"]).reset_index(drop=True)
    diagnostics.to_csv(DIAGNOSTICS_DIR / "step1_diagnostics_lag_stability.csv", index=False)

    preferred = (
        diagnostics.sort_values(["spec", "bic", "aic"])
        .groupby("spec", as_index=False)
        .first()[["spec", "p_lags", "bic", "aic", "spectral_radius", "is_stable"]]
        .rename(columns={"p_lags": "preferred_p_by_bic"})
    )
    preferred.to_csv(DIAGNOSTICS_DIR / "step1_diagnostics_preferred_lags.csv", index=False)

    print("Step 1 diagnostics")
    print(diagnostics.to_string(index=False))
    print("\nPreferred lag by BIC")
    print(preferred.to_string(index=False))


if __name__ == "__main__":
    main()

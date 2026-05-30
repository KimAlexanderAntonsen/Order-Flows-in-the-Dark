"""Lag-selection BIC sweep for the H1/H2 benchmark.

The Menkveld replication fixes ``p = 2``. This script checks whether 
that choice is empirically supported on the H1/H2 sample by computing 
BIC (and AIC) at ``p = 1, 2, 3, 4`` separately for each urgency family 
and each regime. 
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


H1H2_DIR = Path(__file__).resolve().parent
BETA_DIR = H1H2_DIR.parents[0] / "04_VARX"
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))
if str(BETA_DIR) not in sys.path:
    sys.path.insert(0, str(BETA_DIR))


_local = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")
_model = importlib.import_module("06_beta_varx_model")
_data = importlib.import_module("04_beta_varx_data")


OUTPUT_DIR = _local.OUTPUT_DIR
DEFAULT_RIDGE = _local.DEFAULT_RIDGE
FAMILY_NAMES = _local.FAMILY_NAMES
PRE_WINDOW = _local.PRE_WINDOW
POST_WINDOW = _local.POST_WINDOW

Y_COLS = _helpers.Y_COLS
FAMILY_SPECS = _helpers.FAMILY_SPECS
iter_windowed_pieces = _helpers.iter_windowed_pieces

BaselinePanelVARX = _model.BaselinePanelVARX
load_sp500_universe = _data.load_sp500_universe


CANDIDATE_P_LAGS = (1, 2, 3, 4)
REGIMES = (PRE_WINDOW, POST_WINDOW)


def _diagnose_one_fit(
    family,
    *,
    tickers,
    window,
    p_lags: int,
) -> dict[str, object]:
    """Fit the baseline VARX at one lag length and collect information criteria."""

    model = BaselinePanelVARX(
        p_lags=p_lags,
        ridge=DEFAULT_RIDGE,
        entity_fixed_effects=True,
    )
    materialized = list(tickers)

    def iterator_factory():
        return iter_windowed_pieces(family, tickers=materialized, window=window)

    result = model.fit_from_iterator(
        iterator_factory(),
        entity_col="asset",
        time_col="timestamp",
        y_cols=Y_COLS,
        common_x_cols=family.common_x_cols,
        panel_x_cols=family.panel_x_cols,
    )
    diagnostics = model.diagnostics_from_iterator(
        iterator_factory(),
        result=result,
        entity_col="asset",
        time_col="timestamp",
        y_cols=Y_COLS,
        common_x_cols=family.common_x_cols,
        panel_x_cols=family.panel_x_cols,
    )
    return {
        "family": family.name,
        "regime": window.name,
        "p_lags": int(p_lags),
        "nobs": int(result.design_info["nobs"]),
        "n_entities": int(result.design_info["n_entities"]),
        "k_regressors": int(result.design_info["k_regressors"]),
        "loglik": float(diagnostics["loglik"]),
        "aic": float(diagnostics["aic"]),
        "bic": float(diagnostics["bic"]),
        "spectral_radius": float(diagnostics["spectral_radius"]),
        "is_stable": bool(diagnostics["is_stable"]),
    }


def build_bic_table(universe: list[str]) -> pd.DataFrame:
    """Run the full family x regime x p sweep and return the BIC table."""

    rows: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        family = FAMILY_SPECS[family_name]
        for window in REGIMES:
            for p in CANDIDATE_P_LAGS:
                print(f"  {family_name:<8} {window.name:<4} p={p}")
                rows.append(
                    _diagnose_one_fit(
                        family,
                        tickers=universe,
                        window=window,
                        p_lags=p,
                    )
                )
    return pd.DataFrame(rows)


def _picked_minima(table: pd.DataFrame) -> pd.DataFrame:
    """For each family x regime, mark the BIC-minimising p."""

    out = table.copy()
    out["bic_is_min"] = False
    out["aic_is_min"] = False
    for (family, regime), group in out.groupby(["family", "regime"], sort=False):
        bic_idx = group["bic"].idxmin()
        aic_idx = group["aic"].idxmin()
        out.loc[bic_idx, "bic_is_min"] = True
        out.loc[aic_idx, "aic_is_min"] = True
    return out


def main() -> None:
    """Run the lag-selection BIC sweep and save the outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()
    print(
        f"Running H1/H2 lag-selection sweep on {len(universe)} stocks, "
        f"p = {CANDIDATE_P_LAGS}"
    )
    table = build_bic_table(universe)
    table = _picked_minima(table)
    out_path = OUTPUT_DIR / "h1h2_lag_selection_bic.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

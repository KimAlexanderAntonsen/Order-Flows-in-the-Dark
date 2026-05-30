"""Run the Step 1 audit and full baseline VARX fits.

This script has three jobs:

1. Confirm that the pipeline is using the intended constant-membership
   universe.
2. Audit how that universe maps into the market, Robintrack, and 
   earnings inputs used elsewhere in the project.
3. Run the Step 1 baseline VARX fits on the intended sample and 
   save compact output tables.

The script is deliberately lightweight and reuses the beta modules.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_model = importlib.import_module("06_beta_varx_model")
_panel = importlib.import_module("05_beta_varx_panel")
_utils = importlib.import_module("03_beta_varx_utils")

BASELINE_MODEL = _config.BASELINE_MODEL
BETA_VARX_DIR = _config.BETA_VARX_DIR
EARNINGS_PANEL_PATH = _config.EARNINGS_PANEL_PATH
MINUTE_BAR_DIR = _config.MINUTE_BAR_DIR
RETAIL_DIR = _config.RETAIL_DIR
SP500_PATH = _config.SP500_PATH
TICKER_RENAMES = _config.TICKER_RENAMES
VARX_DATA_DIR = _config.VARX_DATA_DIR

load_sp500_universe = _data.load_sp500_universe
BaselinePanelVARX = _model.BaselinePanelVARX
iter_earnings_panel_pieces = _panel.iter_earnings_panel_pieces
iter_macro_panel_pieces = _panel.iter_macro_panel_pieces
iter_vix_panel_pieces = _panel.iter_vix_panel_pieces
normalize_ticker = _utils.normalize_ticker


OUTPUT_DIR = BETA_VARX_DIR / "output"
AUDIT_DIR = OUTPUT_DIR / "audit"
EARNINGS_COVERAGE_PATH = VARX_DATA_DIR / "data_clean" / "earnings_events_coverage.json"


def _load_canonical_universe_raw() -> pd.DataFrame:
    """Load the canonical stock list without changing ticker labels."""

    frame = pd.read_csv(SP500_PATH)
    ticker_col = "ticker" if "ticker" in frame.columns else frame.columns[0]
    out = frame[[ticker_col]].rename(columns={ticker_col: "ticker_raw"}).copy()
    out["ticker"] = out["ticker_raw"].map(lambda value: normalize_ticker(value, TICKER_RENAMES))
    return out


def _minute_bar_coverage(universe: list[str]) -> tuple[list[str], list[str]]:
    """Return covered and missing tickers after ticker normalization."""

    available = {
        normalize_ticker(path.stem.replace("_1m_lit_dark", ""), TICKER_RENAMES)
        for path in MINUTE_BAR_DIR.glob("*_1m_lit_dark.csv")
    }
    covered = sorted(set(universe) & available)
    missing = sorted(set(universe) - available)
    return covered, missing


def _write_audit_tables() -> pd.DataFrame:
    """Create and save the constant-universe audit."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    universe_raw = _load_canonical_universe_raw()
    universe = universe_raw["ticker"].tolist()
    universe_set = set(universe)

    covered_minute_bars, missing_minute_bars = _minute_bar_coverage(universe)

    retail_scores = pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_score_asset_table.csv")
    retail_tickers = set(retail_scores["ticker"].astype(str))
    missing_retail_scores = sorted(universe_set - retail_tickers)

    treated = pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_treated_group.csv")
    matched = pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_matched_control_group.csv")
    reference = pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_reference_group.csv")

    earnings_sparse = pd.read_csv(EARNINGS_PANEL_PATH, usecols=["asset"]).drop_duplicates()
    earnings_assets = set(earnings_sparse["asset"].astype(str))
    missing_earnings_assets = sorted(universe_set - earnings_assets)
    earnings_raw_matched_tickers = None
    if EARNINGS_COVERAGE_PATH.exists():
        coverage_payload = json.loads(EARNINGS_COVERAGE_PATH.read_text())
        earnings_raw_matched_tickers = int(coverage_payload.get("matched_tickers", 0))

    audit_rows = [
        {"metric": "constant_universe_count", "value": len(universe)},
        {"metric": "minute_bar_coverage_count", "value": len(covered_minute_bars)},
        {"metric": "minute_bar_missing_count", "value": len(missing_minute_bars)},
        {"metric": "retail_score_asset_count", "value": retail_scores["ticker"].nunique()},
        {"metric": "retail_score_missing_count", "value": len(missing_retail_scores)},
        {"metric": "treated_group_size", "value": treated["ticker"].nunique()},
        {"metric": "matched_control_group_size", "value": matched["ticker"].nunique()},
        {"metric": "least_retail_reference_group_size", "value": reference["ticker"].nunique()},
        {"metric": "earnings_raw_matched_ticker_count", "value": earnings_raw_matched_tickers},
        {"metric": "earnings_sparse_asset_count", "value": len(earnings_assets)},
        {"metric": "earnings_sparse_missing_count", "value": len(missing_earnings_assets)},
    ]
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_DIR / "step1_universe_audit.csv", index=False)

    pd.DataFrame({"ticker": missing_minute_bars}).to_csv(
        AUDIT_DIR / "step1_missing_minute_bars.csv",
        index=False,
    )
    pd.DataFrame({"ticker": missing_retail_scores}).to_csv(
        AUDIT_DIR / "step1_missing_retail_score_assets.csv",
        index=False,
    )
    pd.DataFrame({"ticker": missing_earnings_assets}).to_csv(
        AUDIT_DIR / "step1_missing_earnings_assets.csv",
        index=False,
    )

    audit_payload = {
        "universe_raw_tickers": universe_raw["ticker_raw"].tolist(),
        "universe_normalized_tickers": universe,
        "minute_bar_missing": missing_minute_bars,
        "retail_score_missing": missing_retail_scores,
        "earnings_sparse_missing": missing_earnings_assets,
    }
    (AUDIT_DIR / "step1_universe_audit.json").write_text(json.dumps(audit_payload, indent=2))

    return audit


def _fit_spec(
    *,
    name: str,
    iterator_factory,
    common_x_cols: tuple[str, ...],
    panel_x_cols: tuple[str, ...],
    y_cols: tuple[str, ...],
) -> dict[str, object]:
    """Fit one baseline specification and save its coefficient table."""

    model = BaselinePanelVARX(
        p_lags=BASELINE_MODEL.p_lags,
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
    result.coefficient_table.to_csv(OUTPUT_DIR / f"step1_baseline_{name}_coefficients.csv", index=False)

    return {
        "spec": name,
        **result.design_info,
        "y_cols": "|".join(result.y_cols),
        "common_x_cols": "|".join(result.common_x_cols),
        "panel_x_cols": "|".join(result.panel_x_cols),
    }


def main() -> None:
    """Run the final Step 1 audit and baseline fits."""

    audit = _write_audit_tables()
    print("Step 1 universe audit")
    print(audit.to_string(index=False))

    universe = load_sp500_universe()
    print(f"\nRunning Step 1 baseline on the canonical universe of {len(universe)} stocks...")

    summaries = [
        _fit_spec(
            name="vix",
            iterator_factory=lambda: iter_vix_panel_pieces(universe),
            common_x_cols=("dVIX_pos_inv", "dVIX_neg_inv", "VIX_close"),
            panel_x_cols=(),
            y_cols=BASELINE_MODEL.y_cols,
        ),
        _fit_spec(
            name="macro",
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
            y_cols=BASELINE_MODEL.y_cols,
        ),
        _fit_spec(
            name="earnings",
            iterator_factory=lambda: iter_earnings_panel_pieces(universe),
            common_x_cols=(),
            panel_x_cols=tuple(f"post_ea_{k}" for k in range(1, 14)),
            y_cols=BASELINE_MODEL.y_cols,
        ),
    ]

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "step1_baseline_run_summary.csv", index=False)
    print("\nBaseline run summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

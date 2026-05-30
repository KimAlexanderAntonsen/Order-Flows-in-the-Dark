"""Panel construction for the beta VARX baseline.

This module does two related jobs:

1. Build stock panels as ordinary DataFrames.
2. Expose asset-by-asset iterators that let the full baseline run on 
   the complete stock universe without first stacking the entire 
   sample in memory.

The second path matters for Step 1. The full sample is large enough 
that a fully materialized stock-minute panel can become unnecessarily 
heavy, even though the underlying model is still simple. By yielding 
one stock at a time, we can keep the economic specification unchanged 
while making the full run practical and reproducible.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Iterator

import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_utils = importlib.import_module("03_beta_varx_utils")

EARNINGS_X_COLS = _config.EARNINGS_X_COLS
MACRO_SESSION = _config.MACRO_SESSION
MACRO_X_COLS = _config.MACRO_X_COLS
MACRO_FOMC_X_COLS = _config.MACRO_FOMC_X_COLS
MACRO_INFLATION_X_COLS = _config.MACRO_INFLATION_X_COLS
REALIZED_VARIANCE_FLOOR = _config.REALIZED_VARIANCE_FLOOR
REGULAR_SESSION = _config.REGULAR_SESSION
VOLUME_FLOOR = _config.VOLUME_FLOOR
VIX_X_COLS = _config.VIX_X_COLS
Y_COLS = _config.Y_COLS

load_earnings_panel = _data.load_earnings_panel
load_macro_panel = _data.load_macro_panel
load_macro_fomc_panel = _data.load_macro_fomc_panel
load_macro_inflation_panel = _data.load_macro_inflation_panel
load_minute_bar = _data.load_minute_bar
load_sp500_universe = _data.load_sp500_universe
load_vix_panel = _data.load_vix_panel

filter_session = _utils.filter_session
safe_log = _utils.safe_log


@dataclass
class PanelSpec:
    """Minimal description of a baseline panel to be estimated."""

    name: str
    df: pd.DataFrame
    y_cols: tuple[str, ...]
    common_x_cols: tuple[str, ...]
    panel_x_cols: tuple[str, ...]


def _build_single_asset_endogenous_panel(
    ticker: str,
    *,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    """Construct the reduced endogenous vector for one stock.

    The baseline endogenous system uses:
    - dark volume,
    - lit volume,
    - total realized variance,
    all in logs.
    """

    frame = load_minute_bar(ticker)
    frame = filter_session(
        frame,
        timestamp_col="timestamp",
        session_start=session_start,
        session_end=session_end,
    )

    dark_volume = pd.to_numeric(frame["dark_volume"], errors="coerce").fillna(0.0)
    lit_volume = pd.to_numeric(frame["lit_volume"], errors="coerce").fillna(0.0)
    total_realized_variance = (
        pd.to_numeric(frame["dark_realized_variance"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["lit_realized_variance"], errors="coerce").fillna(0.0)
    )

    out = pd.DataFrame(
        {
            "asset": frame["asset"].to_numpy(),
            "timestamp": frame["timestamp"].to_numpy(),
            "dark_volume_t": dark_volume.to_numpy(),
            "lit_volume_t": lit_volume.to_numpy(),
            "total_realized_variance_t": total_realized_variance.to_numpy(),
        }
    )
    out["log_dark_volume_t"] = safe_log(out["dark_volume_t"], VOLUME_FLOOR)
    out["log_lit_volume_t"] = safe_log(out["lit_volume_t"], VOLUME_FLOOR)
    out["log_total_realized_variance_t"] = safe_log(
        out["total_realized_variance_t"],
        REALIZED_VARIANCE_FLOOR,
    )
    return out


def iter_endogenous_panel_pieces(
    tickers: Iterable[str] | None = None,
    *,
    session_start: str,
    session_end: str,
) -> Iterator[pd.DataFrame]:
    """Yield one stock's endogenous panel at a time."""

    if tickers is None:
        tickers = load_sp500_universe()

    for ticker in tickers:
        yield _build_single_asset_endogenous_panel(
            ticker,
            session_start=session_start,
            session_end=session_end,
        )


def build_endogenous_panel(
    tickers: Iterable[str] | None = None,
    *,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    """Build a true stock-minute panel from the raw minute-bar files."""

    pieces = list(
        iter_endogenous_panel_pieces(
            tickers=tickers,
            session_start=session_start,
            session_end=session_end,
        )
    )
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    return panel[["asset", "timestamp", "dark_volume_t", "lit_volume_t", "total_realized_variance_t", *Y_COLS]]


def _merge_common_exog(panel: pd.DataFrame, common_exog: pd.DataFrame, x_cols: tuple[str, ...]) -> pd.DataFrame:
    """Merge common exogenous variables by timestamp only."""

    merged = panel.merge(common_exog[["timestamp", *x_cols]], on="timestamp", how="inner")
    return merged


def _merge_panel_exog(panel: pd.DataFrame, panel_exog: pd.DataFrame, x_cols: tuple[str, ...]) -> pd.DataFrame:
    """Merge firm-specific exogenous variables by asset and timestamp."""

    merged = panel.merge(panel_exog[["asset", "timestamp", *x_cols]], on=["asset", "timestamp"], how="left")
    merged[list(x_cols)] = merged[list(x_cols)].fillna(0.0)
    return merged


def build_vix_panel(tickers: Iterable[str] | None = None) -> PanelSpec:
    """Baseline panel for the VIX specification."""

    panel = build_endogenous_panel(
        tickers=tickers,
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    )
    vix = load_vix_panel()
    panel = _merge_common_exog(panel, vix, VIX_X_COLS)
    return PanelSpec(
        name="vix",
        df=panel,
        y_cols=Y_COLS,
        common_x_cols=VIX_X_COLS,
        panel_x_cols=(),
    )


def iter_vix_panel_pieces(tickers: Iterable[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield stock-minute VIX panels one asset at a time."""

    vix = load_vix_panel()
    for piece in iter_endogenous_panel_pieces(
        tickers=tickers,
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    ):
        yield _merge_common_exog(piece, vix, VIX_X_COLS)


def build_macro_panel(tickers: Iterable[str] | None = None) -> PanelSpec:
    """Baseline panel for the macro-news specification."""

    panel = build_endogenous_panel(
        tickers=tickers,
        session_start=MACRO_SESSION.start,
        session_end=MACRO_SESSION.end,
    )
    macro = load_macro_panel()
    panel = _merge_common_exog(panel, macro, MACRO_X_COLS)
    return PanelSpec(
        name="macro",
        df=panel,
        y_cols=Y_COLS,
        common_x_cols=MACRO_X_COLS,
        panel_x_cols=(),
    )


def iter_macro_panel_pieces(tickers: Iterable[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield stock-minute macro panels one asset at a time."""

    macro = load_macro_panel()
    for piece in iter_endogenous_panel_pieces(
        tickers=tickers,
        session_start=MACRO_SESSION.start,
        session_end=MACRO_SESSION.end,
    ):
        yield _merge_common_exog(piece, macro, MACRO_X_COLS)


def iter_macro_fomc_panel_pieces(
    tickers: Iterable[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield stock-minute panels with the FOMC-only macro dummies.

    Mirrors :func:`iter_macro_panel_pieces` but merges only the 
    rate-decision columns. Used by the macro decomposition diagnostic.
    """

    macro_fomc = load_macro_fomc_panel()
    for piece in iter_endogenous_panel_pieces(
        tickers=tickers,
        session_start=MACRO_SESSION.start,
        session_end=MACRO_SESSION.end,
    ):
        yield _merge_common_exog(piece, macro_fomc, MACRO_FOMC_X_COLS)


def iter_macro_inflation_panel_pieces(
    tickers: Iterable[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield stock-minute panels with the CPI/PPI-only macro dummies.

    Mirrors :func:`iter_macro_panel_pieces` but merges only the 
    inflation columns. Used together with 
    :func:`iter_macro_fomc_panel_pieces` for the macro decomposition 
    diagnostic.
    """

    macro_inflation = load_macro_inflation_panel()
    for piece in iter_endogenous_panel_pieces(
        tickers=tickers,
        session_start=MACRO_SESSION.start,
        session_end=MACRO_SESSION.end,
    ):
        yield _merge_common_exog(piece, macro_inflation, MACRO_INFLATION_X_COLS)


def build_earnings_panel(tickers: Iterable[str] | None = None) -> PanelSpec:
    """Baseline panel for the earnings specification."""

    panel = build_endogenous_panel(
        tickers=tickers,
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    )
    earnings = load_earnings_panel()
    if tickers is not None:
        earnings = earnings[earnings["asset"].isin(list(tickers))].copy()
    panel = _merge_panel_exog(panel, earnings, EARNINGS_X_COLS)
    return PanelSpec(
        name="earnings",
        df=panel,
        y_cols=Y_COLS,
        common_x_cols=(),
        panel_x_cols=EARNINGS_X_COLS,
    )


def iter_earnings_panel_pieces(tickers: Iterable[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield stock-minute earnings panels one asset at a time.

    Earnings regressors are sparse and firm-specific, so we pre-split 
    the earnings panel by asset once and then merge only the relevant 
    rows into each stock's market data.
    """

    earnings = load_earnings_panel()
    if tickers is not None:
        tickers = list(tickers)
        earnings = earnings[earnings["asset"].isin(tickers)].copy()

    earnings_by_asset = {
        asset: frame.copy()
        for asset, frame in earnings.groupby("asset", sort=False)
    }

    for piece in iter_endogenous_panel_pieces(
        tickers=tickers,
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    ):
        asset = str(piece["asset"].iloc[0])
        asset_earnings = earnings_by_asset.get(asset)
        if asset_earnings is None:
            out = piece.copy()
            out[list(EARNINGS_X_COLS)] = 0.0
            yield out
            continue
        yield _merge_panel_exog(piece, asset_earnings, EARNINGS_X_COLS)

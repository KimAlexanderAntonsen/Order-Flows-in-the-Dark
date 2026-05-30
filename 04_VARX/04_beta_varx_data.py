"""Data loading layer for the beta VARX build.

This module loads existing project files in a consistent format that 
Step 1 can use.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# These files use numeric prefixes so the project steps are easy to 
# scan. We load the sibling modules through importlib and then pull 
# out the names we need.
_config = importlib.import_module("02_beta_varx_config")
_utils = importlib.import_module("03_beta_varx_utils")

ANALYSIS_EXCLUDE_WINDOWS = _config.ANALYSIS_EXCLUDE_WINDOWS
EARNINGS_PANEL_PATH = _config.EARNINGS_PANEL_PATH
EARNINGS_X_COLS = _config.EARNINGS_X_COLS
MACRO_PANEL_PATH = _config.MACRO_PANEL_PATH
MACRO_X_COLS = _config.MACRO_X_COLS
MACRO_FOMC_X_COLS = _config.MACRO_FOMC_X_COLS
MACRO_INFLATION_X_COLS = _config.MACRO_INFLATION_X_COLS
MINUTE_BAR_DIR = _config.MINUTE_BAR_DIR
SAMPLE_END = _config.SAMPLE_END
SAMPLE_START = _config.SAMPLE_START
SP500_PATH = _config.SP500_PATH
TICKER_RENAMES = _config.TICKER_RENAMES
VIX_PATH = _config.VIX_PATH
VIX_X_COLS = _config.VIX_X_COLS

ensure_required_columns = _utils.ensure_required_columns
filter_sample_window = _utils.filter_sample_window
minute_bar_utc_to_local_minute_end = _utils.minute_bar_utc_to_local_minute_end
normalize_ticker = _utils.normalize_ticker
parse_naive_timestamp = _utils.parse_naive_timestamp


def load_sp500_universe() -> list[str]:
    """Load the canonical constant-membership S&P 500 universe.

    Downstream code uses `sp500_tickers.csv` as the restricted 
    stock universe that stays in the index throughout the analysis 
    sample window.
    """

    universe = pd.read_csv(SP500_PATH)
    ticker_col = "ticker" if "ticker" in universe.columns else universe.columns[0]
    tickers = [normalize_ticker(t, TICKER_RENAMES) for t in universe[ticker_col]]
    return tickers


def load_minute_bar(ticker: str) -> pd.DataFrame:
    """Load one stock's minute-bar file in a baseline-friendly format."""

    path = MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv"
    usecols = [
        "ticker",
        "timestamp",
        "dark_volume",
        "lit_volume",
        "dark_realized_variance",
        "lit_realized_variance",
        "dark_close",
        "lit_close",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    ensure_required_columns(frame, usecols, label=path.name)

    frame["timestamp"] = minute_bar_utc_to_local_minute_end(frame["timestamp"])
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )
    frame["asset"] = ticker
    return frame


def load_vix_panel() -> pd.DataFrame:
    """Load intraday VIX and construct Menkveld-style innovations.

    The VIX file is already minute-based. We parse the date and time 
    columns into a single naive New York timestamp and then estimate 
    a simple AR(1) on minute-by-minute changes in VIX:

        dVIX_t = a + b * dVIX_{t-1} + u_t

    The residual u_t is treated as the unexpected VIX innovation and
    is split into positive and negative components.
    """

    frame = pd.read_csv(VIX_PATH)
    ensure_required_columns(frame, ["Date", "Time", "Close"], label=VIX_PATH.name)

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["time_str"] = frame["Time"].astype(str).str.zfill(4)
    frame["timestamp"] = pd.to_datetime(
        frame["Date"].dt.strftime("%Y-%m-%d") + " " + frame["time_str"],
        format="%Y-%m-%d %H%M",
    )
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )

    frame = frame.rename(columns={"Close": "VIX_close"}).sort_values("timestamp").reset_index(drop=True)
    frame["dVIX"] = frame["VIX_close"].diff()
    frame["dVIX_l1"] = frame["dVIX"].shift(1)

    ar_sample = frame.dropna(subset=["dVIX", "dVIX_l1"]).copy()
    x = np.column_stack([np.ones(len(ar_sample)), ar_sample["dVIX_l1"].to_numpy(dtype=float)])
    y = ar_sample["dVIX"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta

    frame["dVIX_innovation"] = np.nan
    frame.loc[ar_sample.index, "dVIX_innovation"] = resid
    frame["dVIX_pos_inv"] = frame["dVIX_innovation"].clip(lower=0.0)
    frame["dVIX_neg_inv"] = -frame["dVIX_innovation"].clip(upper=0.0)

    return frame[["timestamp", *VIX_X_COLS]].dropna().reset_index(drop=True)


def load_macro_panel() -> pd.DataFrame:
    """Load the already-constructed minute-level macro-news panel."""

    frame = pd.read_csv(MACRO_PANEL_PATH)
    ensure_required_columns(frame, ["timestamp", *MACRO_X_COLS], label=MACRO_PANEL_PATH.name)

    frame["timestamp"] = parse_naive_timestamp(frame["timestamp"])
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )
    return frame[["timestamp", *MACRO_X_COLS]].copy()


def load_macro_fomc_panel() -> pd.DataFrame:
    """Load the macro panel restricted to the FOMC (rate-decision) 
    dummies.

    The same minute panel is used as for the combined macro family,
    but only the FOMC dummy columns are retained. The 6 FOMC events
    in the sample window split 3-3 across the pre and post regimes
    (none fall inside the Oct 1-10 exclusion window).
    """

    frame = pd.read_csv(MACRO_PANEL_PATH)
    ensure_required_columns(
        frame,
        ["timestamp", *MACRO_FOMC_X_COLS],
        label=MACRO_PANEL_PATH.name,
    )

    frame["timestamp"] = parse_naive_timestamp(frame["timestamp"])
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )
    return frame[["timestamp", *MACRO_FOMC_X_COLS]].copy()


def load_macro_inflation_panel() -> pd.DataFrame:
    """Load the macro panel restricted to the CPI/PPI (inflation) 
    dummies.

    Same minute panel, only the inflation dummy columns are retained.
    The CPI/PPI events fire pre-market at 08:30 ET. The source list has
    9 CPI and 9 PPI releases, but the Oct 10 CPI and Oct 8 PPI releases
    fall inside the Oct 1-10 exclusion window and are dropped at
    sample-load time, leaving 16 CPI/PPI events (8 CPI + 8 PPI, split
    4-4 across the pre and post regimes in each subfamily). This loader
    is the foil to ``load_macro_fomc_panel``.
    """

    frame = pd.read_csv(MACRO_PANEL_PATH)
    ensure_required_columns(
        frame,
        ["timestamp", *MACRO_INFLATION_X_COLS],
        label=MACRO_PANEL_PATH.name,
    )

    frame["timestamp"] = parse_naive_timestamp(frame["timestamp"])
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )
    return frame[["timestamp", *MACRO_INFLATION_X_COLS]].copy()


def load_earnings_panel() -> pd.DataFrame:
    """Load the already-constructed stock-minute earnings urgency 
    panel.

    The sparse earnings panel already contains two timestamp 
    conventions. We use `timestamp_legacy_varx` because it matches 
    the minute-end alignment used elsewhere in the project.
    """

    frame = pd.read_csv(EARNINGS_PANEL_PATH)
    ensure_required_columns(
        frame,
        ["asset", "timestamp_legacy_varx", *EARNINGS_X_COLS],
        label=EARNINGS_PANEL_PATH.name,
    )

    frame["timestamp"] = parse_naive_timestamp(frame["timestamp_legacy_varx"])
    frame = filter_sample_window(
        frame,
        timestamp_col="timestamp",
        start=SAMPLE_START,
        end=SAMPLE_END,
        exclude_windows=ANALYSIS_EXCLUDE_WINDOWS,
    )
    return frame[["asset", "timestamp", *EARNINGS_X_COLS]].copy()

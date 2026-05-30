"""Build the sparse Menkveld-style earnings urgency panel.

See ``02_urgency_construction.md`` (section 1) for the methodology. This
script reads the raw Nasdaq earnings calendar produced by
``03_fetch_nasdaq_earnings.py`` and expands each event into 13 minute-level
``post_ea_k`` variables on the next trading day, scaled by the absolute EPS
surprise over the previous trading day's close.

Inputs:
- ``data_raw/earnings_events.csv``
- ``../01_Data_Pull/data_clean/sp500_tickers.csv``
- ``../01_Data_Pull/data_clean/minute_bars/{ticker}_1m_lit_dark.csv``

Output:
- ``data_clean/earnings_urgency_sparse_panel.csv``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARX_DATA_DIR = PROJECT_ROOT / "03_VARX_Data"
MINUTE_BAR_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "minute_bars"
SP500_PATH = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "sp500_tickers.csv"
RAW_EARNINGS_PATH = VARX_DATA_DIR / "data_raw" / "earnings_events.csv"
OUT_DIR = VARX_DATA_DIR / "data_clean"
OUT_PATH = OUT_DIR / "earnings_urgency_sparse_panel.csv"

SAMPLE_START = pd.Timestamp("2019-06-10")
SAMPLE_END = pd.Timestamp("2020-02-19")
MARKET_OPEN = pd.Timestamp("09:30").time()
NY_TZ = "America/New_York"


def _to_ny_timestamp(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(s, errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(NY_TZ, nonexistent="NaT", ambiguous="NaT")
        else:
            ts = ts.dt.tz_convert(NY_TZ)
        return ts
    return ts.dt.tz_convert(NY_TZ)


def regular_half_hour_block(ts_local: pd.Series) -> pd.Series:
    minutes = ts_local.dt.hour * 60 + ts_local.dt.minute
    start = 9 * 60 + 30
    end = 16 * 60
    out = pd.Series(pd.NA, index=ts_local.index, dtype="Int64")
    mask = (minutes >= start) & (minutes < end)
    out.loc[mask] = ((minutes.loc[mask] - start) // 30 + 1).astype("int64")
    return out


def load_regular_session_minutes(ticker: str) -> pd.DataFrame:
    path = MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "timestamp_utc", "timestamp_local", "trading_date", "ea_block"])

    df = pd.read_csv(path, usecols=["ticker", "timestamp"])
    df = df.rename(columns={"timestamp": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(NY_TZ)
    df["trading_date"] = df["timestamp_local"].dt.normalize()
    df["ea_block"] = regular_half_hour_block(df["timestamp_local"])
    df = df[df["ea_block"].notna()].copy()
    df["ea_block"] = df["ea_block"].astype(int)
    return df[["ticker", "timestamp_utc", "timestamp_local", "trading_date", "ea_block"]].sort_values("timestamp_local")


def load_regular_close_series(ticker: str) -> pd.DataFrame:
    path = MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv"
    if not path.exists():
        return pd.DataFrame(columns=["trading_date", "regular_close"])

    df = pd.read_csv(path, usecols=["timestamp", "lit_close", "dark_close"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(NY_TZ)
    df["trading_date"] = df["timestamp_local"].dt.normalize()
    df["ea_block"] = regular_half_hour_block(df["timestamp_local"])
    df = df[df["ea_block"].notna()].copy()
    df["close_pref"] = df["lit_close"].combine_first(df["dark_close"])
    return (
        df.dropna(subset=["close_pref"])
        .sort_values("timestamp_local")
        .groupby("trading_date", as_index=False)
        .tail(1)[["trading_date", "close_pref"]]
        .rename(columns={"close_pref": "regular_close"})
        .reset_index(drop=True)
    )


def prior_close_date(announce_ts: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if pd.isna(announce_ts):
        return None
    announce_date = announce_ts.normalize()
    if announce_ts.time() < MARKET_OPEN:
        prior = trading_dates[trading_dates < announce_date]
        return prior.max() if len(prior) else None
    if announce_date in set(trading_dates):
        return announce_date
    prior = trading_dates[trading_dates < announce_date]
    return prior.max() if len(prior) else None


def effect_trading_date(announce_ts: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if pd.isna(announce_ts):
        return None
    announce_date = announce_ts.normalize()
    if announce_ts.time() < MARKET_OPEN and announce_date in set(trading_dates):
        return announce_date
    future = trading_dates[trading_dates > announce_date]
    return future.min() if len(future) else None


def normalize_raw_earnings(raw: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "ticker": ["ticker", "symbol"],
        "announce_datetime": [
            "announce_datetime",
            "announcement_datetime",
            "earnings_datetime",
            "reported_date",
            "date",
        ],
        "reported_eps": ["reported_eps", "actual_eps", "eps_actual", "reportedEPS"],
        "expected_eps": ["expected_eps", "estimate_eps", "eps_estimate", "consensus_eps", "expectedEPS"],
    }

    cols = {c.lower(): c for c in raw.columns}
    rename = {}
    for target, aliases in alias_map.items():
        for alias in aliases:
            if alias.lower() in cols:
                rename[cols[alias.lower()]] = target
                break

    out = raw.rename(columns=rename).copy()
    required = ["ticker", "announce_datetime", "reported_eps", "expected_eps"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}. Found columns: {list(raw.columns)}"
        )

    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["announce_datetime"] = _to_ny_timestamp(out["announce_datetime"])
    out["reported_eps"] = pd.to_numeric(out["reported_eps"], errors="coerce")
    out["expected_eps"] = pd.to_numeric(out["expected_eps"], errors="coerce")
    out = out.dropna(subset=["ticker", "announce_datetime", "reported_eps", "expected_eps"]).copy()
    return (
        out[["ticker", "announce_datetime", "reported_eps", "expected_eps"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_sparse_earnings_panel_for_ticker(ticker: str, earnings_t: pd.DataFrame) -> pd.DataFrame:
    minutes = load_regular_session_minutes(ticker)
    closes = load_regular_close_series(ticker)
    if minutes.empty or earnings_t.empty or closes.empty:
        return pd.DataFrame()

    trading_dates = pd.DatetimeIndex(sorted(minutes["trading_date"].dropna().unique()))
    close_map = dict(zip(closes["trading_date"], closes["regular_close"]))

    rows = []
    for event in earnings_t.itertuples(index=False):
        pre_close_date = prior_close_date(event.announce_datetime, trading_dates)
        effect_date = effect_trading_date(event.announce_datetime, trading_dates)
        if pre_close_date is None or effect_date is None:
            continue
        if effect_date.date() < SAMPLE_START.date() or effect_date.date() > SAMPLE_END.date():
            continue
        prior_close = close_map.get(pre_close_date)
        if prior_close is None or pd.isna(prior_close) or prior_close == 0:
            continue

        earnings_surprise = abs(event.reported_eps - event.expected_eps) / float(prior_close)
        if pd.isna(earnings_surprise):
            continue

        effect_minutes = minutes.loc[minutes["trading_date"] == effect_date].copy()
        if effect_minutes.empty:
            continue

        for k in range(1, 14):
            effect_minutes[f"post_ea_{k}"] = np.where(
                effect_minutes["ea_block"] == k, earnings_surprise, 0.0
            )

        effect_minutes["announce_datetime"] = event.announce_datetime
        effect_minutes["pre_close_date"] = pre_close_date
        effect_minutes["effect_date"] = effect_date
        effect_minutes["reported_eps"] = event.reported_eps
        effect_minutes["expected_eps"] = event.expected_eps
        effect_minutes["prior_close"] = prior_close
        effect_minutes["earnings_surprise"] = earnings_surprise
        rows.append(effect_minutes)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "ticker", "timestamp_utc", "timestamp_local", "trading_date", "ea_block",
        "announce_datetime", "pre_close_date", "effect_date", "reported_eps",
        "expected_eps", "prior_close", "earnings_surprise",
    ] + [f"post_ea_{k}" for k in range(1, 14)]
    return out[keep_cols].sort_values(["timestamp_local", "announce_datetime"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_EARNINGS_PATH.exists():
        raise FileNotFoundError(
            f"Raw earnings event file not found: {RAW_EARNINGS_PATH}\n"
            "Run 03_fetch_nasdaq_earnings.py first."
        )

    sp500 = pd.read_csv(SP500_PATH)
    universe_tickers = sorted(sp500["ticker"].dropna().astype(str).unique())

    raw_earnings = pd.read_csv(RAW_EARNINGS_PATH)
    earnings_events = normalize_raw_earnings(raw_earnings)
    earnings_events = earnings_events[earnings_events["ticker"].isin(universe_tickers)].copy()
    print(f"Universe tickers: {len(universe_tickers)}")
    print(f"Normalized earnings events: {len(earnings_events)}")

    all_sparse = []
    for idx, ticker in enumerate(universe_tickers, start=1):
        events_t = earnings_events.loc[earnings_events["ticker"] == ticker].copy()
        if events_t.empty:
            continue
        sparse_t = build_sparse_earnings_panel_for_ticker(ticker, events_t)
        if not sparse_t.empty:
            all_sparse.append(sparse_t)
        if idx % 50 == 0:
            print(f"Processed {idx} / {len(universe_tickers)} tickers")

    if not all_sparse:
        raise RuntimeError(
            "No ticker produced non-empty earnings urgency rows. "
            "Check raw input columns and announcement timestamps."
        )

    earnings_sparse_panel = pd.concat(all_sparse, ignore_index=True)

    earnings_sparse_panel["asset"] = (
        earnings_sparse_panel["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    )
    earnings_sparse_panel["timestamp"] = (
        pd.to_datetime(earnings_sparse_panel["timestamp_local"]).dt.tz_localize(None)
    )
    earnings_sparse_panel["timestamp_legacy_varx"] = (
        pd.to_datetime(earnings_sparse_panel["timestamp_utc"], utc=True)
        - pd.Timedelta(hours=3, minutes=59)
    ).dt.tz_localize(None)

    post_ea_cols = [f"post_ea_{k}" for k in range(1, 14)]
    earnings_sparse_panel[post_ea_cols] = earnings_sparse_panel[post_ea_cols].fillna(0.0)

    ordered_cols = [
        "asset", "ticker", "timestamp", "timestamp_legacy_varx", "timestamp_utc",
        "timestamp_local", "trading_date", "ea_block", "announce_datetime",
        "pre_close_date", "effect_date", "reported_eps", "expected_eps",
        "prior_close", "earnings_surprise",
    ] + post_ea_cols
    earnings_sparse_panel = earnings_sparse_panel[ordered_cols].copy()
    earnings_sparse_panel.to_csv(OUT_PATH, index=False)

    print(f"Saved sparse earnings urgency panel to: {OUT_PATH}")
    print(f"Rows: {len(earnings_sparse_panel)}")


if __name__ == "__main__":
    main()

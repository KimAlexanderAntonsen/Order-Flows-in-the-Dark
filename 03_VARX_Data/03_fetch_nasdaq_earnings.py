from __future__ import annotations

"""
Fetch historical earnings events from the Nasdaq earnings calendar API 
and prepare a ticker-level event file that plugs directly into
04_construct_earnings_urgency.py.

Important limitation:
- the historical daily endpoint often returns `time-not-supplied`
  instead of a before-market / after-hours label. To remain compatible with
  the Menkveld-style next-trading-day construction, this script treats
  missing announcement time as an outside-regular-hours event and sets the
  announcement timestamp to 16:01 New York time on the reported date.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep
from typing import Iterable, Optional

import json
import math
import re
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


NY_TZ = ZoneInfo("America/New_York")
SAMPLE_START = pd.Timestamp("2019-06-10")
SAMPLE_END = pd.Timestamp("2020-02-19")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARX_DATA_DIR = PROJECT_ROOT / "03_VARX_Data"
SP500_PATH = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "sp500_tickers.csv"

RAW_DIR = VARX_DATA_DIR / "data_raw"
RAW_JSON_DIR = RAW_DIR / "earnings_nasdaq_calendar_daily"
PER_TICKER_DIR = RAW_DIR / "earnings_events_by_ticker"
OUT_CLEAN_DIR = VARX_DATA_DIR / "data_clean"

EVENTS_PATH = RAW_DIR / "earnings_events.csv"
FULL_EVENTS_PATH = OUT_CLEAN_DIR / "earnings_events_nasdaq_full.csv"
COVERAGE_PATH = OUT_CLEAN_DIR / "earnings_events_coverage.json"

NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# Current/alternate symbols that should map back to the fixed 2019 S&P 500 universe.
SYMBOL_ALIASES = {
    "META": "FB",
    "COR": "ABC",
    "BKR": "BHGE",
    "BALL": "BLL",
    "LUMN": "CTL",
    "GL": "TMK",
    "J": "JEC",
    "RVTY": "PKI",
    "ELV": "ANTM",
    "PEAK": "HCP",
    "WTW": "WLTW",
    "GEN": "SYMC",
    "BBWI": "LB",
    "FBIN": "FBHS",
    "DINO": "HFC",
    "VTRS": "MYL",
    "FI": "FISV",
    "BRK-B": "BRK-B",
    "BRK.B": "BRK-B",
    "BF-B": "BF-B",
    "BF.B": "BF-B",
}

# Some current symbols correspond to multiple historical share classes in the
# fixed 2019 universe. These need duplication rather than one-to-one remapping.
SYMBOL_MULTI_ALIASES = {
    "FOXA": ["FOX", "FOXA"],
    "NWSA": ["NWS", "NWSA"],
}

# Yahoo fallback symbols for names that Nasdaq's historical calendar does not
# cover well. 
YF_FALLBACK_SYMBOLS = {
    "ADS": ["ADS", "BFH"],
    "DISCA": ["DISCA", "WBD"],
    "DISCK": ["DISCK", "WBD"],
    "FLT": ["FLT", "CPAY"],
    "HCP": ["HCP", "PEAK"],
    "MYL": ["MYL", "VTRS"],
    "RE": ["RE", "EG"],
}


@dataclass
class FetchSummary:
    matched_rows: int
    matched_tickers: int
    raw_rows: int
    raw_symbols: int
    missing_tickers: list[str]
    unmatched_source_symbols: list[str]


def ensure_dirs() -> None:
    for path in [RAW_DIR, RAW_JSON_DIR, PER_TICKER_DIR, OUT_CLEAN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_universe() -> list[str]:
    sp500 = pd.read_csv(SP500_PATH)
    return sorted(sp500["ticker"].dropna().astype(str).str.upper().unique())


def clean_number(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"n/a", "na", "none", "null", "--"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = s.replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        out = float(s)
    except ValueError:
        return None
    return -out if neg else out


def normalize_symbol(symbol: object) -> str:
    s = str(symbol).strip().upper().replace(".", "-").replace("/", "-")
    return SYMBOL_ALIASES.get(s, s)


def normalize_symbol_targets(symbol: object) -> list[str]:
    s = str(symbol).strip().upper().replace(".", "-").replace("/", "-")
    if s in SYMBOL_MULTI_ALIASES:
        return SYMBOL_MULTI_ALIASES[s]
    return [SYMBOL_ALIASES.get(s, s)]


def infer_announce_datetime_local(report_date: pd.Timestamp, raw_time: object) -> tuple[pd.Timestamp, str]:
    raw = "" if raw_time is None else str(raw_time).strip().lower()
    announce_date = pd.Timestamp(report_date.date(), tz=NY_TZ)

    if raw in {"pre-market", "before-market", "premarket", "before market open"}:
        return announce_date + pd.Timedelta(hours=8), "source_pre_market"
    if raw in {"post-market", "after-hours", "after-market", "after market close"}:
        return announce_date + pd.Timedelta(hours=16, minutes=1), "source_post_market"
    if raw in {"time-not-supplied", "", "nan", "none", "null"}:
        return announce_date + pd.Timedelta(hours=16, minutes=1), "assumed_post_market_time_not_supplied"

    # Try generic clock-time strings if Nasdaq ever returns them.
    parsed = pd.to_datetime(f"{report_date.date()} {raw}", errors="coerce")
    if pd.notna(parsed):
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(NY_TZ)
        else:
            parsed = parsed.tz_convert(NY_TZ)
        return parsed, "source_clock_time"

    return announce_date + pd.Timedelta(hours=16, minutes=1), "assumed_post_market_unparsed_time"


def fetch_daily_calendar(session: requests.Session, day: pd.Timestamp, force: bool = False) -> dict:
    date_str = day.strftime("%Y-%m-%d")
    raw_path = RAW_JSON_DIR / f"{date_str}.json"
    if raw_path.exists() and not force:
        with raw_path.open() as f:
            return json.load(f)

    url = NASDAQ_URL.format(date=date_str)
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    with raw_path.open("w") as f:
        json.dump(data, f)
    return data


def extract_rows(payload: dict, report_date: pd.Timestamp) -> list[dict]:
    rows = (((payload or {}).get("data") or {}).get("rows")) or []
    out = []
    for row in rows:
        source_symbol = str(row.get("symbol", "")).strip().upper()
        announce_dt, timing_rule = infer_announce_datetime_local(report_date, row.get("time"))
        for normalized_symbol in normalize_symbol_targets(source_symbol):
            out.append(
                {
                    "announce_date": report_date.date().isoformat(),
                    "announce_datetime": announce_dt.isoformat(),
                    "source_symbol": source_symbol,
                    "ticker": normalized_symbol,
                    "company_name": row.get("name"),
                    "source_time_label": row.get("time"),
                    "timing_rule": timing_rule,
                    "reported_eps": clean_number(row.get("eps")),
                    "expected_eps": clean_number(row.get("epsForecast")),
                    "surprise_pct": clean_number(row.get("surprise")),
                    "fiscal_quarter_ending": row.get("fiscalQuarterEnding"),
                    "n_estimates": clean_number(row.get("noOfEsts")),
                    "source": "nasdaq_calendar_daily",
                }
            )
    return out


def yahoo_candidates_for_ticker(ticker: str) -> list[str]:
    return YF_FALLBACK_SYMBOLS.get(ticker, [ticker])


def fetch_yfinance_fallback_rows(target_ticker: str) -> list[dict]:
    sample_start = pd.Timestamp(SAMPLE_START, tz=NY_TZ)
    sample_end = pd.Timestamp(SAMPLE_END, tz=NY_TZ) + pd.Timedelta(days=1)

    for candidate in yahoo_candidates_for_ticker(target_ticker):
        try:
            raw = yf.Ticker(candidate).get_earnings_dates(limit=100)
        except Exception:
            raw = None

        if raw is None or raw.empty:
            continue

        df = raw.reset_index().copy()
        date_col = df.columns[0]
        df = df.rename(
            columns={
                date_col: "announce_datetime",
                "EPS Estimate": "expected_eps",
                "Reported EPS": "reported_eps",
                "Surprise(%)": "surprise_pct",
            }
        )
        df["announce_datetime"] = pd.to_datetime(df["announce_datetime"], errors="coerce")
        if df["announce_datetime"].dt.tz is None:
            df["announce_datetime"] = df["announce_datetime"].dt.tz_localize(NY_TZ)
        else:
            df["announce_datetime"] = df["announce_datetime"].dt.tz_convert(NY_TZ)

        df["reported_eps"] = pd.to_numeric(df["reported_eps"], errors="coerce")
        df["expected_eps"] = pd.to_numeric(df["expected_eps"], errors="coerce")
        df["surprise_pct"] = pd.to_numeric(df["surprise_pct"], errors="coerce")

        df = df[
            (df["announce_datetime"] >= sample_start)
            & (df["announce_datetime"] < sample_end)
        ].dropna(subset=["announce_datetime", "reported_eps", "expected_eps"])

        if df.empty:
            continue

        rows: list[dict] = []
        for _, row in df.iterrows():
            announce_dt = pd.Timestamp(row["announce_datetime"]).tz_convert(NY_TZ)
            rows.append(
                {
                    "announce_date": announce_dt.date().isoformat(),
                    "announce_datetime": announce_dt.isoformat(),
                    "source_symbol": candidate,
                    "ticker": target_ticker,
                    "company_name": None,
                    "source_time_label": announce_dt.strftime("%H:%M"),
                    "timing_rule": "source_yfinance_timestamp",
                    "reported_eps": float(row["reported_eps"]),
                    "expected_eps": float(row["expected_eps"]),
                    "surprise_pct": float(row["surprise_pct"]) if pd.notna(row["surprise_pct"]) else None,
                    "fiscal_quarter_ending": None,
                    "n_estimates": None,
                    "source": "yfinance_earnings_dates",
                }
            )
        return rows

    return []


def supplement_with_yfinance(raw_df: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = set(load_universe())
    missing = sorted(universe - set(events["ticker"].unique()))
    if not missing:
        return raw_df, events

    fallback_rows: list[dict] = []
    for idx, ticker in enumerate(missing, start=1):
        rows = fetch_yfinance_fallback_rows(ticker)
        fallback_rows.extend(rows)
        if idx % 10 == 0:
            print(f"Checked Yahoo fallback for {idx} / {len(missing)} missing tickers", flush=True)

    if not fallback_rows:
        return raw_df, events

    fallback_df = pd.DataFrame(fallback_rows)
    fallback_df["announce_datetime"] = pd.to_datetime(fallback_df["announce_datetime"], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    fallback_df = fallback_df.sort_values(["announce_datetime", "ticker"]).reset_index(drop=True)
    fallback_df = fallback_df.reindex(columns=raw_df.columns)

    combined_raw = pd.concat([raw_df, fallback_df], ignore_index=True, sort=False)
    combined_events = pd.concat([events, fallback_df], ignore_index=True, sort=False)
    combined_events = combined_events.sort_values(["ticker", "announce_datetime", "source"]).drop_duplicates(
        subset=["ticker", "announce_date"],
        keep="first",
    ).reset_index(drop=True)
    return combined_raw, combined_events


def build_events(force: bool = False, sleep_seconds: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []
    days = pd.date_range(SAMPLE_START, SAMPLE_END, freq="D")
    session = requests.Session()
    for idx, day in enumerate(days, start=1):
        payload = fetch_daily_calendar(session, day, force=force)
        all_rows.extend(extract_rows(payload, day))
        if idx % 25 == 0:
            print(f"Fetched {idx} / {len(days)} dates", flush=True)
        sleep(sleep_seconds)

    raw_df = pd.DataFrame(all_rows)
    if raw_df.empty:
        return raw_df, raw_df

    raw_df["announce_datetime"] = pd.to_datetime(raw_df["announce_datetime"], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    raw_df["reported_eps"] = pd.to_numeric(raw_df["reported_eps"], errors="coerce")
    raw_df["expected_eps"] = pd.to_numeric(raw_df["expected_eps"], errors="coerce")
    raw_df["surprise_pct"] = pd.to_numeric(raw_df["surprise_pct"], errors="coerce")
    raw_df["n_estimates"] = pd.to_numeric(raw_df["n_estimates"], errors="coerce")
    raw_df = raw_df.dropna(subset=["ticker", "announce_datetime", "reported_eps", "expected_eps"]).copy()
    raw_df = raw_df.sort_values(["announce_datetime", "ticker"]).reset_index(drop=True)

    universe = set(load_universe())
    events = raw_df[raw_df["ticker"].isin(universe)].copy()
    events = events.drop_duplicates(subset=["ticker", "announce_date"], keep="first").reset_index(drop=True)
    raw_df, events = supplement_with_yfinance(raw_df, events)
    return raw_df, events


def save_outputs(raw_df: pd.DataFrame, events: pd.DataFrame) -> FetchSummary:
    raw_df.to_csv(FULL_EVENTS_PATH, index=False)
    events.to_csv(EVENTS_PATH, index=False)

    for old_file in PER_TICKER_DIR.glob("*.csv"):
        old_file.unlink()

    for ticker, grp in events.groupby("ticker", sort=True):
        grp.to_csv(PER_TICKER_DIR / f"{ticker}.csv", index=False)

    universe = set(load_universe())
    matched_tickers = sorted(events["ticker"].unique()) if not events.empty else []
    missing_tickers = sorted(universe - set(matched_tickers))
    unmatched_source_symbols = sorted(set(raw_df["source_symbol"]) - universe - set(SYMBOL_ALIASES.keys())) if not raw_df.empty else []

    summary = FetchSummary(
        matched_rows=int(len(events)),
        matched_tickers=int(len(matched_tickers)),
        raw_rows=int(len(raw_df)),
        raw_symbols=int(raw_df["source_symbol"].nunique() if not raw_df.empty else 0),
        missing_tickers=missing_tickers,
        unmatched_source_symbols=unmatched_source_symbols,
    )

    with COVERAGE_PATH.open("w") as f:
        json.dump(
            {
                "matched_rows": summary.matched_rows,
                "matched_tickers": summary.matched_tickers,
                "raw_rows": summary.raw_rows,
                "raw_symbols": summary.raw_symbols,
                "missing_tickers": summary.missing_tickers,
                "unmatched_source_symbols": summary.unmatched_source_symbols,
            },
            f,
            indent=2,
        )

    return summary


def main(force: bool = False) -> None:
    ensure_dirs()
    raw_df, events = build_events(force=force)
    summary = save_outputs(raw_df, events)

    print("\nSaved:")
    print(" -", FULL_EVENTS_PATH)
    print(" -", EVENTS_PATH)
    print(" -", PER_TICKER_DIR)
    print(" -", COVERAGE_PATH)
    print("\nSummary:")
    print("Raw rows:", summary.raw_rows)
    print("Matched event rows:", summary.matched_rows)
    print("Matched tickers:", summary.matched_tickers)
    print("Missing universe tickers:", len(summary.missing_tickers))
    if summary.missing_tickers:
        print("First 25 missing tickers:", summary.missing_tickers[:25])


if __name__ == "__main__":
    main(force=False)

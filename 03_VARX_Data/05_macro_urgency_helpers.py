"""Helpers for the focused macro urgency panel.

Holds the curated CPI/PPI/FOMC release list inline and provides the
build / template / panel utilities consumed by the macro construction
script. See ``02_urgency_construction.md`` (section 2) for the
methodology.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARX_DATA_DIR = PROJECT_ROOT / "03_VARX_Data"
MINUTE_BAR_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "minute_bars"

CLEAN_EVENTS_PATH = VARX_DATA_DIR / "data_clean" / "macro_news_events_clean.csv"
MACRO_PANEL_PATH = VARX_DATA_DIR / "data_clean" / "macro_news_minute_panel.csv"

SAMPLE_START = pd.Timestamp("2019-06-10", tz="America/New_York")
SAMPLE_END = pd.Timestamp("2020-02-19 23:59:59", tz="America/New_York")
NY_TZ = "America/New_York"
MARKET_DATA_END_OF_MINUTE_SHIFT = pd.Timedelta(minutes=1)
LEGACY_VARX_SHIFT = pd.Timedelta(hours=3, minutes=59)

COMBINED_DUMMY_COLS = [
    "pre_news_1min",
    "post_news_0min",
    "post_news_1min",
    "post_news_2min",
    "post_news_3min",
    "post_news_4min",
]

INFLATION_DUMMY_COLS = [
    "pre_inflation_1min",
    "post_inflation_0min",
    "post_inflation_1min",
    "post_inflation_2min",
    "post_inflation_3min",
    "post_inflation_4min",
]

RATE_DUMMY_COLS = [
    "pre_rate_1min",
    "post_rate_0min",
    "post_rate_1min",
    "post_rate_2min",
    "post_rate_3min",
    "post_rate_4min",
]

WINDOW_OFFSETS = {
    "pre": -1,
    "post_0": 0,
    "post_1": 1,
    "post_2": 2,
    "post_3": 3,
    "post_4": 4,
}


_BLS_ARCHIVE = "https://www.bls.gov/news.release/archives"
_FED_PRESS = "https://www.federalreserve.gov/newsevents/pressreleases"


def _bls_url(prefix: str, date: str) -> str:
    return f"{_BLS_ARCHIVE}/{prefix}_{pd.Timestamp(date).strftime('%m%d%Y')}.htm"


def _fed_url(date: str) -> str:
    return f"{_FED_PRESS}/monetary{pd.Timestamp(date).strftime('%Y%m%d')}a.htm"


CPI_DATES = [
    "2019-06-12", "2019-07-11", "2019-08-13", "2019-09-12", "2019-10-10",
    "2019-11-13", "2019-12-11", "2020-01-14", "2020-02-13",
]

PPI_DATES = [
    "2019-06-11", "2019-07-12", "2019-08-09", "2019-09-11", "2019-10-08",
    "2019-11-14", "2019-12-12", "2020-01-15", "2020-02-19",
]

FOMC_DATES = [
    "2019-06-19", "2019-07-31", "2019-09-18",
    "2019-10-30", "2019-12-11", "2020-01-29",
]


def build_macro_news_events() -> pd.DataFrame:
    """Return the curated CPI / PPI / FOMC release list as a DataFrame."""

    rows: list[dict[str, str]] = []

    for date in CPI_DATES:
        rows.append(
            {
                "event_group": "inflation",
                "event_type": "cpi",
                "release_name": "Consumer Price Index (CPI)",
                "announce_date": date,
                "announce_time": "08:30",
                "official_source": "BLS",
                "source_url": _bls_url("cpi", date),
            }
        )

    for date in PPI_DATES:
        rows.append(
            {
                "event_group": "inflation",
                "event_type": "ppi",
                "release_name": "Producer Price Index (PPI)",
                "announce_date": date,
                "announce_time": "08:30",
                "official_source": "BLS",
                "source_url": _bls_url("ppi", date),
            }
        )

    for date in FOMC_DATES:
        rows.append(
            {
                "event_group": "rate",
                "event_type": "fomc_rate_decision",
                "release_name": "FOMC statement and rate decision",
                "announce_date": date,
                "announce_time": "14:00",
                "official_source": "Federal Reserve",
                "source_url": _fed_url(date),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["announce_date", "announce_time", "event_group", "event_type"])
        .reset_index(drop=True)
    )


def prepare_macro_events(raw: pd.DataFrame) -> pd.DataFrame:
    events = raw.copy()
    events["announce_datetime"] = pd.to_datetime(
        events["announce_date"].astype(str) + " " + events["announce_time"].astype(str),
        errors="coerce",
    ).dt.tz_localize(NY_TZ, nonexistent="NaT", ambiguous="NaT")
    events = events.dropna(subset=["announce_datetime"]).copy()
    events = events.loc[
        (events["announce_datetime"] >= SAMPLE_START) & (events["announce_datetime"] <= SAMPLE_END)
    ].copy()
    events["announce_minute"] = events["announce_datetime"].dt.floor("min")
    events["anchor_timestamp_local"] = events["announce_minute"] + MARKET_DATA_END_OF_MINUTE_SHIFT
    events["anchor_timestamp"] = events["anchor_timestamp_local"].dt.tz_localize(None)
    events["anchor_timestamp_legacy_varx"] = (
        events["announce_datetime"].dt.tz_convert("UTC") - LEGACY_VARX_SHIFT
    ).dt.tz_localize(None)
    events = events.sort_values("announce_datetime").reset_index(drop=True)
    return events[
        [
            "event_group",
            "event_type",
            "release_name",
            "announce_date",
            "announce_time",
            "announce_datetime",
            "announce_minute",
            "anchor_timestamp_local",
            "anchor_timestamp",
            "anchor_timestamp_legacy_varx",
            "official_source",
            "source_url",
        ]
    ]


def load_template_minutes(ticker: str = "AAPL") -> pd.DataFrame:
    path = MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv"
    if not path.exists():
        raise FileNotFoundError(f"Template minute-bar file not found: {path}")

    raw = pd.read_csv(path, usecols=["timestamp"]).copy()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["timestamp_utc"]).copy()
    trading_dates = (
        raw["timestamp_utc"].dt.tz_convert(NY_TZ).dt.normalize().drop_duplicates().sort_values().tolist()
    )

    session_minutes = []
    for trading_date in trading_dates:
        session_minutes.append(
            pd.DataFrame(
                {
                    "timestamp_local": pd.date_range(
                        start=trading_date + pd.Timedelta(hours=4, minutes=1),
                        end=trading_date + pd.Timedelta(hours=20),
                        freq="min",
                        tz=NY_TZ,
                    )
                }
            )
        )

    template = pd.concat(session_minutes, ignore_index=True)
    template["timestamp_utc"] = (template["timestamp_local"] - MARKET_DATA_END_OF_MINUTE_SHIFT).dt.tz_convert("UTC")
    template["timestamp"] = template["timestamp_local"].dt.tz_localize(None)
    template["timestamp_legacy_varx"] = (template["timestamp_utc"] - LEGACY_VARX_SHIFT).dt.tz_localize(None)
    template["trading_date"] = template["timestamp_local"].dt.normalize()
    template = template.sort_values("timestamp").reset_index(drop=True)
    return template[["timestamp_utc", "timestamp_local", "timestamp", "timestamp_legacy_varx", "trading_date"]]


def _mark_event_windows(panel: pd.DataFrame, event_times: pd.Series, columns: list[str]) -> None:
    if event_times.empty:
        for col in columns:
            panel[col] = 0
        return

    event_set = set(event_times)
    panel[columns[0]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["pre"]) for ts in event_set}).astype(int)
    panel[columns[1]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["post_0"]) for ts in event_set}).astype(int)
    panel[columns[2]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["post_1"]) for ts in event_set}).astype(int)
    panel[columns[3]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["post_2"]) for ts in event_set}).astype(int)
    panel[columns[4]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["post_3"]) for ts in event_set}).astype(int)
    panel[columns[5]] = panel["timestamp"].isin({ts + pd.Timedelta(minutes=WINDOW_OFFSETS["post_4"]) for ts in event_set}).astype(int)


def build_macro_panel(events: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    panel = template.copy()

    inflation_events = events.loc[events["event_group"] == "inflation", "anchor_timestamp"]
    rate_events = events.loc[events["event_group"] == "rate", "anchor_timestamp"]
    all_events = events["anchor_timestamp"]

    _mark_event_windows(panel, all_events, COMBINED_DUMMY_COLS)
    _mark_event_windows(panel, inflation_events, INFLATION_DUMMY_COLS)
    _mark_event_windows(panel, rate_events, RATE_DUMMY_COLS)

    summary = (
        events.groupby("anchor_timestamp", as_index=False)
        .agg(
            n_events=("release_name", "size"),
            event_groups=("event_group", lambda s: " | ".join(sorted(pd.Series(s).astype(str).unique()))),
            release_names=("release_name", lambda s: " | ".join(sorted(pd.Series(s).astype(str).unique()))),
        )
        .sort_values("anchor_timestamp")
    )

    panel = panel.merge(summary, how="left", left_on="timestamp", right_on="anchor_timestamp")
    panel = panel.drop(columns=["anchor_timestamp"])
    panel["n_events"] = panel["n_events"].fillna(0).astype(int)
    panel["event_groups"] = panel["event_groups"].fillna("")
    panel["release_names"] = panel["release_names"].fillna("")
    return panel


def save_outputs(events: pd.DataFrame, panel: pd.DataFrame) -> None:
    CLEAN_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MACRO_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(CLEAN_EVENTS_PATH, index=False)
    panel.to_csv(MACRO_PANEL_PATH, index=False)


def build_and_save_outputs(template_ticker: str = "AAPL") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = build_macro_news_events()
    events = prepare_macro_events(raw)
    template = load_template_minutes(template_ticker)
    panel = build_macro_panel(events, template)
    save_outputs(events, panel)
    return events, panel

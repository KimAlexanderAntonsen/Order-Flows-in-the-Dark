"""Shared utility helpers for the beta VARX build."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Raise a helpful error if a required input column is missing."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_ticker(ticker: str, rename_map: dict[str, str]) -> str:
    """Map external ticker names into the naming used by the minute-bar files."""

    return rename_map.get(str(ticker).upper(), str(ticker).upper())


def safe_log(values: pd.Series | np.ndarray, floor: float) -> np.ndarray:
    """Compute a numerically safe logarithm.

    Many market variables are zero in some minutes. We therefore clip 
    them away from zero before taking logs.
    """

    array = np.asarray(values, dtype=float)
    return np.log(np.clip(array, floor, None))


def minute_bar_utc_to_local_minute_end(timestamp: pd.Series) -> pd.Series:
    """Convert Massive timestamps into naive New York minute-end 
    timestamps.

    The raw minute-bar files store timestamps in UTC. We shift them 
    forward by one minute after converting to America/New_York, so 
    that each row is interpreted as the end of the one-minute 
    interval.
    """

    return (
        pd.to_datetime(timestamp, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        + pd.Timedelta(minutes=1)
    )


def parse_naive_timestamp(timestamp: pd.Series) -> pd.Series:
    """Parse timestamps into naive pandas datetimes."""

    return pd.to_datetime(timestamp).dt.tz_localize(None)


def filter_sample_window(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    start: str,
    end: str,
    exclude_windows: Iterable[tuple[str, str]] = (),
) -> pd.DataFrame:
    """Restrict a DataFrame to the analysis sample window.

    The VARX pipeline also allows explicit exclusion windows inside
    the sample. We use this to remove the October 1 to October 10,
    2019 exclusion window from the analysis sample.
    """

    mask = (df[timestamp_col] >= pd.Timestamp(start)) & (df[timestamp_col] <= pd.Timestamp(end))
    for window_start, window_end in exclude_windows:
        mask &= ~(
            (df[timestamp_col] >= pd.Timestamp(window_start))
            & (df[timestamp_col] <= pd.Timestamp(window_end))
        )
    return df.loc[mask].copy()


def filter_session(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    """Restrict a DataFrame to a given intraday session."""

    start_time = pd.Timestamp(session_start).time()
    end_time = pd.Timestamp(session_end).time()
    mask = (df[timestamp_col].dt.time >= start_time) & (df[timestamp_col].dt.time <= end_time)
    return df.loc[mask].copy()

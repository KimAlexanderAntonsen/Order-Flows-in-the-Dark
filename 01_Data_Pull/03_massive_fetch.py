"""Pull and aggregate one-minute lit and dark trade data from Massive.

This script downloads raw trade prints and converts them into the 
minute-bar files used downstream. For stock i and minute t, 
the venue-specific minute volumes are

    V_dark(i,t) = sum_{k in D(i,t)} q(i,k)
    V_lit(i,t)  = sum_{k in L(i,t)} q(i,k)

where q(i,k) is trade size and the dark set D(i,t) follows the project 
rule that a trade is dark when `exchange == 4` and `trf_id` is 
present.

Within each minute and venue segment, realized variance is

    RV_seg(i,t) = sum_{k in seg,t} (Delta log p(i,k))^2

The script writes one CSV per stock to `data_clean/minute_bars/`.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd
from massive import RESTClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "01_Data_Pull" / "data_raw" / "massive_trades"
DATA_CLEAN = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "minute_bars"
ASSET_FILE = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "sp500_tickers.csv"

# Massive expects dot notation for the two class-share tickers below.
MASSIVE_SYMBOL_MAP = {"BF-B": "BF.B", "BRK-B": "BRK.B"}


def build_client() -> RESTClient:
    """Build the Massive client from the REST_key environment variable."""

    api_key = os.getenv("REST_key")
    if not api_key:
        raise ValueError(
            "REST_key is not set. Add it to your environment before running this script."
        )
    return RESTClient(api_key)


def date_range_strings(start_date: str, end_date: str) -> list[str]:
    """Return the inclusive daily date grid as YYYY-MM-DD strings."""

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return dates.strftime("%Y-%m-%d").tolist()


def pull_trades(
    client: RESTClient,
    ticker: str,
    date: str,
    limit: int = 50_000,
    max_retries: int = 3,
) -> list:
    """Pull all trades for one ticker-day with simple retry logic."""

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return list(client.list_trades(ticker, date, limit=limit))
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
    raise last_error


def trades_to_minute_bars_with_rv(
    trades: list,
    use_timestamp: str = "participant_timestamp",
) -> pd.DataFrame:
    """Aggregate raw trade prints into one-minute lit/dark bars."""

    if not trades:
        return pd.DataFrame(columns=["timestamp"])

    rows = []
    for trade in trades:
        timestamp_ns = getattr(trade, use_timestamp, None)
        if timestamp_ns is None:
            continue
        rows.append(
            {
                "ts_ns": timestamp_ns,
                "price": float(getattr(trade, "price", np.nan)),
                "size": int(getattr(trade, "size", 0)),
                "exchange": getattr(trade, "exchange", None),
                "trf_id": getattr(trade, "trf_id", None),
                "sequence_number": getattr(trade, "sequence_number", None),
            }
        )

    frame = pd.DataFrame(rows).dropna(subset=["ts_ns", "price"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp"])

    frame["ts"] = pd.to_datetime(frame["ts_ns"], unit="ns", utc=True)
    frame["timestamp"] = frame["ts"].dt.floor("min")

    # Dark-trade rule.
    frame["is_dark"] = (frame["exchange"] == 4) & frame["trf_id"].notna()
    frame["seg"] = np.where(frame["is_dark"], "dark", "lit")

    sort_cols = ["timestamp", "seg", "ts_ns"]
    if frame["sequence_number"].notna().any():
        sort_cols.append("sequence_number")
    frame = frame.sort_values(sort_cols, kind="mergesort")

    frame["px_sz"] = frame["price"] * frame["size"]
    frame["log_price"] = np.log(frame["price"])
    frame["log_ret"] = frame.groupby(["timestamp", "seg"])["log_price"].diff()
    frame["rv_component"] = frame["log_ret"] ** 2

    grouped = frame.groupby(["timestamp", "seg"], sort=True)
    aggregated = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        vwap_num=("px_sz", "sum"),
        vwap_den=("size", "sum"),
        realized_variance=("rv_component", "sum"),
        n_trades=("price", "count"),
    )

    aggregated["vwap"] = aggregated["vwap_num"] / aggregated["vwap_den"]
    aggregated["realized_volatility"] = np.sqrt(aggregated["realized_variance"])
    aggregated = aggregated.drop(columns=["vwap_num", "vwap_den"])

    wide = aggregated.unstack("seg")
    wide.columns = [f"{seg}_{field}" for field, seg in wide.columns]
    return wide.reset_index()


def load_assets_to_fetch() -> list[str]:
    """Load the canonical universe and skip files that already exist."""

    assets = [
        MASSIVE_SYMBOL_MAP.get(ticker, ticker)
        for ticker in pd.read_csv(ASSET_FILE)["ticker"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
    ]

    existing_assets = {
        path.name.replace("_1m_lit_dark.csv", "")
        for path in DATA_CLEAN.glob("*_1m_lit_dark.csv")
    }
    missing_assets = [asset for asset in assets if asset not in existing_assets]

    print(f"Loaded {len(assets)} assets from {ASSET_FILE.name}")
    print(f"Existing minute-bar files: {len(existing_assets)}")
    print(f"Missing assets to fetch now: {len(missing_assets)}")
    print(f"First 20 missing assets: {missing_assets[:20]}")

    return missing_assets


def retry_call(fn, *args, retries: int = 4, base_sleep: float = 0.75, **kwargs):
    """Retry transient network work with exponential backoff and jitter."""

    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network/API dependent
            last_error = exc
            if attempt == retries:
                raise
            sleep_seconds = base_sleep * (2**attempt) * (1.0 + random.random() * 0.25)
            time.sleep(sleep_seconds)
    raise last_error


def fetch_and_build(
    client: RESTClient,
    asset: str,
    date: str,
    limit: int,
) -> tuple[str, str, pd.DataFrame, int]:
    """Pull one ticker-day and return its minute bars."""

    trades = retry_call(pull_trades, client, asset, date, limit=limit)
    bars = trades_to_minute_bars_with_rv(trades, use_timestamp="participant_timestamp")
    return asset, date, bars, len(trades)


def massive_pull_parallel_safe(
    client: RESTClient,
    assets: list[str],
    dates: list[str],
    limit: int = 50_000,
    max_workers: int = 12,
    max_in_flight: int | None = None,
    watchdog_seconds: int = 120,
) -> Path:
    """Parallel ticker-day pull with safe scheduling and append-only writes."""

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    if max_in_flight is None:
        max_in_flight = max_workers * 2

    first_write = {
        asset: not (DATA_CLEAN / f"{asset}_1m_lit_dark.csv").exists()
        for asset in assets
    }

    def task_iter():
        for asset in assets:
            for date in dates:
                yield asset, date

    iterator = task_iter()
    futures = set()
    metadata: dict = {}
    log_messages: list[str] = []
    total = len(assets) * len(dates)
    completed_count = 0
    last_progress = time.time()

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        try:
            asset, date = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(fetch_and_build, client, asset, date, limit)
        futures.add(future)
        metadata[future] = (asset, date)
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_in_flight):
            if not submit_one(executor):
                break

        while futures:
            completed, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            if not completed:
                if time.time() - last_progress > watchdog_seconds:
                    message = (
                        f"No completed tasks for {watchdog_seconds}s. "
                        f"Still pending={len(futures)}."
                    )
                    print(message)
                    log_messages.append(message)
                    last_progress = time.time()
                continue

            for future in completed:
                futures.remove(future)
                asset, date = metadata.pop(future)
                completed_count += 1

                try:
                    _, _, bars, n_trades = future.result()
                    if bars is not None and not bars.empty:
                        bars.insert(0, "date", date)
                        bars.insert(0, "ticker", asset)
                        out_path = DATA_CLEAN / f"{asset}_1m_lit_dark.csv"
                        bars.to_csv(
                            out_path,
                            index=False,
                            mode="w" if first_write[asset] else "a",
                            header=first_write[asset],
                        )
                        first_write[asset] = False
                    message = (
                        f"Completed {asset} {date} "
                        f"(trades={n_trades}, minutes={0 if bars is None else len(bars)}) "
                        f"[{completed_count}/{total}]"
                    )
                except Exception as exc:  # pragma: no cover - network/API dependent
                    message = f"Error {asset} {date}: {repr(exc)} [{completed_count}/{total}]"

                print(message)
                log_messages.append(message)
                last_progress = time.time()
                submit_one(executor)

    log_path = DATA_CLEAN / "massive_pull_log.txt"
    log_path.write_text("\n".join(log_messages))
    return log_path


def main() -> None:
    """Run the full minute-bar pull for any missing universe tickers."""

    client = build_client()
    assets = load_assets_to_fetch()
    if not assets:
        print("All canonical-universe minute-bar files already exist.")
        return

    dates = date_range_strings("2019-06-10", "2020-02-19")
    log_path = massive_pull_parallel_safe(
        client=client,
        assets=assets,
        dates=dates,
        limit=50_000,
        max_workers=12,
        max_in_flight=24,
        watchdog_seconds=120,
    )
    print(f"Saved pull log: {log_path}")


if __name__ == "__main__":
    main()

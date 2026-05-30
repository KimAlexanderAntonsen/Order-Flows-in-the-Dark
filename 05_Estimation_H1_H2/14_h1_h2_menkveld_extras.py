"""Menkveld-style descriptive figures and tables for the H1/H2 layer.

This module creates:

1. A market-wide aggregate dark-share time series (daily, across the 
   S&P 500 universe) that visualizes the regime break in October 2019.
2. Menkveld-style descriptive tables for the H1/H2 universe covering 
   the pre- and post-regimes separately: average venue shares, 
   log-volume moments, and realized variance moments.

Both products are built from the endogenous minute panel used by the
benchmark VARX so the numbers line up one-to-one with the estimation 
sample.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterator, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parent
BETA_DIR = ESTIMATION_DIR.parents[0] / "04_VARX"
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))
if str(BETA_DIR) not in sys.path:
    sys.path.insert(0, str(BETA_DIR))


_config_beta = importlib.import_module("02_beta_varx_config")
_utils = importlib.import_module("03_beta_varx_utils")
_panel = importlib.import_module("05_beta_varx_panel")
_data = importlib.import_module("04_beta_varx_data")
_local = importlib.import_module("01_estimation_config")
_presentation = importlib.import_module("05_h1_h2_presentation")


TABLE_DIR = _presentation.TABLE_DIR
FIGURE_DIR = _presentation.FIGURE_DIR

PRE_WINDOW = _local.PRE_WINDOW
POST_WINDOW = _local.POST_WINDOW
TRANSITION_START = pd.Timestamp("2019-10-01")
TRANSITION_END = pd.Timestamp("2019-10-10")

REGULAR_SESSION = _config_beta.REGULAR_SESSION

iter_endogenous_panel_pieces = _panel.iter_endogenous_panel_pieces
load_sp500_universe = _data.load_sp500_universe


def _daily_volume_from_pieces(pieces: Iterator[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Sum dark/lit volumes by day across every stock in ``pieces``.

    Returns the daily frame and the count of distinct stocks contributing
    any minute-level observation.
    """

    running: dict[pd.Timestamp, dict[str, float]] = {}
    stocks: set[str] = set()
    for piece in pieces:
        if piece.empty:
            continue
        if "asset" in piece.columns:
            stocks.update(piece["asset"].astype(str).unique().tolist())
        day = pd.to_datetime(piece["timestamp"]).dt.normalize()
        dark = pd.to_numeric(piece["dark_volume_t"], errors="coerce").fillna(0.0)
        lit = pd.to_numeric(piece["lit_volume_t"], errors="coerce").fillna(0.0)
        grouped = pd.DataFrame({"day": day, "dark": dark, "lit": lit}).groupby("day").sum()
        for day_key, row in grouped.iterrows():
            entry = running.setdefault(day_key, {"dark": 0.0, "lit": 0.0})
            entry["dark"] += float(row["dark"])
            entry["lit"] += float(row["lit"])

    if not running:
        return pd.DataFrame(columns=["day", "dark_volume", "lit_volume", "dark_share"]), len(stocks)

    rows = sorted(running.items(), key=lambda kv: kv[0])
    frame = pd.DataFrame(
        {
            "day": [day for day, _ in rows],
            "dark_volume": [entry["dark"] for _, entry in rows],
            "lit_volume": [entry["lit"] for _, entry in rows],
        }
    )
    totals = frame["dark_volume"] + frame["lit_volume"]
    frame["dark_share"] = np.where(totals > 0.0, frame["dark_volume"] / totals, np.nan)
    return frame, len(stocks)


def _iter_minute_bars_full_sample(tickers: Iterable[str]) -> Iterator[pd.DataFrame]:
    """Yield one stock's minute bar at a time without the exclusion filter.

    This mirrors ``iter_endogenous_panel_pieces`` but skips the Oct 1-10
    transition exclusion so the descriptive figure can show every trading
    day in the sample. The estimation pipeline is unaffected.
    """

    for ticker in tickers:
        path = _config_beta.MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["timestamp", "dark_volume", "lit_volume"])
        frame["timestamp"] = _utils.minute_bar_utc_to_local_minute_end(frame["timestamp"])
        frame = _utils.filter_sample_window(
            frame,
            timestamp_col="timestamp",
            start=_config_beta.SAMPLE_START,
            end=_config_beta.SAMPLE_END,
            exclude_windows=(),
        )
        frame = _utils.filter_session(
            frame,
            timestamp_col="timestamp",
            session_start=REGULAR_SESSION.start,
            session_end=REGULAR_SESSION.end,
        )
        if frame.empty:
            continue
        frame = frame.rename(
            columns={"dark_volume": "dark_volume_t", "lit_volume": "lit_volume_t"}
        )
        frame["asset"] = ticker
        yield frame


def build_daily_dark_share_series() -> tuple[pd.DataFrame, int]:
    """Aggregate dark-share across the universe at daily frequency.

    Uses a figure-local minute-bar iterator that bypasses the analysis
    exclusion window so the Oct 1-10 transition days appear in the
    series alongside the pre and post windows.
    """

    tickers = load_sp500_universe()
    pieces = _iter_minute_bars_full_sample(tickers)
    frame, n_stocks = _daily_volume_from_pieces(pieces)

    pre_mask = (frame["day"] >= pd.Timestamp(PRE_WINDOW.start).normalize()) & (
        frame["day"] <= pd.Timestamp(PRE_WINDOW.end).normalize()
    )
    post_mask = (frame["day"] >= pd.Timestamp(POST_WINDOW.start).normalize()) & (
        frame["day"] <= pd.Timestamp(POST_WINDOW.end).normalize()
    )
    transition_mask = (frame["day"] >= TRANSITION_START) & (frame["day"] <= TRANSITION_END)
    frame["regime"] = np.where(
        pre_mask,
        "pre",
        np.where(post_mask, "post", np.where(transition_mask, "transition", "out_of_sample")),
    )
    return frame, n_stocks


def save_daily_dark_share_table(frame: pd.DataFrame) -> Path:
    """Save the daily dark-share series as a CSV companion to the figure."""

    path = TABLE_DIR / "h1h2_daily_dark_share_series.csv"
    frame.to_csv(path, index=False)
    return path


def plot_daily_dark_share_series(frame: pd.DataFrame, n_stocks: int | None = None) -> list[Path]:
    """Plot the universe-wide daily dark-share series with regime shading."""

    _presentation.apply_menkveld_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)

    pre = frame[frame["regime"] == "pre"]
    post = frame[frame["regime"] == "post"]
    transition = frame[frame["regime"] == "transition"]

    ax.axvspan(
        pd.Timestamp(PRE_WINDOW.start),
        pd.Timestamp(PRE_WINDOW.end),
        color="#cde1ff",
        alpha=0.35,
        label="Pre-window",
    )
    ax.axvspan(
        pd.Timestamp(POST_WINDOW.start),
        pd.Timestamp(POST_WINDOW.end),
        color="#ffd7a8",
        alpha=0.35,
        label="Post-window",
    )
    ax.axvspan(
        TRANSITION_START,
        TRANSITION_END,
        color="0.55",
        alpha=0.4,
        label="Exclusion window",
    )
    universe_label = f"{n_stocks}-firm" if n_stocks else "constant-membership"

    ax.plot(
        frame["day"],
        100.0 * frame["dark_share"],
        color="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.plot(
        pre["day"],
        100.0 * pre["dark_share"],
        color="tab:blue",
        linewidth=1.3,
        zorder=4,
    )
    ax.plot(
        post["day"],
        100.0 * post["dark_share"],
        color="tab:orange",
        linewidth=1.3,
        zorder=4,
    )
    if not transition.empty:
        ax.plot(
            transition["day"],
            100.0 * transition["dark_share"],
            color="0.3",
            linewidth=1.3,
            zorder=4,
        )

    pre_mean = float(100.0 * pre["dark_share"].mean()) if not pre.empty else np.nan
    post_mean = float(100.0 * post["dark_share"].mean()) if not post.empty else np.nan
    if np.isfinite(pre_mean):
        ax.axhline(pre_mean, color="tab:blue", linestyle=(0, (4, 3)), linewidth=0.8)
    if np.isfinite(post_mean):
        ax.axhline(post_mean, color="tab:orange", linestyle=(0, (4, 3)), linewidth=0.8)

    ax.set_ylabel("Universe-wide dark share (%)")
    ax.set_xlabel("Month")
    ax.set_title(
        f"Aggregate dark share across the {universe_label} S&P 500 universe, 2019-06 to 2020-02",
        fontsize=11,
        pad=6,
    )
    ax.set_xlim(pd.Timestamp(PRE_WINDOW.start), pd.Timestamp(POST_WINDOW.end))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower right", frameon=True, fontsize=8, facecolor="white", edgecolor="0.7", framealpha=1.0)

    base = FIGURE_DIR / "h1h2_aggregate_dark_share_daily"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def _regime_mask(timestamps: pd.Series, window) -> pd.Series:
    """Boolean mask for a pre/post window on a minute-level timestamp."""

    ts = pd.to_datetime(timestamps)
    return (ts >= pd.Timestamp(window.start)) & (ts <= pd.Timestamp(window.end))


def _summarize_regime(volumes: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    """Compute descriptive stats for one regime slice."""

    slice_ = volumes.loc[mask]
    if slice_.empty:
        return {key: np.nan for key in (
            "n_minutes",
            "n_stocks",
            "mean_dark_volume",
            "mean_lit_volume",
            "mean_dark_share",
            "std_dark_share",
            "mean_log_dark",
            "mean_log_lit",
            "mean_log_realvar",
        )}
    total_volume = slice_["dark_volume_t"] + slice_["lit_volume_t"]
    dark_share = np.where(total_volume > 0.0, slice_["dark_volume_t"] / total_volume, np.nan)
    return {
        "n_minutes": int(slice_["timestamp"].nunique()),
        "n_stocks": int(slice_["asset"].nunique()),
        "mean_dark_volume": float(slice_["dark_volume_t"].mean()),
        "mean_lit_volume": float(slice_["lit_volume_t"].mean()),
        "mean_dark_share": float(np.nanmean(dark_share)),
        "std_dark_share": float(np.nanstd(dark_share, ddof=1)),
        "mean_log_dark": float(slice_["log_dark_volume_t"].mean()),
        "mean_log_lit": float(slice_["log_lit_volume_t"].mean()),
        "mean_log_realvar": float(slice_["log_total_realized_variance_t"].mean()),
    }


def build_descriptive_stats_table() -> pd.DataFrame:
    """Build a Menkveld Table 1/2-style descriptive stats table.

    One pass over the universe panel pieces is enough to accumulate 
    the per-regime counts and moments. We work in running sums to 
    avoid materializing the full 32M-row panel in memory.
    """

    accum = {
        "pre": {
            "n_obs": 0,
            "sum_dark": 0.0,
            "sum_lit": 0.0,
            "sum_dark_share": 0.0,
            "sumsq_dark_share": 0.0,
            "sum_log_dark": 0.0,
            "sum_log_lit": 0.0,
            "sum_log_realvar": 0.0,
            "stocks": set(),
        },
        "post": {
            "n_obs": 0,
            "sum_dark": 0.0,
            "sum_lit": 0.0,
            "sum_dark_share": 0.0,
            "sumsq_dark_share": 0.0,
            "sum_log_dark": 0.0,
            "sum_log_lit": 0.0,
            "sum_log_realvar": 0.0,
            "stocks": set(),
        },
    }
    pieces = iter_endogenous_panel_pieces(
        tickers=None,
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    )
    for piece in pieces:
        if piece.empty:
            continue
        ts = pd.to_datetime(piece["timestamp"])
        for regime_name, window in (("pre", PRE_WINDOW), ("post", POST_WINDOW)):
            mask = (ts >= pd.Timestamp(window.start)) & (ts <= pd.Timestamp(window.end))
            if not mask.any():
                continue
            sub = piece.loc[mask]
            dark = sub["dark_volume_t"].to_numpy(dtype=float)
            lit = sub["lit_volume_t"].to_numpy(dtype=float)
            total = dark + lit
            share = np.where(total > 0.0, dark / total, np.nan)
            log_dark = sub["log_dark_volume_t"].to_numpy(dtype=float)
            log_lit = sub["log_lit_volume_t"].to_numpy(dtype=float)
            log_realvar = sub["log_total_realized_variance_t"].to_numpy(dtype=float)
            valid = np.isfinite(share)
            bucket = accum[regime_name]
            bucket["n_obs"] += int(valid.sum())
            bucket["sum_dark"] += float(dark[valid].sum())
            bucket["sum_lit"] += float(lit[valid].sum())
            bucket["sum_dark_share"] += float(np.nansum(share[valid]))
            bucket["sumsq_dark_share"] += float(np.nansum(share[valid] ** 2))
            bucket["sum_log_dark"] += float(np.nansum(log_dark[valid]))
            bucket["sum_log_lit"] += float(np.nansum(log_lit[valid]))
            bucket["sum_log_realvar"] += float(np.nansum(log_realvar[valid]))
            bucket["stocks"].add(str(sub["asset"].iloc[0]))

    rows: list[dict[str, object]] = []
    for regime_name in ("pre", "post"):
        bucket = accum[regime_name]
        n = bucket["n_obs"]
        if n == 0:
            continue
        mean_share = bucket["sum_dark_share"] / n
        var_share = max(bucket["sumsq_dark_share"] / n - mean_share * mean_share, 0.0)
        rows.append(
            {
                "regime": regime_name,
                "n_stocks": len(bucket["stocks"]),
                "n_minute_obs": n,
                "mean_dark_volume": bucket["sum_dark"] / n,
                "mean_lit_volume": bucket["sum_lit"] / n,
                "aggregate_dark_share": bucket["sum_dark"] / (bucket["sum_dark"] + bucket["sum_lit"]),
                "mean_dark_share": mean_share,
                "sd_dark_share": float(np.sqrt(var_share)),
                "mean_log_dark": bucket["sum_log_dark"] / n,
                "mean_log_lit": bucket["sum_log_lit"] / n,
                "mean_log_realvar": bucket["sum_log_realvar"] / n,
            }
        )
    return pd.DataFrame(rows)


def save_descriptive_stats_table(frame: pd.DataFrame) -> Path:
    """Save the descriptive stats table as CSV."""

    path = TABLE_DIR / "h1h2_descriptive_stats_pre_post.csv"
    frame.to_csv(path, index=False)
    return path


def build_cluster_diagnostics_table() -> pd.DataFrame:
    """Stack the per-family cluster-diagnostic rows into a single table."""

    h1h2_dir = _local.OUTPUT_DIR
    families = ("vix", "macro", "earnings")
    frames = []
    for family in families:
        candidate = h1h2_dir / f"h1h2_{family}_cluster_diagnostics.csv"
        if not candidate.exists():
            continue
        frames.append(pd.read_csv(candidate))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_cluster_diagnostics_table(frame: pd.DataFrame) -> Path:
    """Save a single H1/H2 cluster-diagnostics summary CSV."""

    path = TABLE_DIR / "h1h2_cluster_diagnostics_summary.csv"
    frame.to_csv(path, index=False)
    return path


def run_menkveld_extras() -> dict[str, list[Path] | Path]:
    """Build the motivational figure + descriptive stats + cluster summary."""

    outputs: dict[str, list[Path] | Path] = {}

    print("Building daily dark-share series (requires a full universe pass)...")
    daily, n_stocks = build_daily_dark_share_series()
    outputs["daily_dark_share_csv"] = save_daily_dark_share_table(daily)
    outputs["daily_dark_share_figures"] = plot_daily_dark_share_series(daily, n_stocks=n_stocks)

    print("Building descriptive stats table (requires another universe pass)...")
    descriptive = build_descriptive_stats_table()
    outputs["descriptive_stats_csv"] = save_descriptive_stats_table(descriptive)

    print("Stacking cluster-diagnostics summary...")
    cluster = build_cluster_diagnostics_table()
    if not cluster.empty:
        outputs["cluster_diagnostics_csv"] = save_cluster_diagnostics_table(cluster)

    return outputs

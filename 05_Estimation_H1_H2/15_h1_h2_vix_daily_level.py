"""Daily-average VIX level figure for Chapter 3.

One-panel line plot of the daily-average minute closing VIX across the
H1/H2 sample, regular trading session only, with pre/post regime and
exclusion-window shading. Outputs PDF and PNG to the H1/H2 presentation
figures directory.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

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
_local = importlib.import_module("01_estimation_config")
_presentation = importlib.import_module("05_h1_h2_presentation")


FIGURE_DIR = _presentation.FIGURE_DIR
PRE_WINDOW = _local.PRE_WINDOW
POST_WINDOW = _local.POST_WINDOW
TRANSITION_START = pd.Timestamp("2019-10-01")
TRANSITION_END = pd.Timestamp("2019-10-10")
REGULAR_SESSION = _config_beta.REGULAR_SESSION


def _load_daily_vix() -> pd.DataFrame:
    frame = pd.read_csv(_config_beta.VIX_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["time_str"] = frame["Time"].astype(str).str.zfill(4)
    frame["timestamp"] = pd.to_datetime(
        frame["Date"].dt.strftime("%Y-%m-%d") + " " + frame["time_str"],
        format="%Y-%m-%d %H%M",
    )
    frame = frame[
        (frame["timestamp"] >= pd.Timestamp(_config_beta.SAMPLE_START))
        & (frame["timestamp"] <= pd.Timestamp(_config_beta.SAMPLE_END))
    ].copy()
    frame = _utils.filter_session(
        frame,
        timestamp_col="timestamp",
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    )
    frame["day"] = frame["timestamp"].dt.normalize()
    daily = frame.groupby("day", as_index=False)["Close"].mean()
    return daily.rename(columns={"Close": "vix_daily_mean"})


def _regime_label(day: pd.Series) -> pd.Series:
    pre = (day >= pd.Timestamp(PRE_WINDOW.start).normalize()) & (
        day <= pd.Timestamp(PRE_WINDOW.end).normalize()
    )
    post = (day >= pd.Timestamp(POST_WINDOW.start).normalize()) & (
        day <= pd.Timestamp(POST_WINDOW.end).normalize()
    )
    trans = (day >= TRANSITION_START) & (day <= TRANSITION_END)
    return np.where(pre, "pre", np.where(post, "post", np.where(trans, "transition", "other")))


def build_figure() -> dict[str, float | int]:
    daily = _load_daily_vix()
    daily["regime"] = _regime_label(daily["day"])

    pre = daily[daily["regime"] == "pre"]
    post = daily[daily["regime"] == "post"]
    transition = daily[daily["regime"] == "transition"]

    pre_mean = float(pre["vix_daily_mean"].mean())
    post_mean = float(post["vix_daily_mean"].mean())

    _presentation.apply_menkveld_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)

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

    ax.plot(
        daily["day"],
        daily["vix_daily_mean"],
        color="black",
        linewidth=0.9,
        zorder=3,
    )
    ax.plot(
        pre["day"], pre["vix_daily_mean"],
        color="tab:blue", linewidth=1.3, zorder=4,
    )
    ax.plot(
        post["day"], post["vix_daily_mean"],
        color="tab:orange", linewidth=1.3, zorder=4,
    )
    if np.isfinite(pre_mean):
        ax.axhline(pre_mean, color="tab:blue", linestyle=(0, (4, 3)), linewidth=0.8)
    if np.isfinite(post_mean):
        ax.axhline(post_mean, color="tab:orange", linestyle=(0, (4, 3)), linewidth=0.8)

    ax.set_ylabel("Daily-average VIX (index points)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", frameon=True, fontsize=8,
              facecolor="white", edgecolor="0.7", framealpha=1.0)
    ax.margins(x=0.005)

    base = FIGURE_DIR / "h1h2_vix_daily_level"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "n_pre_days": int(len(pre)),
        "n_post_days": int(len(post)),
        "pre_max": float(pre["vix_daily_mean"].max()),
        "post_max": float(post["vix_daily_mean"].max()),
        "pre_max_day": str(pre.loc[pre["vix_daily_mean"].idxmax(), "day"].date()),
        "post_max_day": str(post.loc[post["vix_daily_mean"].idxmax(), "day"].date()),
    }


if __name__ == "__main__":
    stats = build_figure()
    for key, value in stats.items():
        print(f"{key}: {value}")

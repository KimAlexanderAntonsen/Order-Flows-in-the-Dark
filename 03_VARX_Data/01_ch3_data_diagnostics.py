"""Chapter 3 data diagnostics: VIX innovation tails (pre vs post) and the
pre/post split of the matched earnings announcement set.

Replicates the AR(1) construction used by 04_VARX/04_beta_varx_data.py so
the numbers reported in the chapter are identical to the ones the
estimation pipeline actually consumes.

Run from the repo root:
    python 03_VARX_Data/ch3_data_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VIX_PATH = ROOT / "03_VARX_Data" / "data_raw" / "VIX.txt"
EARNINGS_PATH = ROOT / "03_VARX_Data" / "data_clean" / "earnings_urgency_sparse_panel.csv"
OUTPUT_PATH = ROOT / "03_VARX_Data" / "data_clean" / "ch3_diagnostics.json"

SAMPLE_START = pd.Timestamp("2019-06-10 00:00:00")
SAMPLE_END = pd.Timestamp("2020-02-19 23:59:59")
TRANSITION_START = pd.Timestamp("2019-10-01 00:00:00")
TRANSITION_END = pd.Timestamp("2019-10-10 23:59:59")


def load_vix_innovations() -> pd.DataFrame:
    """Replicate the Menkveld-style AR(1) innovation construction."""
    frame = pd.read_csv(VIX_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["time_str"] = frame["Time"].astype(str).str.zfill(4)
    frame["timestamp"] = pd.to_datetime(
        frame["Date"].dt.strftime("%Y-%m-%d") + " " + frame["time_str"],
        format="%Y-%m-%d %H%M",
    )

    # Apply sample window and exclude exclusion window.
    mask = (frame["timestamp"] >= SAMPLE_START) & (frame["timestamp"] <= SAMPLE_END)
    transition = (frame["timestamp"] >= TRANSITION_START) & (frame["timestamp"] <= TRANSITION_END)
    frame = frame.loc[mask & ~transition].copy()

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
    frame["regime"] = np.where(frame["timestamp"] < TRANSITION_START, "pre", "post")

    return frame.dropna(subset=["dVIX_innovation"]).reset_index(drop=True)


def vix_diagnostics(frame: pd.DataFrame) -> dict:
    out: dict = {}
    full_sigma = float(frame["dVIX_innovation"].std(ddof=1))
    out["full_sigma"] = full_sigma

    for regime in ("pre", "post"):
        sub = frame[frame["regime"] == regime]
        eps = sub["dVIX_innovation"].to_numpy()
        pos = sub["dVIX_pos_inv"].to_numpy()
        pos = pos[pos > 0]
        abs_eps = np.abs(eps)

        cell = {
            "n_minutes": int(len(sub)),
            "sigma": float(eps.std(ddof=1)),
            "max_abs": float(abs_eps.max()),
            "max_positive": float(eps.max()),
            "min_negative": float(eps.min()),
            "p99_abs": float(np.percentile(abs_eps, 99)),
            "p999_abs": float(np.percentile(abs_eps, 99.9)),
            "p9999_abs": float(np.percentile(abs_eps, 99.99)),
            "p99_positive": float(np.percentile(pos, 99)) if pos.size else None,
            "p999_positive": float(np.percentile(pos, 99.9)) if pos.size else None,
        }
        for k in (3, 5, 10):
            thr = k * full_sigma
            cell[f"n_abs_ge_{k}sigma"] = int((abs_eps >= thr).sum())
            cell[f"n_pos_ge_{k}sigma"] = int((eps >= thr).sum())
            cell[f"n_neg_le_-{k}sigma"] = int((eps <= -thr).sum())
        out[regime] = cell

    # Top-5 single-minute innovations per regime (positive part), with date stamps.
    top_minutes = {}
    for regime in ("pre", "post"):
        sub = frame[frame["regime"] == regime].nlargest(5, "dVIX_pos_inv")
        top_minutes[regime] = [
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "VIX_close": float(vc),
                "innovation": float(inv),
            }
            for ts, vc, inv in zip(sub["timestamp"], sub["VIX_close"], sub["dVIX_pos_inv"])
        ]
    out["top_positive_minutes"] = top_minutes

    # Concentration: share of positive-innovation mass on the single largest day.
    pos_mass = {}
    for regime in ("pre", "post"):
        sub = frame[frame["regime"] == regime].copy()
        sub["date"] = sub["timestamp"].dt.date
        daily_sum = sub.groupby("date")["dVIX_pos_inv"].sum()
        total = float(daily_sum.sum())
        top_day = daily_sum.idxmax()
        top_day_share = float(daily_sum.max() / total) if total > 0 else 0.0
        pos_mass[regime] = {
            "total_positive_innovation_mass": total,
            "top_day": str(top_day),
            "top_day_share": top_day_share,
        }
    out["positive_mass_concentration"] = pos_mass

    return out


def earnings_split() -> dict:
    df = pd.read_csv(EARNINGS_PATH, usecols=["ticker", "announce_datetime", "effect_date"])
    df["announce_datetime"] = pd.to_datetime(df["announce_datetime"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["effect_date"] = pd.to_datetime(df["effect_date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)

    events = df.drop_duplicates(subset=["ticker", "announce_datetime"]).copy()
    events["regime"] = np.where(events["announce_datetime"] < TRANSITION_START, "pre", "post")
    # Drop any events landing inside the transition window itself.
    in_transition = (events["announce_datetime"] >= TRANSITION_START) & (events["announce_datetime"] <= TRANSITION_END)
    n_in_transition = int(in_transition.sum())
    events = events.loc[~in_transition]

    return {
        "total_unique_announcements": int(len(events) + n_in_transition),
        "in_transition_window": n_in_transition,
        "pre_count": int((events["regime"] == "pre").sum()),
        "post_count": int((events["regime"] == "post").sum()),
        "unique_tickers_pre": int(events.loc[events["regime"] == "pre", "ticker"].nunique()),
        "unique_tickers_post": int(events.loc[events["regime"] == "post", "ticker"].nunique()),
    }


def main() -> None:
    frame = load_vix_innovations()
    vix_out = vix_diagnostics(frame)
    earnings_out = earnings_split()
    summary = {"vix": vix_out, "earnings": earnings_out}

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

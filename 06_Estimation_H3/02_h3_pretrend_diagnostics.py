"""Pre-trend and comparability diagnostics for H3.

This module is intentionally simpler than the final H3 estimator. 
Its job is to answer one question first: are the current treated and 
control groups credible enough to compare in a DiD-style design?
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

BETA_DIR = H3_DIR.parents[0] / "04_VARX"
if str(BETA_DIR) not in sys.path:
    sys.path.insert(0, str(BETA_DIR))

_config = importlib.import_module("01_h3_config")
_beta_config = importlib.import_module("02_beta_varx_config")
_beta_data = importlib.import_module("04_beta_varx_data")
_beta_utils = importlib.import_module("03_beta_varx_utils")


FIGURE_DIR = _config.FIGURE_DIR
GROUP_LABELS = _config.GROUP_LABELS
GROUP_ORDER = _config.GROUP_ORDER
MAIN_GROUPS = _config.MAIN_GROUPS
MATCHED_CONTROL_PATH = _config.MATCHED_CONTROL_PATH
OUTCOME_COLS = _config.OUTCOME_COLS
OUTCOME_LABELS = _config.OUTCOME_LABELS
PLACEBO_SPLIT_DATE = pd.Timestamp(_config.PLACEBO_SPLIT_DATE)
PRE_PERIOD_END = pd.Timestamp(_config.PRE_PERIOD_END)
PRE_PERIOD_START = pd.Timestamp(_config.PRE_PERIOD_START)
REFERENCE_PATH = _config.REFERENCE_PATH
SCORE_TABLE_PATH = _config.SCORE_TABLE_PATH
TABLE_DIR = _config.TABLE_DIR
TREATED_PATH = _config.TREATED_PATH

REALIZED_VARIANCE_FLOOR = _beta_config.REALIZED_VARIANCE_FLOOR
REGULAR_SESSION = _beta_config.REGULAR_SESSION
VOLUME_FLOOR = _beta_config.VOLUME_FLOOR

filter_session = _beta_utils.filter_session
load_minute_bar = _beta_data.load_minute_bar
safe_log = _beta_utils.safe_log


def ensure_output_dirs() -> None:
    """Create the H3 output folders if they are missing."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_group_definitions() -> dict[str, list[str]]:
    """Load the benchmark H3 group files."""

    treated = pd.read_csv(TREATED_PATH)["ticker"].astype(str).tolist()
    matched = pd.read_csv(MATCHED_CONTROL_PATH)["ticker"].astype(str).tolist()
    reference = pd.read_csv(REFERENCE_PATH)["ticker"].astype(str).tolist()
    return {
        "treated": treated,
        "matched_control": matched,
        "least_retail_reference": reference,
    }


def load_score_table() -> pd.DataFrame:
    """Load the asset-level retail-score table used in the classification."""

    return pd.read_csv(SCORE_TABLE_PATH)


def _collapse_stock_to_days(ticker: str, group_name: str) -> pd.DataFrame:
    """Collapse one stock's minute data into pre-period stock-day 
    outcomes.

    The H3 diagnostics do not need the full minute panel yet. A 
    stock-day view is much easier to inspect and is already 
    informative for pre-trend checks.
    """

    frame = load_minute_bar(ticker)
    frame = filter_session(
        frame,
        timestamp_col="timestamp",
        session_start=REGULAR_SESSION.start,
        session_end=REGULAR_SESSION.end,
    )
    frame = frame[
        (frame["timestamp"] >= PRE_PERIOD_START) & (frame["timestamp"] <= PRE_PERIOD_END)
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["total_volume_t"] = (
        pd.to_numeric(frame["dark_volume"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["lit_volume"], errors="coerce").fillna(0.0)
    )
    frame["total_realized_variance_t"] = (
        pd.to_numeric(frame["dark_realized_variance"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["lit_realized_variance"], errors="coerce").fillna(0.0)
    )
    frame["date"] = frame["timestamp"].dt.normalize()

    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            dark_volume=("dark_volume", "sum"),
            lit_volume=("lit_volume", "sum"),
            total_volume=("total_volume_t", "sum"),
            total_realized_variance=("total_realized_variance_t", "sum"),
            minute_obs=("timestamp", "size"),
        )
        .sort_values("date")
    )

    total_volume = daily["total_volume"].to_numpy(dtype=float)
    dark_volume = daily["dark_volume"].to_numpy(dtype=float)
    lit_volume = daily["lit_volume"].to_numpy(dtype=float)

    daily["dark_share"] = np.divide(
        dark_volume,
        total_volume,
        out=np.zeros_like(dark_volume, dtype=float),
        where=total_volume > 0,
    )
    daily["lit_share"] = np.divide(
        lit_volume,
        total_volume,
        out=np.zeros_like(lit_volume, dtype=float),
        where=total_volume > 0,
    )
    daily["log_total_volume"] = safe_log(daily["total_volume"], VOLUME_FLOOR)
    daily["log_total_realized_variance"] = safe_log(
        daily["total_realized_variance"],
        REALIZED_VARIANCE_FLOOR,
    )

    daily["asset"] = ticker
    daily["group"] = group_name
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
    return daily[
        [
            "asset",
            "group",
            "date",
            "week_start",
            "dark_volume",
            "lit_volume",
            "total_volume",
            "total_realized_variance",
            "minute_obs",
            *OUTCOME_COLS,
        ]
    ].copy()


def build_pretrend_stock_day_panel(group_definitions: dict[str, list[str]]) -> pd.DataFrame:
    """Build the combined pre-period stock-day panel for the H3 groups."""

    pieces: list[pd.DataFrame] = []
    for group_name in GROUP_ORDER:
        for ticker in group_definitions[group_name]:
            daily = _collapse_stock_to_days(ticker, group_name)
            if not daily.empty:
                pieces.append(daily)

    panel = pd.concat(pieces, ignore_index=True)
    return panel.sort_values(["group", "asset", "date"]).reset_index(drop=True)


def build_sample_summary(stock_day: pd.DataFrame) -> pd.DataFrame:
    """Summarize group sizes and coverage in the pre-period diagnostic panel."""

    summary = (
        stock_day.groupby("group", as_index=False)
        .agg(
            n_assets=("asset", "nunique"),
            n_stock_days=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            mean_stock_days=("asset", lambda s: s.size / s.nunique()),
        )
        .sort_values("group")
    )
    summary["group_label"] = summary["group"].map(GROUP_LABELS)
    return summary[["group", "group_label", "n_assets", "n_stock_days", "mean_stock_days", "first_date", "last_date"]]


def build_comparability_summary(stock_day: pd.DataFrame) -> pd.DataFrame:
    """Create a simple pre-period balance table on stock-level means."""

    stock_means = (
        stock_day.groupby(["group", "asset"], as_index=False)[list(OUTCOME_COLS)].mean()
    )

    def _extract(group_name: str, column: str) -> pd.Series:
        return stock_means.loc[stock_means["group"] == group_name, column].astype(float)

    def _smd(a: pd.Series, b: pd.Series) -> float:
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        if pooled == 0 or np.isnan(pooled):
            return np.nan
        return float((a.mean() - b.mean()) / pooled)

    rows = []
    for outcome in OUTCOME_COLS:
        treated = _extract("treated", outcome)
        matched = _extract("matched_control", outcome)
        reference = _extract("least_retail_reference", outcome)
        rows.append(
            {
                "outcome": outcome,
                "treated_mean": treated.mean(),
                "matched_control_mean": matched.mean(),
                "least_retail_reference_mean": reference.mean(),
                "treated_vs_matched_diff": treated.mean() - matched.mean(),
                "treated_vs_reference_diff": treated.mean() - reference.mean(),
                "treated_vs_matched_smd": _smd(treated, matched),
                "treated_vs_reference_smd": _smd(treated, reference),
            }
        )
    return pd.DataFrame(rows)


def build_group_series(stock_day: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    """Collapse the stock-day panel into group-average daily or weekly 
    series.

    We average at the stock level within each day or week, so the 
    plots are not dominated by a handful of extremely active names.
    """

    if frequency == "daily":
        time_col = "date"
        collapsed = stock_day.copy()
    elif frequency == "weekly":
        time_col = "week_start"
        collapsed = (
            stock_day.groupby(["group", "asset", "week_start"], as_index=False)[list(OUTCOME_COLS)].mean()
        )
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    grouped = (
        collapsed.groupby(["group", time_col], as_index=False)
        .agg(
            n_assets=("asset", "nunique"),
            **{col: (col, "mean") for col in OUTCOME_COLS},
        )
        .sort_values(["group", time_col])
    )
    grouped = grouped.rename(columns={time_col: "time"})
    grouped["frequency"] = frequency
    return grouped


def build_gap_series(group_series: pd.DataFrame) -> pd.DataFrame:
    """Construct treated-minus-control gaps from the collapsed group series."""

    treated = group_series[group_series["group"] == "treated"].copy()
    rows = []
    for control_group in ("matched_control", "least_retail_reference"):
        control = group_series[group_series["group"] == control_group].copy()
        merged = treated.merge(
            control,
            on=["time", "frequency"],
            suffixes=("_treated", "_control"),
            how="inner",
        )
        for outcome in OUTCOME_COLS:
            merged[f"{outcome}_gap"] = merged[f"{outcome}_treated"] - merged[f"{outcome}_control"]
        keep_cols = ["time", "frequency"]
        keep_cols.extend(f"{outcome}_gap" for outcome in OUTCOME_COLS)
        gap = merged[keep_cols].copy()
        gap["comparison"] = f"treated_minus_{control_group}"
        rows.append(gap)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["comparison", "frequency", "time"]).reset_index(drop=True)


def _run_fixed_effect_regression(formula: str, data: pd.DataFrame, cluster_col: str):
    """Fit a simple OLS with stock fixed effects and clustered standard errors."""

    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]},
    )
    return model


def build_trend_test_summary(stock_day: pd.DataFrame) -> pd.DataFrame:
    """Test whether treated stocks have different pre-period linear trends."""

    rows = []
    for control_group in ("matched_control", "least_retail_reference"):
        subset = stock_day[stock_day["group"].isin(["treated", control_group])].copy()
        subset["treated_flag"] = (subset["group"] == "treated").astype(int)
        subset["day_index"] = (subset["date"] - subset["date"].min()).dt.days.astype(int)

        for outcome in OUTCOME_COLS:
            model = _run_fixed_effect_regression(
                f"{outcome} ~ day_index + treated_flag:day_index + C(asset)",
                subset,
                cluster_col="asset",
            )
            term = "treated_flag:day_index"
            rows.append(
                {
                    "comparison": f"treated_vs_{control_group}",
                    "outcome": outcome,
                    "coef": model.params.get(term, np.nan),
                    "std_error": model.bse.get(term, np.nan),
                    "t_stat": model.tvalues.get(term, np.nan),
                    "p_value": model.pvalues.get(term, np.nan),
                    "n_obs": int(model.nobs),
                    "n_assets": subset["asset"].nunique(),
                }
            )

    return pd.DataFrame(rows)


def build_placebo_did_summary(stock_day: pd.DataFrame) -> pd.DataFrame:
    """Run a simple placebo DiD inside the pre period.

    We split the pre period into an early and late window, long before
    the October 2019 break, and ask whether the treated group appears 
    to receive a fake treatment effect. Large placebo effects would 
    make the H3 benchmark much less convincing.
    """

    rows = []
    for control_group in ("matched_control", "least_retail_reference"):
        subset = stock_day[stock_day["group"].isin(["treated", control_group])].copy()
        subset["treated_flag"] = (subset["group"] == "treated").astype(int)
        subset["pseudo_post"] = (subset["date"] >= PLACEBO_SPLIT_DATE).astype(int)

        for outcome in OUTCOME_COLS:
            model = _run_fixed_effect_regression(
                f"{outcome} ~ pseudo_post + treated_flag:pseudo_post + C(asset)",
                subset,
                cluster_col="asset",
            )
            term = "treated_flag:pseudo_post"
            rows.append(
                {
                    "comparison": f"treated_vs_{control_group}",
                    "outcome": outcome,
                    "placebo_split_date": PLACEBO_SPLIT_DATE.date().isoformat(),
                    "coef": model.params.get(term, np.nan),
                    "std_error": model.bse.get(term, np.nan),
                    "t_stat": model.tvalues.get(term, np.nan),
                    "p_value": model.pvalues.get(term, np.nan),
                    "n_obs": int(model.nobs),
                    "n_assets": subset["asset"].nunique(),
                }
            )

    return pd.DataFrame(rows)


def plot_group_series(group_series: pd.DataFrame, *, frequency: str, path: Path) -> None:
    """Plot pre-period group averages for the main H3 outcomes."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    colors = {
        "treated": "#1f77b4",
        "matched_control": "#2ca02c",
        "least_retail_reference": "#7f7f7f",
    }

    for ax, outcome in zip(axes, OUTCOME_COLS):
        for group_name in GROUP_ORDER:
            series = group_series[group_series["group"] == group_name]
            ax.plot(
                series["time"],
                series[outcome],
                label=GROUP_LABELS[group_name],
                color=colors[group_name],
                linewidth=1.4,
            )
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(alpha=0.2, linewidth=0.6)
        if "share" in outcome:
            ax.set_ylim(0, 1)

    if frequency == "daily":
        locator = mdates.WeekdayLocator(interval=2)
        formatter = mdates.DateFormatter("%Y-%m-%d")
        x_label = "Pre-period day"
    else:
        locator = mdates.WeekdayLocator(interval=2)
        formatter = mdates.DateFormatter("%Y-%m-%d")
        x_label = "Pre-period week"

    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    axes[0].legend(frameon=False, ncol=1, fontsize=9)
    fig.supxlabel(x_label)
    fig.suptitle("H3 pre-period group averages", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap_series(gap_series: pd.DataFrame, *, frequency: str, path: Path) -> None:
    """Plot treated-minus-control pre-period gaps for the main outcomes."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    styles = {
        "treated_minus_matched_control": ("#1f77b4", "-"),
        "treated_minus_least_retail_reference": ("#7f7f7f", "--"),
    }

    for ax, outcome in zip(axes, OUTCOME_COLS):
        for comparison, (color, linestyle) in styles.items():
            series = gap_series[gap_series["comparison"] == comparison]
            ax.plot(
                series["time"],
                series[f"{outcome}_gap"],
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=comparison.replace("_", " "),
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{OUTCOME_LABELS[outcome]} gap")
        ax.grid(alpha=0.2, linewidth=0.6)

    locator = mdates.WeekdayLocator(interval=2)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    axes[0].legend(frameon=False, fontsize=9)
    fig.supxlabel("Pre-period time")
    fig.suptitle("H3 pre-period treated-minus-control gaps", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

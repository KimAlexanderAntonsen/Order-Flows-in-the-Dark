"""Generate the retail-score histogram, the retail-score vs
trading-volume scatter, the retail-score vs market-cap scatter, and
the level-analogue group-cells table used in Chapter 5.

Outputs
07_Thesis/latex/figures/retail_score_distribution.pdf
  Histogram of the pre-period retail-intensity score across all scored
  firms.

07_Thesis/latex/figures/retail_score_vs_volume.pdf
  Scatter of retail-intensity score vs log mean-minute trading volume
  for all scored firms, with treated and matched-control groups
  highlighted.

07_Thesis/latex/figures/retail_score_vs_mktcap.pdf
  Scatter of retail-intensity score vs average market capitalisation
  (mean of daily market-cap readings across the Nasdaq earnings
  calendar JSON files) for all scored firms, with treated and
  matched-control groups highlighted.

07_Thesis/latex/tables/group_cells.tex
  LaTeX snippet for tab:group-cells: the four (group, regime) cells of
  mean dark share, with Welch t-tests on the treated-minus-matched gap
  in each window and on the post-minus-pre difference-in-differences.

07_Thesis/latex/tables/group_balance.tex
  LaTeX snippet for the 5-row Welch balance table (retail score plus 
  the four matching features). Kept available for reuse but not 
  auto-included in the chapter.

06_Estimation_H3/output/group_cells_table.csv
  Row-level numbers behind the balance LaTeX snippet.

06_Estimation_H3/output/group_cells_levels_table.csv
  Row-level numbers behind the 4-cell level-analogue table.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETAIL_DIR = PROJECT_ROOT / "02_RetailClassification"
MINUTE_BAR_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "minute_bars"
MARKETCAP_DIR = (
    PROJECT_ROOT
    / "03_VARX_Data"
    / "data_raw"
    / "earnings_nasdaq_calendar_daily"
)
PRETREND_PANEL = (
    PROJECT_ROOT
    / "06_Estimation_H3"
    / "output"
    / "pretrend"
    / "tables"
    / "h3_pretrend_stock_day_panel.csv"
)
FIGURE_OUT = PROJECT_ROOT / "07_Thesis" / "latex" / "figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

# Import the same minute-bar loader and session filter the pretrend
# diagnostics use, so the post-window panel here is built on identical
# code paths.
H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))
BETA_DIR = PROJECT_ROOT / "04_VARX"
if str(BETA_DIR) not in sys.path:
    sys.path.insert(0, str(BETA_DIR))

_h3_config = importlib.import_module("01_h3_config")
_beta_config = importlib.import_module("02_beta_varx_config")
_beta_data = importlib.import_module("04_beta_varx_data")
_beta_utils = importlib.import_module("03_beta_varx_utils")

load_minute_bar = _beta_data.load_minute_bar
filter_session = _beta_utils.filter_session
REGULAR_SESSION = _beta_config.REGULAR_SESSION

# Sample windows
PRE_START = pd.Timestamp(_h3_config.PRE_PERIOD_START)
PRE_END = pd.Timestamp(_h3_config.PRE_PERIOD_END)
POST_START = pd.Timestamp(_h3_config.POST_PERIOD_START)
POST_END = pd.Timestamp(_h3_config.POST_PERIOD_END)
SESSION_START = REGULAR_SESSION.start
SESSION_END = REGULAR_SESSION.end

# Load group definitions
def load_groups() -> tuple[list[str], list[str]]:
    treated = (
        pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_treated_group.csv")["ticker"]
        .astype(str)
        .tolist()
    )
    matched = (
        pd.read_csv(RETAIL_DIR / "group_outputs" / "retail_matched_control_group.csv")["ticker"]
        .astype(str)
        .tolist()
    )
    return treated, matched


def load_score_table() -> pd.DataFrame:
    return pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_score_with_matching_features.csv"
    )

# Group balance table (tab:group-cells)
TABLE_OUT = PROJECT_ROOT / "06_Estimation_H3" / "output" / "group_cells_table.csv"
LATEX_TABLE_OUT = (
    PROJECT_ROOT / "07_Thesis" / "latex" / "tables" / "group_balance.tex"
)
TABLE_REF_OUT = (
    PROJECT_ROOT / "06_Estimation_H3" / "output" / "group_cells_table_reference.csv"
)
LATEX_TABLE_REF_OUT = (
    PROJECT_ROOT
    / "07_Thesis"
    / "latex"
    / "tables"
    / "group_balance_reference.tex"
)

LEVELS_CSV_OUT = (
    PROJECT_ROOT / "06_Estimation_H3" / "output" / "group_cells_levels_table.csv"
)
LEVELS_LATEX_OUT = (
    PROJECT_ROOT / "07_Thesis" / "latex" / "tables" / "group_cells.tex"
)

# (column_in_csv, display_label, format_spec, is_matching_feature)
BALANCE_FEATURES = [
    ("retail_score",              "Retail score",          ".3f", False),
    ("pre_mean_log_total_volume", "Mean log total volume", ".3f", True),
    ("pre_mean_dark_share",       "Mean dark share",       ".3f", True),
    ("pre_return_std",            "Return volatility",     ".5f", True),
    ("pre_median_price",          "Median price",          ".2f", True),
]


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Welch t-test; returns (t_stat, p_value)."""

    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def compute_and_write_balance_table() -> None:
    """Welch t-tests of treated vs matched control on retail score and the
    four matching features. Writes tab:group-cells LaTeX snippet and a CSV
    of the row-level numbers."""

    treated_df = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_treated_group.csv"
    )
    matched_df = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_matched_control_group.csv"
    )
    n_t = len(treated_df)
    n_c = len(matched_df)

    rows: list[dict] = []
    for col, label, spec, is_match in BALANCE_FEATURES:
        a = treated_df[col].dropna().to_numpy(dtype=float)
        b = matched_df[col].dropna().to_numpy(dtype=float)
        t_stat, p_val = _ttest(a, b)
        rows.append({
            "feature": col,
            "label": label,
            "treated_mean": float(a.mean()),
            "matched_mean": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "t_stat": t_stat,
            "p_value": p_val,
            "stars": _stars(p_val),
            "is_matching_feature": is_match,
            "_spec": spec,
        })

    # CSV (without internal _spec column)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
        TABLE_OUT, index=False
    )
    print(f"\nSaved: {TABLE_OUT}")

    # Console pretty-print
    print("\n" + "=" * 90)
    print("BALANCE TABLE: treated vs matched control (tab:group-cells)")
    print("=" * 90)
    print(f"{'Variable':<26} {'Treated':>11} {'Matched':>11} {'Diff':>11} {'t-stat':>9} {'p-value':>9}")
    print("-" * 90)
    for r in rows:
        spec = r["_spec"]
        t_str = f"{r['treated_mean']:{spec}}"
        c_str = f"{r['matched_mean']:{spec}}"
        d_str = f"{r['diff']:+{spec}}"
        print(
            f"{r['label']:<26} {t_str:>11} {c_str:>11} {d_str:>11} "
            f"{r['t_stat']:>+9.3f} {r['p_value']:>9.4f} {r['stars']}"
        )
    print("=" * 90)
    print(f"Sample sizes: treated n = {n_t}, matched control n = {n_c}")

    # LaTeX snippet
    def _row(r: dict) -> str:
        spec = r["_spec"]
        return (
            rf"{r['label']} & "
            rf"${r['treated_mean']:{spec}}$ & "
            rf"${r['matched_mean']:{spec}}$ & "
            rf"${r['diff']:+{spec}}$ & "
            rf"${r['t_stat']:+.2f}$ & "
            rf"${r['p_value']:.4f}${r['stars']} \\"
        )

    score_row = _row(rows[0])
    match_rows = "\n".join(_row(r) for r in rows[1:])

    snippet = rf"""% Requires packages: booktabs, tabularx, array, caption
% Auto-generated by 06_Estimation_H3/04_h3_group_cells_and_scatter.py.
% Do not edit by hand; rerun the script to refresh.
\begin{{table}}[H]
\centering\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\caption{{Pre-window balance: treated vs matched control.}}
\label{{tab:group-balance}}
\begin{{tabularx}}{{\textwidth}}{{Xrrrrr}}
\toprule
Variable & Treated mean & Matched control mean & Diff ($T - C$) & $t$-stat & $p$-value \\
\midrule
{score_row}
\midrule
{match_rows}
\bottomrule
\end{{tabularx}}
\caption*{{\footnotesize Note: Welch's two-sample $t$-tests of treated ($n = {n_t}$) versus matched control ($n = {n_c}$) on pre-window (June 10 -- September 27, 2019) characteristics. Retail score is shown for reference; the four rows below it are the features used in the one-to-one nearest-neighbour match (mean log total volume, mean dark share, intraday return volatility, median price). * $p < 0.10$, ** $p < 0.05$, *** $p < 0.01$.}}
\end{{table}}
"""

    LATEX_TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    LATEX_TABLE_OUT.write_text(snippet, encoding="utf-8")
    print(f"Saved: {LATEX_TABLE_OUT}")


def compute_and_write_balance_table_vs_reference() -> None:
    """Welch t-tests of treated vs least-retail reference on retail score
    and the four matching features. Mirrors compute_and_write_balance_table
    but swaps the matched-control group for the least-retail reference
    group, which is the bottom 70 firms by retail score within the H3
    candidate pool."""

    treated_df = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_treated_group.csv"
    )
    reference_df = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_reference_group.csv"
    )
    n_t = len(treated_df)
    n_r = len(reference_df)

    rows: list[dict] = []
    for col, label, spec, is_match in BALANCE_FEATURES:
        a = treated_df[col].dropna().to_numpy(dtype=float)
        b = reference_df[col].dropna().to_numpy(dtype=float)
        t_stat, p_val = _ttest(a, b)
        rows.append({
            "feature": col,
            "label": label,
            "treated_mean": float(a.mean()),
            "reference_mean": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "t_stat": t_stat,
            "p_value": p_val,
            "stars": _stars(p_val),
            "is_matching_feature": is_match,
            "_spec": spec,
        })

    TABLE_REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
        TABLE_REF_OUT, index=False
    )
    print(f"\nSaved: {TABLE_REF_OUT}")

    print("\n" + "=" * 90)
    print("BALANCE TABLE: treated vs least-retail reference")
    print("=" * 90)
    print(f"{'Variable':<26} {'Treated':>11} {'Reference':>11} {'Diff':>11} {'t-stat':>9} {'p-value':>9}")
    print("-" * 90)
    for r in rows:
        spec = r["_spec"]
        t_str = f"{r['treated_mean']:{spec}}"
        c_str = f"{r['reference_mean']:{spec}}"
        d_str = f"{r['diff']:+{spec}}"
        print(
            f"{r['label']:<26} {t_str:>11} {c_str:>11} {d_str:>11} "
            f"{r['t_stat']:>+9.3f} {r['p_value']:>9.4f} {r['stars']}"
        )
    print("=" * 90)
    print(f"Sample sizes: treated n = {n_t}, least-retail reference n = {n_r}")

    def _row(r: dict) -> str:
        spec = r["_spec"]
        return (
            rf"{r['label']} & "
            rf"${r['treated_mean']:{spec}}$ & "
            rf"${r['reference_mean']:{spec}}$ & "
            rf"${r['diff']:+{spec}}$ & "
            rf"${r['t_stat']:+.2f}$ & "
            rf"${r['p_value']:.4f}${r['stars']} \\"
        )

    score_row = _row(rows[0])
    match_rows = "\n".join(_row(r) for r in rows[1:])

    snippet = rf"""% Requires packages: booktabs, tabularx, array, caption
% Auto-generated by 06_Estimation_H3/04_h3_group_cells_and_scatter.py.
% Do not edit by hand; rerun the script to refresh.
\begin{{table}}[H]
\centering\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\caption{{Pre-window balance: treated vs least-retail reference.}}
\label{{tab:group-balance-reference}}
\begin{{tabularx}}{{\textwidth}}{{Xrrrrr}}
\toprule
Variable & Treated mean & Least-retail reference mean & Diff ($T - R$) & $t$-stat & $p$-value \\
\midrule
{score_row}
\midrule
{match_rows}
\bottomrule
\end{{tabularx}}
\caption*{{\footnotesize Note: Welch's two-sample $t$-tests of treated ($n = {n_t}$) versus least-retail reference ($n = {n_r}$) on pre-window (June 10 -- September 27, 2019) characteristics. The least-retail reference is the bottom {n_r} firms by retail score within the H3 candidate pool. Retail score is shown for reference; the four rows below it are the features used in the one-to-one nearest-neighbour match against the matched control (mean log total volume, mean dark share, intraday return volatility, median price), shown here on the treated vs reference contrast for comparability with Table~\ref{{tab:h3-balance}}. * $p < 0.10$, ** $p < 0.05$, *** $p < 0.01$.}}
\end{{table}}
"""

    LATEX_TABLE_REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    LATEX_TABLE_REF_OUT.write_text(snippet, encoding="utf-8")
    print(f"Saved: {LATEX_TABLE_REF_OUT}")


# Level-analogue 4-cell table (tab:group-cells)
def _collapse_stock_dark_share(ticker: str, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Return the stock's mean daily dark share over [t0, t1].

    Aggregates minute bars to daily totals (regular session only, June 10
    cutoff inclusive), computes the daily dark/total ratio, and averages
    across days. Returns NaN if the ticker has no usable minute bars in
    the window.
    """

    frame = load_minute_bar(ticker)
    frame = filter_session(
        frame,
        timestamp_col="timestamp",
        session_start=SESSION_START,
        session_end=SESSION_END,
    )
    frame = frame[(frame["timestamp"] >= t0) & (frame["timestamp"] <= t1)].copy()
    if frame.empty:
        return float("nan")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["dark"] = pd.to_numeric(frame["dark_volume"], errors="coerce").fillna(0.0)
    frame["lit"] = pd.to_numeric(frame["lit_volume"], errors="coerce").fillna(0.0)
    frame["tot"] = frame["dark"] + frame["lit"]
    daily = (
        frame.groupby("date", as_index=False)
        .agg(dark=("dark", "sum"), tot=("tot", "sum"))
    )
    daily = daily[daily["tot"] > 0]
    if daily.empty:
        return float("nan")
    return float((daily["dark"] / daily["tot"]).mean())


def _stock_dark_shares(tickers: list[str], t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Series:
    """Stock-level mean dark share across the window for each ticker."""

    return pd.Series({t: _collapse_stock_dark_share(t, t0, t1) for t in tickers},
                     name="dark_share").dropna()


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch t-test on two arrays; returns (diff, t_stat, p_value)."""

    res = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(res.statistic), float(res.pvalue)


def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def compute_and_write_levels_table() -> None:
    """Build the 4-cell (group, regime) mean-dark-share table.

    Each cell is the cross-sectional mean of stock-level mean daily dark
    shares in that (group, window). Margins report the Welch t-test on the
    treated-minus-matched gap within each window and on the post-minus-pre
    DiD across stock-level changes.
    """

    treated = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_treated_group.csv"
    )["ticker"].astype(str).tolist()
    matched = pd.read_csv(
        RETAIL_DIR / "group_outputs" / "retail_matched_control_group.csv"
    )["ticker"].astype(str).tolist()

    print("\nBuilding pre-window stock-level dark shares (treated)...")
    pre_T = _stock_dark_shares(treated, PRE_START, PRE_END)
    print("Building pre-window stock-level dark shares (matched control)...")
    pre_C = _stock_dark_shares(matched, PRE_START, PRE_END)
    print("Building post-window stock-level dark shares (treated)...")
    post_T = _stock_dark_shares(treated, POST_START, POST_END)
    print("Building post-window stock-level dark shares (matched control)...")
    post_C = _stock_dark_shares(matched, POST_START, POST_END)

    pre_diff, pre_t, pre_p = _welch(pre_T.to_numpy(), pre_C.to_numpy())
    post_diff, post_t, post_p = _welch(post_T.to_numpy(), post_C.to_numpy())

    common_T = pre_T.index.intersection(post_T.index)
    common_C = pre_C.index.intersection(post_C.index)
    dT = (post_T.loc[common_T] - pre_T.loc[common_T]).to_numpy()
    dC = (post_C.loc[common_C] - pre_C.loc[common_C]).to_numpy()
    did, did_t, did_p = _welch(dT, dC)

    n_t = len(pre_T)
    n_c = len(pre_C)

    # CSV with the underlying cell values and tests
    rows = [
        {"cell": "treated_pre",    "mean": float(pre_T.mean()),  "n": int(n_t)},
        {"cell": "treated_post",   "mean": float(post_T.mean()), "n": int(len(post_T))},
        {"cell": "matched_pre",    "mean": float(pre_C.mean()),  "n": int(n_c)},
        {"cell": "matched_post",   "mean": float(post_C.mean()), "n": int(len(post_C))},
        {"cell": "diff_pre",       "mean": pre_diff,  "n": int(min(n_t, n_c)),
         "t_stat": pre_t,  "p_value": pre_p},
        {"cell": "diff_post",      "mean": post_diff, "n": int(min(len(post_T), len(post_C))),
         "t_stat": post_t, "p_value": post_p},
        {"cell": "change_treated", "mean": float(dT.mean()), "n": int(len(dT))},
        {"cell": "change_matched", "mean": float(dC.mean()), "n": int(len(dC))},
        {"cell": "did",            "mean": did, "n": int(min(len(dT), len(dC))),
         "t_stat": did_t, "p_value": did_p},
    ]
    LEVELS_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LEVELS_CSV_OUT, index=False)
    print(f"\nSaved: {LEVELS_CSV_OUT}")

    print("\n" + "=" * 80)
    print("LEVEL-ANALOGUE 4-CELL TABLE (tab:group-cells)")
    print("=" * 80)
    print(f"{'':<22}{'Pre window':>14}{'Post window':>16}{'Post - Pre':>14}")
    print("-" * 80)
    print(f"{'Retail-treated':<22}{pre_T.mean():>14.4f}{post_T.mean():>16.4f}{dT.mean():>+14.4f}")
    print(f"{'Matched control':<22}{pre_C.mean():>14.4f}{post_C.mean():>16.4f}{dC.mean():>+14.4f}")
    print(f"{'Diff (T - C)':<22}{pre_diff:>+14.4f}{post_diff:>+16.4f}{did:>+14.4f}")
    print(f"{'  t-stat':<22}{pre_t:>+14.3f}{post_t:>+16.3f}{did_t:>+14.3f}")
    print(f"{'  p-value':<22}{pre_p:>14.4f}{post_p:>16.4f}{did_p:>14.4f}")
    print("=" * 80)

    snippet = rf"""% Requires packages: booktabs, tabularx, array, caption
% Auto-generated by 06_Estimation_H3/04_h3_group_cells_and_scatter.py.
% Do not edit by hand; rerun the script to refresh.
\begin{{table}}[H]
\centering\footnotesize
\setlength{{\tabcolsep}}{{6pt}}
\caption{{Mean dark share in the four (group, regime) cells.}}
\label{{tab:group-cells}}
\begin{{tabularx}}{{\textwidth}}{{Xrrr}}
\toprule
 & Pre window & Post window & Post $-$ Pre \\
\midrule
Retail-treated   & ${pre_T.mean():.4f}$ & ${post_T.mean():.4f}$ & ${dT.mean():+.4f}$ \\
Matched control  & ${pre_C.mean():.4f}$ & ${post_C.mean():.4f}$ & ${dC.mean():+.4f}$ \\
\midrule
Diff ($T - C$)   & ${pre_diff:+.4f}$ & ${post_diff:+.4f}$ & ${did:+.4f}$ \\
\quad $t$-stat   & ${pre_t:+.2f}$    & ${post_t:+.2f}$    & ${did_t:+.2f}$ \\
\quad $p$-value  & ${_fmt_p(pre_p)}$ & ${_fmt_p(post_p)}$ & ${_fmt_p(did_p)}$ \\
\bottomrule
\end{{tabularx}}
\caption*{{\footnotesize Note: Each cell is the cross-sectional mean of stock-level mean daily dark shares across $n = {n_t}$ treated and $n = {n_c}$ matched-control stocks. Pre window is June 10--September 30, 2019; post window is October 11, 2019--February 19, 2020. The diff row is the treated-minus-matched gap; the post-minus-pre column is the within-stock change. The bottom-right cell is the level-analogue difference-in-differences: the average within-stock dark-share change in the treated group minus the same in the matched-control group. $t$-statistics and $p$-values are from two-sided Welch tests on stock-level means (within-window cells) or stock-level changes (DiD).}}
\end{{table}}
"""

    LEVELS_LATEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    LEVELS_LATEX_OUT.write_text(snippet, encoding="utf-8")
    print(f"Saved: {LEVELS_LATEX_OUT}")

# Market-cap loader: average of daily market-cap across earnings JSONs
def load_average_market_cap() -> pd.Series:
    """Return the per-ticker average daily market cap.

    Walks every JSON in the Nasdaq earnings-calendar directory and
    averages each symbol's market-cap readings across the days on which
    it appears. Strings like ``"$31,516,812,833"`` are parsed to floats
    in USD; empty or non-numeric values are skipped.
    """

    records: dict[str, list[float]] = {}
    for path in sorted(MARKETCAP_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        data = payload.get("data") or {}
        rows = data.get("rows") or []
        for row in rows:
            symbol = str(row.get("symbol", "")).strip()
            raw = str(row.get("marketCap", "")).strip()
            if not symbol or not raw:
                continue
            cleaned = raw.replace("$", "").replace(",", "").strip()
            if not cleaned:
                continue
            try:
                value = float(cleaned)
            except ValueError:
                continue
            if value <= 0:
                continue
            records.setdefault(symbol, []).append(value)

    means = {sym: float(np.mean(vals)) for sym, vals in records.items() if vals}
    return pd.Series(means, name="avg_market_cap")


# Scatter figure: retail score vs average market cap
def plot_retail_score_vs_mktcap(
    score_table: pd.DataFrame,
    treated: list[str],
    matched: list[str],
    out_path: Path,
) -> None:
    """Scatter of retail-intensity score vs average market capitalisation.

    All scored firms appear as small grey dots. Treated and matched-control
    groups are highlighted, so a reader can see whether the retail score is
    capturing firm size or something distinct. Market cap is on a log scale
    because the cross-section spans many orders of magnitude.
    """

    treated_set = set(treated)
    matched_set = set(matched)

    mcap = load_average_market_cap()
    df = score_table[["ticker", "retail_score"]].dropna().copy()
    df["avg_market_cap"] = df["ticker"].map(mcap)
    df = df.dropna(subset=["avg_market_cap"])

    rest = df[~df["ticker"].isin(treated_set | matched_set)]
    t_df = df[df["ticker"].isin(treated_set)]
    c_df = df[df["ticker"].isin(matched_set)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.scatter(
        rest["avg_market_cap"],
        rest["retail_score"],
        s=14,
        color="0.75",
        linewidths=0,
        zorder=1,
        label="Other scored firms",
    )
    ax.scatter(
        c_df["avg_market_cap"],
        c_df["retail_score"],
        s=22,
        color="#2ca02c",
        marker="^",
        linewidths=0.4,
        edgecolors="#1a7d1a",
        zorder=3,
        label="Matched control",
    )
    ax.scatter(
        t_df["avg_market_cap"],
        t_df["retail_score"],
        s=26,
        color="#1f77b4",
        marker="o",
        linewidths=0.4,
        edgecolors="#0d4e80",
        zorder=4,
        label="Retail-treated",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Average market capitalisation (USD, log scale)", fontsize=10)
    ax.set_ylabel("Retail-intensity score", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"\nFigure saved: {out_path.with_suffix('.pdf')} "
        f"(n = {len(df)} firms with market-cap coverage)"
    )

# Scatter figure: retail score vs log trading volume
def plot_retail_score_vs_volume(
    score_table: pd.DataFrame,
    treated: list[str],
    matched: list[str],
    out_path: Path,
) -> None:
    """Scatter of retail-intensity score vs log mean-minute trading volume.

    All scored firms appear as small grey dots. Treated and matched-control
    groups are highlighted score is capturing trading-size or 
    something distinct.
    """

    treated_set = set(treated)
    matched_set = set(matched)

    df = score_table[["ticker", "retail_score", "pre_mean_log_total_volume"]].dropna().copy()

    rest = df[~df["ticker"].isin(treated_set | matched_set)]
    t_df = df[df["ticker"].isin(treated_set)]
    c_df = df[df["ticker"].isin(matched_set)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.scatter(
        rest["pre_mean_log_total_volume"],
        rest["retail_score"],
        s=14,
        color="0.75",
        linewidths=0,
        zorder=1,
        label="Other scored firms",
    )
    ax.scatter(
        c_df["pre_mean_log_total_volume"],
        c_df["retail_score"],
        s=22,
        color="#2ca02c",
        marker="^",
        linewidths=0.4,
        edgecolors="#1a7d1a",
        zorder=3,
        label=f"Matched control ($n={len(c_df)}$)",
    )
    ax.scatter(
        t_df["pre_mean_log_total_volume"],
        t_df["retail_score"],
        s=26,
        color="#1f77b4",
        marker="o",
        linewidths=0.4,
        edgecolors="#0d4e80",
        zorder=4,
        label=f"Retail-treated ($n={len(t_df)}$)",
    )

    ax.set_xlabel("Log mean-minute trading volume (pre-window)", fontsize=10)
    ax.set_ylabel("Retail-intensity score", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path.with_suffix('.pdf')}")

# Histogram figure: cross-sectional retail-score distribution
def plot_retail_score_histogram(
    score_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """Histogram of the pre-window retail-intensity score across scored firms."""

    scores = pd.to_numeric(score_table["retail_score"], errors="coerce").dropna()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    lo = float(scores.min())
    hi = float(scores.max())
    bin_width = 0.05
    edges = np.arange(np.floor(lo / bin_width) * bin_width,
                      np.ceil(hi / bin_width) * bin_width + bin_width,
                      bin_width)
    ax.hist(scores, bins=edges, color="#4c8cbf", edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Retail-intensity score (pre-October 2019)", fontsize=10)
    ax.set_ylabel("Number of firms", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path.with_suffix('.pdf')} (n = {len(scores)} firms)")

# Main
def main() -> None:
    treated, matched = load_groups()
    print(f"Groups loaded — treated: {len(treated)}, matched control: {len(matched)}")

    score_table = load_score_table()

    # --- Figures ---
    plot_retail_score_histogram(
        score_table,
        out_path=FIGURE_OUT / "retail_score_distribution",
    )
    plot_retail_score_vs_volume(
        score_table,
        treated,
        matched,
        out_path=FIGURE_OUT / "retail_score_vs_volume",
    )
    if MARKETCAP_DIR.exists() and any(MARKETCAP_DIR.glob("*.json")):
        plot_retail_score_vs_mktcap(
            score_table,
            treated,
            matched,
            out_path=FIGURE_OUT / "retail_score_vs_mktcap",
        )
    else:
        print(
            f"Skipping retail_score_vs_mktcap: {MARKETCAP_DIR} has no JSONs. "
            "Keeping existing figure on disk."
        )

    # Balance table (treated vs matched control on matching features)
    compute_and_write_balance_table()

    # Balance table (treated vs least-retail reference on matching features)
    compute_and_write_balance_table_vs_reference()

    # Level-analogue 4-cell table (tab:group-cells)
    compute_and_write_levels_table()


if __name__ == "__main__":
    main()

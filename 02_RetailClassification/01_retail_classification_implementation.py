"""Construct the retail-intensity score and the benchmark H3 groups.

See ``00_README.md`` (section "Classification Logic") for the
score definition, LOWESS estimation, winsorization, and matching
distance used by this script.

Outputs:
- group_outputs/retail_score_asset_table.csv
- group_outputs/preperiod_market_characteristics_pre_oct2019.csv
- group_outputs/retail_score_with_matching_features.csv
- group_outputs/retail_treated_group.csv
- group_outputs/retail_reference_group.csv
- group_outputs/retail_matched_control_group.csv
- group_outputs/retail_matched_pairs.csv
- group_outputs/retail_group_balance_summary.csv
- group_outputs/retail_group_balance_smd.csv
- group_outputs/retail_group_match_distance_summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from statsmodels.nonparametric.smoothers_lowess import lowess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "02_RetailClassification" / "data_clean"
MINUTE_BAR_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "minute_bars"
OUTPUT_DIR = PROJECT_ROOT / "02_RetailClassification" / "group_outputs"
UNIVERSE_FILE = PROJECT_ROOT / "01_Data_Pull" / "data_clean" / "sp500_tickers.csv"
EARNINGS_PANEL_PATH = (
    PROJECT_ROOT / "03_VARX_Data" / "data_clean" / "earnings_urgency_sparse_panel.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Pre-October 2019 cutoff. Filtering before differencing stops the
# holder-change series from bridging across the later regime break.
CUTOFF = pd.Timestamp("2019-10-01").tz_localize("US/Eastern")

# H1/H2 / H3 pre and post regime windows. 
PRE_WINDOW_START = pd.Timestamp("2019-06-10")
PRE_WINDOW_END = pd.Timestamp("2019-09-30 23:59:59")
POST_WINDOW_START = pd.Timestamp("2019-10-11")
POST_WINDOW_END = pd.Timestamp("2020-02-19 23:59:59")

# Tickers with known data irregularities removed from the score construction.
EXCLUDE_TICKERS = {"LEN", "STZ", "TAP", "MKC"}

# Score-construction settings.
MIN_OBS = 100
TRIM_VOLUME_QUANTILE = 0.99
TRIM_HOLDER_CHANGE_QUANTILE = 0.9999
LOWESS_FRAC = 0.20
LOWESS_IT = 1
LOWESS_SAMPLE_N = 100_000
RNG_SEED = 0
WINSOR_QUANTILE = 0.01
GROUP_SHARE = 0.20 # This gets overridden by H3_GROUP_SIZE if that is set.
H3_GROUP_SIZE = 70

# Matching settings.
MATCHING_SESSION_START = "09:31"
MATCHING_SESSION_END = "16:00"
MATCHING_CHARACTERISTICS_CACHE = (
    OUTPUT_DIR / "preperiod_market_characteristics_pre_oct2019.csv"
)
REBUILD_MATCHING_CHARACTERISTICS = False
MATCHING_FEATURES = [
    "pre_mean_log_total_volume",
    "pre_mean_dark_share",
    "pre_return_std",
    "pre_median_price",
]


def load_earnings_eligible_tickers() -> set[str]:
    """Load tickers with at least one active earnings-event row in both
    the pre and the post regime window.
    """

    frame = pd.read_csv(
        EARNINGS_PANEL_PATH,
        usecols=["asset", "timestamp_legacy_varx", "post_ea_1"],
    )
    frame["timestamp_legacy_varx"] = pd.to_datetime(
        frame["timestamp_legacy_varx"], errors="coerce"
    )
    frame["asset_normalized"] = (
        frame["asset"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    active = frame[frame["post_ea_1"] != 0]

    pre_assets = set(
        active.loc[
            (active["timestamp_legacy_varx"] >= PRE_WINDOW_START)
            & (active["timestamp_legacy_varx"] <= PRE_WINDOW_END),
            "asset_normalized",
        ]
    )
    post_assets = set(
        active.loc[
            (active["timestamp_legacy_varx"] >= POST_WINDOW_START)
            & (active["timestamp_legacy_varx"] <= POST_WINDOW_END),
            "asset_normalized",
        ]
    )
    return pre_assets & post_assets


def load_universe_tickers() -> set[str]:
    """Load the canonical constant-membership universe.

    Reads sp500_tickers.csv. The current count is reported by
    sp500_constant_sample_audit.csv.
    """

    return set(
        pd.read_csv(UNIVERSE_FILE)["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )


def build_preperiod_panel(universe_tickers: set[str]) -> pd.DataFrame:
    """Build the pre-period Robintrack-volume panel used in the score fit."""

    frames = []

    for file_name in sorted(os.listdir(DATA_CLEAN_DIR)):
        if not file_name.endswith("_rh_with_massive_volume.csv"):
            continue

        ticker = file_name.split("_", 1)[0].upper().replace(".", "-")
        if ticker not in universe_tickers or ticker in EXCLUDE_TICKERS:
            continue

        frame = pd.read_csv(DATA_CLEAN_DIR / file_name)
        required_cols = {"users_holding", "massive_volume", "adjusted_timestamp"}
        if not required_cols.issubset(frame.columns):
            continue

        frame["users_holding"] = pd.to_numeric(frame["users_holding"], errors="coerce")
        frame["massive_volume"] = pd.to_numeric(frame["massive_volume"], errors="coerce")
        frame["adjusted_timestamp"] = pd.to_datetime(
            frame["adjusted_timestamp"], errors="coerce", utc=True
        ).dt.tz_convert("US/Eastern")

        frame = frame[frame["adjusted_timestamp"] < CUTOFF].copy()
        frame = frame.sort_values("adjusted_timestamp").reset_index(drop=True)
        frame["abs_accounts_change"] = frame["users_holding"].diff().abs()
        frame["ticker"] = ticker

        keep_cols = ["ticker", "adjusted_timestamp", "abs_accounts_change", "massive_volume"]
        frame = frame[keep_cols].dropna()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "adjusted_timestamp", "abs_accounts_change", "massive_volume"]
        )

    panel = pd.concat(frames, ignore_index=True)
    print(f"Pre-period panel rows: {len(panel):,}")
    print(f"Pre-period panel tickers: {panel['ticker'].nunique()}")
    return panel


def prepare_fit_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """Trim and transform the pre-period panel for LOWESS fitting."""

    fit = panel[["ticker", "abs_accounts_change", "massive_volume"]].copy()
    fit["abs_accounts_change"] = pd.to_numeric(fit["abs_accounts_change"], errors="coerce")
    fit["massive_volume"] = pd.to_numeric(fit["massive_volume"], errors="coerce")
    fit = fit.dropna(subset=["ticker", "abs_accounts_change", "massive_volume"])
    fit = fit[fit["massive_volume"] > 0].copy()

    if fit.empty:
        raise ValueError("The retail-score fit panel is empty after filtering.")

    volume_cap = fit["massive_volume"].quantile(TRIM_VOLUME_QUANTILE)
    holder_change_cap = fit["abs_accounts_change"].quantile(
        TRIM_HOLDER_CHANGE_QUANTILE
    )
    fit = fit[
        (fit["massive_volume"] < volume_cap)
        & (fit["abs_accounts_change"] < holder_change_cap)
    ].copy()

    fit["is_active"] = (fit["abs_accounts_change"] > 0).astype(int)
    fit["x"] = np.log1p(fit["massive_volume"])
    fit["y"] = np.log1p(fit["abs_accounts_change"])

    print(f"LOWESS fit rows: {len(fit):,}")
    print(f"LOWESS fit tickers: {fit['ticker'].nunique()}")
    return fit


def fit_retail_score(fit: pd.DataFrame) -> pd.DataFrame:
    """Estimate the pooled LOWESS benchmark and aggregate ticker scores."""

    rng = np.random.default_rng(RNG_SEED)
    if len(fit) > LOWESS_SAMPLE_N:
        sample_index = np.sort(rng.choice(len(fit), size=LOWESS_SAMPLE_N, replace=False))
        fit_for_lowess = fit.iloc[sample_index].copy()
    else:
        fit_for_lowess = fit.copy()

    fit_for_lowess = fit_for_lowess.sort_values("x")
    lowess_fit = lowess(
        endog=fit_for_lowess["y"],
        exog=fit_for_lowess["x"],
        frac=LOWESS_FRAC,
        it=LOWESS_IT,
        return_sorted=True,
    )

    x_fit = lowess_fit[:, 0]
    y_fit = lowess_fit[:, 1]
    fit["yhat"] = np.interp(fit["x"], x_fit, y_fit)
    fit["retail_resid"] = fit["y"] - fit["yhat"]

    resid_lo = fit["retail_resid"].quantile(WINSOR_QUANTILE)
    resid_hi = fit["retail_resid"].quantile(1 - WINSOR_QUANTILE)
    fit["retail_resid_winsor"] = fit["retail_resid"].clip(
        lower=resid_lo, upper=resid_hi
    )

    asset_table = (
        fit.groupby("ticker")
        .agg(
            retail_score=("retail_resid_winsor", "mean"),
            retail_score_mean=("retail_resid", "mean"),
            retail_score_median=("retail_resid", "median"),
            retail_score_std=("retail_resid", "std"),
            active_share=("is_active", "mean"),
            n_obs=("retail_resid", "count"),
        )
        .reset_index()
    )

    asset_table = asset_table[asset_table["n_obs"] >= MIN_OBS].copy()
    asset_table = asset_table.sort_values(
        ["retail_score", "active_share", "ticker"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    asset_table["retail_rank"] = np.arange(1, len(asset_table) + 1)

    print(f"Scored tickers: {len(asset_table)}")
    return asset_table


def build_extreme_groups(
    asset_table: pd.DataFrame,
    n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build the treated group and the least-retail reference group.

    If ``n`` is supplied, it sets the explicit group size; otherwise the
    size falls back to ``ceil(len(asset_table) * GROUP_SHARE)``.
    """

    top_n = int(n) if n is not None else int(np.ceil(len(asset_table) * GROUP_SHARE))

    treated = (
        asset_table.sort_values(
            ["retail_score", "active_share", "ticker"],
            ascending=[False, False, True],
        )
        .head(top_n)
        .copy()
    )

    reference = (
        asset_table.sort_values(
            ["retail_score", "active_share", "ticker"],
            ascending=[True, True, True],
        )
        .head(top_n)
        .copy()
    )

    print(f"Treated group size: {len(treated)}")
    print(f"Reference group size: {len(reference)}")
    return treated, reference, top_n


def build_preperiod_market_characteristics(
    tickers: pd.Series,
    rebuild: bool = REBUILD_MATCHING_CHARACTERISTICS,
) -> pd.DataFrame:
    """Build the cacheable pre-period market-characteristics table."""

    if MATCHING_CHARACTERISTICS_CACHE.exists() and not rebuild:
        cached = pd.read_csv(MATCHING_CHARACTERISTICS_CACHE)
        cached["ticker"] = cached["ticker"].astype(str).str.upper()
        return cached

    rows = []

    for ticker in sorted(set(tickers)):
        file_candidates = [
            MINUTE_BAR_DIR / f"{ticker.replace('-', '.')}_1m_lit_dark.csv",
            MINUTE_BAR_DIR / f"{ticker}_1m_lit_dark.csv",
        ]
        minute_path = next((path for path in file_candidates if path.exists()), None)
        if minute_path is None:
            continue

        minute_data = pd.read_csv(
            minute_path,
            usecols=["timestamp", "dark_volume", "lit_volume", "dark_close", "lit_close"],
        )

        minute_data["ts_eastern"] = pd.to_datetime(
            minute_data["timestamp"], errors="coerce", utc=True
        ).dt.tz_convert("US/Eastern")
        minute_data = minute_data[minute_data["ts_eastern"] < CUTOFF].copy()

        hhmm = minute_data["ts_eastern"].dt.tz_localize(None).dt.strftime("%H:%M")
        minute_data = minute_data[
            (hhmm >= MATCHING_SESSION_START) & (hhmm <= MATCHING_SESSION_END)
        ].copy()
        if minute_data.empty:
            continue

        dark = pd.to_numeric(minute_data["dark_volume"], errors="coerce").fillna(0.0)
        lit = pd.to_numeric(minute_data["lit_volume"], errors="coerce").fillna(0.0)
        total = dark + lit

        close_px = pd.to_numeric(minute_data["lit_close"], errors="coerce").combine_first(
            pd.to_numeric(minute_data["dark_close"], errors="coerce")
        ).ffill()

        dark_share = np.where(total > 0, dark / total, np.nan)
        log_return = np.log(close_px.replace(0, np.nan)).diff()

        rows.append(
            {
                "ticker": ticker,
                "pre_mean_log_total_volume": (
                    np.log1p(total[total > 0]).mean() if (total > 0).any() else np.nan
                ),
                "pre_mean_dark_share": float(np.nanmean(dark_share)),
                "pre_std_dark_share": float(np.nanstd(dark_share, ddof=1)),
                "pre_return_std": float(log_return.std()),
                "pre_median_price": float(close_px.median()),
                "pre_minute_obs": int(len(minute_data)),
            }
        )

    characteristics = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    characteristics.to_csv(MATCHING_CHARACTERISTICS_CACHE, index=False)
    return characteristics


def optimal_match_without_replacement(
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    match_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match treated stocks to a non-treated control pool without replacement."""

    treated_sorted = treated_df.sort_values(
        ["retail_score", "active_share", "ticker"], ascending=[False, False, True]
    ).reset_index(drop=True)
    control_sorted = control_df.sort_values(
        ["retail_score", "active_share", "ticker"], ascending=[False, False, True]
    ).reset_index(drop=True)

    pooled = pd.concat([treated_sorted[match_cols], control_sorted[match_cols]], ignore_index=True)
    pooled_mean = pooled.mean()
    pooled_std = pooled.std().replace(0, 1.0)

    treated_z = ((treated_sorted[match_cols] - pooled_mean) / pooled_std).to_numpy(dtype=float)
    control_z = ((control_sorted[match_cols] - pooled_mean) / pooled_std).to_numpy(dtype=float)
    cost = np.sqrt(((treated_z[:, None, :] - control_z[None, :, :]) ** 2).sum(axis=2))

    treated_idx, control_idx = linear_sum_assignment(cost)
    pair_rows = []
    matched_rows = []

    for treated_position, control_position in sorted(
        zip(treated_idx, control_idx), key=lambda item: (item[0], item[1])
    ):
        treated_row = treated_sorted.iloc[treated_position]
        control_row = control_sorted.iloc[control_position]
        pair_rows.append(
            {
                "treated_ticker": treated_row["ticker"],
                "control_ticker": control_row["ticker"],
                "match_distance": float(cost[treated_position, control_position]),
                "treated_score": float(treated_row["retail_score"]),
                "control_score": float(control_row["retail_score"]),
                "score_gap": float(
                    treated_row["retail_score"] - control_row["retail_score"]
                ),
            }
        )
        matched_rows.append(control_row.to_dict())

    matched_pairs = pd.DataFrame(pair_rows)
    matched_controls = pd.DataFrame(matched_rows).sort_values(
        ["retail_score", "ticker"], ascending=[True, True]
    ).reset_index(drop=True)
    return matched_pairs, matched_controls


def standardized_mean_difference(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    column: str,
) -> float:
    """Compute the standardized mean difference for one balance feature."""

    var_a = group_a[column].var(ddof=1)
    var_b = group_b[column].var(ddof=1)
    pooled_sd = np.sqrt((var_a + var_b) / 2)
    if pd.isna(pooled_sd) or pooled_sd == 0:
        return np.nan
    return (group_a[column].mean() - group_b[column].mean()) / pooled_sd


def build_balance_tables(
    treated_group: pd.DataFrame,
    matched_control_group: pd.DataFrame,
    reference_group: pd.DataFrame,
    matched_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the balance outputs used later in H3."""

    balance_features = ["retail_score", "active_share", "n_obs", *MATCHING_FEATURES]

    balance_summary = pd.DataFrame(
        {
            "feature": balance_features,
            "treated_mean": [treated_group[col].mean() for col in balance_features],
            "matched_control_mean": [
                matched_control_group[col].mean() for col in balance_features
            ],
            "least_retail_reference_mean": [
                reference_group[col].mean() for col in balance_features
            ],
            "treated_median": [treated_group[col].median() for col in balance_features],
            "matched_control_median": [
                matched_control_group[col].median() for col in balance_features
            ],
            "least_retail_reference_median": [
                reference_group[col].median() for col in balance_features
            ],
        }
    )

    smd_table = pd.DataFrame(
        {
            "feature": balance_features,
            "treated_vs_matched_smd": [
                standardized_mean_difference(treated_group, matched_control_group, col)
                for col in balance_features
            ],
            "treated_vs_least_retail_smd": [
                standardized_mean_difference(treated_group, reference_group, col)
                for col in balance_features
            ],
        }
    )

    distance_summary = matched_pairs["match_distance"].describe().rename("value").to_frame()
    return balance_summary, smd_table, distance_summary


def main() -> None:
    """Run the retail-score construction and save the benchmark H3 groups."""

    universe_tickers = load_universe_tickers()
    panel = build_preperiod_panel(universe_tickers)
    fit = prepare_fit_dataset(panel)
    asset_table = fit_retail_score(fit)

    # Gate the H3 group selection on earnings eligibility so the treated,
    # matched-control, and least-retail reference groups stay constant
    # across the VIX, macro, and earnings families. 
    earnings_eligible = load_earnings_eligible_tickers()
    h3_candidate_table = asset_table[
        asset_table["ticker"].isin(earnings_eligible)
    ].copy()
    print(
        "H3 candidate pool (retail-scored intersect earnings in both windows): "
        f"{len(h3_candidate_table)} of {len(asset_table)} scored tickers."
    )

    treated_score_group, reference_score_group, top_n = build_extreme_groups(
        h3_candidate_table, n=H3_GROUP_SIZE
    )

    market_chars = build_preperiod_market_characteristics(asset_table["ticker"])

    match_table = asset_table.merge(market_chars, on="ticker", how="left")
    match_table = match_table.dropna(subset=MATCHING_FEATURES).copy()

    # The matching pool must also be restricted to earnings-eligible firms
    # so we never pair a treated firm with a control that would zero-pad
    # the earnings family.
    h3_match_table = match_table[
        match_table["ticker"].isin(earnings_eligible)
    ].copy()

    treated_group = h3_match_table[
        h3_match_table["ticker"].isin(treated_score_group["ticker"])
    ].copy()
    reference_group = h3_match_table[
        h3_match_table["ticker"].isin(reference_score_group["ticker"])
    ].copy()

    control_pool = h3_match_table[
        ~h3_match_table["ticker"].isin(treated_group["ticker"])
    ].copy()

    if len(treated_group) < top_n:
        raise ValueError(
            "The treated group lost names after merging market characteristics."
        )
    if len(reference_group) < top_n:
        raise ValueError(
            "The reference group lost names after merging market characteristics."
        )
    if len(control_pool) < len(treated_group):
        raise ValueError("The control pool is too small for one-to-one matching.")

    matched_pairs, matched_control_group = optimal_match_without_replacement(
        treated_group,
        control_pool,
        MATCHING_FEATURES,
    )
    matched_control_group["control_rank"] = np.arange(1, len(matched_control_group) + 1)

    balance_summary, smd_table, distance_summary = build_balance_tables(
        treated_group,
        matched_control_group,
        reference_group,
        matched_pairs,
    )

    asset_table.to_csv(OUTPUT_DIR / "retail_score_asset_table.csv", index=False)
    market_chars.to_csv(OUTPUT_DIR / "preperiod_market_characteristics_pre_oct2019.csv", index=False)
    match_table.to_csv(OUTPUT_DIR / "retail_score_with_matching_features.csv", index=False)
    treated_group.to_csv(OUTPUT_DIR / "retail_treated_group.csv", index=False)
    reference_group.to_csv(OUTPUT_DIR / "retail_reference_group.csv", index=False)
    matched_control_group.to_csv(OUTPUT_DIR / "retail_matched_control_group.csv", index=False)
    matched_pairs.to_csv(OUTPUT_DIR / "retail_matched_pairs.csv", index=False)
    balance_summary.to_csv(OUTPUT_DIR / "retail_group_balance_summary.csv", index=False)
    smd_table.to_csv(OUTPUT_DIR / "retail_group_balance_smd.csv", index=False)
    distance_summary.to_csv(OUTPUT_DIR / "retail_group_match_distance_summary.csv")

    print(f"Saved score table for {len(asset_table)} tickers.")
    print(f"Treated group size: {len(treated_group)}")
    print(f"Matched control size: {len(matched_control_group)}")
    print(f"Reference group size: {len(reference_group)}")
    print("Saved benchmark H3 group-construction outputs to group_outputs/.")


if __name__ == "__main__":
    main()

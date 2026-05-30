"""Run the first H3 task: pre-trend and comparability diagnostics.

This scripts job is to decide whether the current treated and 
matched-control groups are credible enough to support the later 
treatment-control design.
"""

from __future__ import annotations

import importlib

_config = importlib.import_module("01_h3_config")
_diag = importlib.import_module("02_h3_pretrend_diagnostics")


def main() -> None:
    """Run the H3 pre-period diagnostics and save the outputs."""

    _diag.ensure_output_dirs()

    groups = _diag.load_group_definitions()
    score_table = _diag.load_score_table()

    print("Building pre-period stock-day panel for the H3 benchmark groups...")
    stock_day = _diag.build_pretrend_stock_day_panel(groups)
    daily = _diag.build_group_series(stock_day, frequency="daily")
    weekly = _diag.build_group_series(stock_day, frequency="weekly")
    gaps_daily = _diag.build_gap_series(daily)
    gaps_weekly = _diag.build_gap_series(weekly)

    sample_summary = _diag.build_sample_summary(stock_day)
    comparability = _diag.build_comparability_summary(stock_day)
    trend_tests = _diag.build_trend_test_summary(stock_day)
    placebo_did = _diag.build_placebo_did_summary(stock_day)

    # Save the core diagnostic tables so the review notebook can read them.
    stock_day.to_csv(_config.TABLE_DIR / "h3_pretrend_stock_day_panel.csv", index=False)
    daily.to_csv(_config.TABLE_DIR / "h3_pretrend_daily_group_series.csv", index=False)
    weekly.to_csv(_config.TABLE_DIR / "h3_pretrend_weekly_group_series.csv", index=False)
    gaps_daily.to_csv(_config.TABLE_DIR / "h3_pretrend_daily_gap_series.csv", index=False)
    gaps_weekly.to_csv(_config.TABLE_DIR / "h3_pretrend_weekly_gap_series.csv", index=False)
    sample_summary.to_csv(_config.TABLE_DIR / "h3_pretrend_sample_summary.csv", index=False)
    comparability.to_csv(_config.TABLE_DIR / "h3_pretrend_comparability_summary.csv", index=False)
    trend_tests.to_csv(_config.TABLE_DIR / "h3_pretrend_trend_tests.csv", index=False)
    placebo_did.to_csv(_config.TABLE_DIR / "h3_pretrend_placebo_did_summary.csv", index=False)
    score_table.to_csv(_config.TABLE_DIR / "h3_retail_score_asset_table_snapshot.csv", index=False)

    # Save a small number of easy-to-read plots.
    _diag.plot_group_series(
        daily,
        frequency="daily",
        path=_config.FIGURE_DIR / "h3_pretrend_daily_group_series.png",
    )
    _diag.plot_group_series(
        weekly,
        frequency="weekly",
        path=_config.FIGURE_DIR / "h3_pretrend_weekly_group_series.png",
    )
    _diag.plot_gap_series(
        gaps_daily,
        frequency="daily",
        path=_config.FIGURE_DIR / "h3_pretrend_daily_gap_series.png",
    )
    _diag.plot_gap_series(
        gaps_weekly,
        frequency="weekly",
        path=_config.FIGURE_DIR / "h3_pretrend_weekly_gap_series.png",
    )

    print("\nH3 pre-trend sample summary")
    print(sample_summary.to_string(index=False))
    print("\nH3 comparability summary")
    print(comparability.to_string(index=False))
    print("\nH3 linear pre-trend tests")
    print(trend_tests.to_string(index=False))
    print("\nH3 placebo DiD summary")
    print(placebo_did.to_string(index=False))


if __name__ == "__main__":
    main()

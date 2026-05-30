"""Trend-controlled robustness pass for the H3 benchmark."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

H1H2_DIR = H3_DIR.parents[0] / "05_Estimation_H1_H2"
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

_config = importlib.import_module("01_h3_config")
_robust = importlib.import_module("10_h3_robustness")
_h1h2 = importlib.import_module("02_estimation_h1_h2")
_benchmark = importlib.import_module("05_h3_estimation")


FAMILY_NAMES = _config.FAMILY_NAMES
PRE_PERIOD_START = pd.Timestamp(_config.PRE_PERIOD_START)
PRESENTATION_TABLE_DIR = _config.PRESENTATION_TABLE_DIR

FAMILY_SPECS = _h1h2.FAMILY_SPECS
FamilySpec = _h1h2.FamilySpec

ESTIMATION_OUTPUT_DIR = _config.ESTIMATION_OUTPUT_DIR
run_family_h3 = _benchmark.run_family_h3
build_run_summary = _benchmark.build_run_summary
build_key_summary = _benchmark.build_key_summary

_save_group_outputs = _robust._save_group_outputs
_save_h3_outputs = _robust._save_h3_outputs
_save_run_bundle_helper = _robust._save_run_bundle
_merge_with_benchmark = _robust._merge_with_benchmark


ROBUSTNESS_TREND_DIR: Path = _config.H3_DIR / "output" / "robustness_trend"
ROBUSTNESS_N_DRAWS = 5000
TREND_COL_NAME = "day_index"


def _trend_value(timestamps: pd.Series) -> pd.Series:
    """Calendar days since the pre-period anchor.

    Using a fixed anchor means the pre-regime fit sees ``day_index`` 
    roughly in [0, 110] and the post-regime fit sees it in [120, 260]. 
    The level offset is irrelevant after the within-stock demeaning 
    the VARX applies, so only the slope (= linear trend) is 
    identified.
    """

    return (pd.to_datetime(timestamps).dt.normalize() - PRE_PERIOD_START).dt.days.astype(float)


def _wrap_iterator(
    base_factory: Callable[[Iterable[str] | None], Iterator[pd.DataFrame]],
) -> Callable[[Iterable[str] | None], Iterator[pd.DataFrame]]:
    """Return an iterator factory that injects ``day_index`` per piece."""

    def factory(tickers: Iterable[str] | None) -> Iterator[pd.DataFrame]:
        for piece in base_factory(tickers):
            piece = piece.copy()
            piece[TREND_COL_NAME] = _trend_value(piece["timestamp"])
            yield piece

    return factory


def build_trend_family_spec(family_name: str) -> FamilySpec:
    """Return a copy of a family spec with ``day_index`` injected."""

    base = FAMILY_SPECS[family_name]
    return FamilySpec(
        name=base.name,
        common_x_cols=(TREND_COL_NAME, *base.common_x_cols),
        panel_x_cols=base.panel_x_cols,
        horizon_end=base.horizon_end,
        iterator_factory=_wrap_iterator(base.iterator_factory),
        shock_builder=base.shock_builder,
    )


def ensure_output_dirs() -> None:
    """Create the trend-robustness output folder."""

    ROBUSTNESS_TREND_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_trend_robustness() -> list[dict[str, object]]:
    """Run the trend-controlled H3 robustness pass."""

    results: list[dict[str, object]] = []
    for family_name in FAMILY_NAMES:
        print(f"  trend robustness: estimating {family_name}...", flush=True)
        trend_spec = build_trend_family_spec(family_name)
        family_result = run_family_h3(
            family_name,
            treated_group="treated",
            control_group="matched_control",
            n_draws=ROBUSTNESS_N_DRAWS,
            p_lags=_config.BENCHMARK_P_LAGS,
            family_spec_override=trend_spec,
        )
        _save_group_outputs(
            family_result, output_dir=ROBUSTNESS_TREND_DIR, prefix="h3_trend"
        )
        _save_h3_outputs(
            family_result, output_dir=ROBUSTNESS_TREND_DIR, prefix="h3_trend"
        )
        results.append(family_result)
    return results


def save_trend_outputs(results: list[dict[str, object]]) -> dict[str, Path]:
    """Save the compact trend-robustness summaries and comparison table."""

    _, key_path = _save_run_bundle_helper(
        results,
        output_dir=ROBUSTNESS_TREND_DIR,
        run_name="h3_trend",
    )
    key_summary = pd.read_csv(key_path)
    merged = _merge_with_benchmark(key_summary, robustness_label="trend")
    comparison_path = ROBUSTNESS_TREND_DIR / "h3_trend_vs_benchmark_comparison.csv"
    merged.to_csv(comparison_path, index=False)
    return {
        "trend_key_summary": key_path,
        "trend_comparison": comparison_path,
    }

"""Trend-controlled robustness pass for H1/H2 (family-level).

Adds a regime-specific linear ``day_index`` covariate to the VARX 
exogenous block before the impulse response is constructed. If the 
post-minus-pre IRF shifts materially after partialling out the trend, 
the headline reading has to be tempered; if it does not, the reading 
is not a slow-drift artefact.

This is the family-level analogue of
06_Estimation_H3/18_h3_trend_robustness.py. The H3 version applies the 
trend control inside the treated/matched-control fits; the H1/H2 
version applies it inside the family-level fit on the full 
constant-membership panel.

The runner script forwards a modified ``FamilySpec`` (with 
``day_index`` prepended to ``common_x_cols`` and the iterator factory 
wrapped to inject the column on each piece) to ``run_family_h1_h2`` 
via the ``family_spec_override`` hook.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd


H1H2_DIR = Path(__file__).resolve().parent
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

H3_DIR = H1H2_DIR.parents[0] / "06_Estimation_H3"
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

_h1h2 = importlib.import_module("02_estimation_h1_h2")
_h1h2_config = importlib.import_module("01_estimation_config")
_h3_config = importlib.import_module("01_h3_config")


FAMILY_NAMES = _h1h2_config.FAMILY_NAMES
FAMILY_SPECS = _h1h2.FAMILY_SPECS
FamilySpec = _h1h2.FamilySpec
run_family_h1_h2 = _h1h2.run_family_h1_h2

PRE_PERIOD_START = pd.Timestamp(_h3_config.PRE_PERIOD_START)
TREND_COL_NAME = "day_index"

OUTPUT_DIR: Path = H1H2_DIR / "output" / "robustness_trend"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    def factory(tickers: Iterable[str] | None) -> Iterator[pd.DataFrame]:
        for piece in base_factory(tickers):
            piece = piece.copy()
            piece[TREND_COL_NAME] = _trend_value(piece["timestamp"])
            yield piece

    return factory


def build_trend_family_spec(family_name: str) -> FamilySpec:
    base = FAMILY_SPECS[family_name]
    return FamilySpec(
        name=base.name,
        common_x_cols=(TREND_COL_NAME, *base.common_x_cols),
        panel_x_cols=base.panel_x_cols,
        horizon_end=base.horizon_end,
        iterator_factory=_wrap_iterator(base.iterator_factory),
        shock_builder=base.shock_builder,
    )


def run_family_trend_robustness(
    family_name: str,
    *,
    n_draws: int,
) -> dict[str, object]:
    """Run the trend-controlled H1/H2 estimation for one family."""

    trend_spec = build_trend_family_spec(family_name)
    return run_family_h1_h2(
        family_name,
        n_draws=n_draws,
        family_spec_override=trend_spec,
    )

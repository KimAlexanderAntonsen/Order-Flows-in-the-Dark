"""Within-pre IRF stability test for H1/H2 (family-level).

Splits the pre period at 2019-08-07 (the same date used by the H3 
placebo DiD in 02_h3_pretrend_diagnostics.py) and re-fits the 
family-level VARX on each half. The output is the within-pre drift IRF

    drift_family = IRF(pre_b)_family - IRF(pre_a)_family

with bootstrap bands. If drift covers zero at every relevant horizon 
the H2 verdict is not confounded by within-pre IRF drift; if it 
excludes zero at the same horizons where the H2 post-minus-pre 
rejects, the headline reading is partly pre-existing drift.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


H1H2_DIR = Path(__file__).resolve().parent
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

H3_DIR = H1H2_DIR.parents[0] / "06_Estimation_H3"
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

_h1h2 = importlib.import_module("02_estimation_h1_h2")
_h1h2_config = importlib.import_module("01_estimation_config")
_h3_pre_stability = importlib.import_module("16_h3_pre_stability")
_h3_config = importlib.import_module("01_h3_config")


FAMILY_NAMES = _h1h2_config.FAMILY_NAMES
load_sp500_universe = _h1h2.load_sp500_universe

PLACEBO_SPLIT_DATE = pd.Timestamp(_h3_config.PLACEBO_SPLIT_DATE)
PRE_A_WINDOW = _h3_pre_stability.PRE_A_WINDOW
PRE_B_WINDOW = _h3_pre_stability.PRE_B_WINDOW
BENCHMARK_P_LAGS = _h3_pre_stability.BENCHMARK_P_LAGS

OUTPUT_DIR: Path = H1H2_DIR / "output" / "pre_stability"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_family_pre_stability(
    family_name: str,
    *,
    tickers: Iterable[str],
    n_draws: int,
) -> dict[str, object]:
    """Family-level within-pre stability for one urgency family.

    Reuses ``_fit_group_pre_halves`` from the H3 layer with the full 
    panel. The returned ``drift_bands`` per shock is the within-pre 
    placebo at the family level.
    """

    fitted = _h3_pre_stability._fit_group_pre_halves(
        family_name,
        group_name="full_panel",
        tickers=tickers,
        n_draws=n_draws,
        seed_offset=0,
    )

    placebo_outputs: list[dict[str, object]] = []
    for shock in fitted["shock_outputs"]:
        shock_name = str(shock["shock_name"])
        drift_bands = shock["drift_bands"].copy()
        drift_bands.insert(0, "regime", "pre_b_minus_pre_a")
        drift_bands.insert(0, "group", "full_panel")
        drift_bands.insert(0, "shock_name", shock_name)
        drift_bands.insert(0, "family", family_name)

        pre_a_bands = shock["pre_a_bands"].copy()
        pre_a_bands.insert(0, "regime", "pre_a")
        pre_a_bands.insert(0, "group", "full_panel")
        pre_a_bands.insert(0, "shock_name", shock_name)
        pre_a_bands.insert(0, "family", family_name)

        pre_b_bands = shock["pre_b_bands"].copy()
        pre_b_bands.insert(0, "regime", "pre_b")
        pre_b_bands.insert(0, "group", "full_panel")
        pre_b_bands.insert(0, "shock_name", shock_name)
        pre_b_bands.insert(0, "family", family_name)

        placebo_outputs.append(
            {
                "shock_name": shock_name,
                "pre_a_bands": pre_a_bands,
                "pre_b_bands": pre_b_bands,
                "drift_bands": drift_bands,
            }
        )

    return {
        "family": family_name,
        "placebo_outputs": placebo_outputs,
    }


def _key_horizons_for_family(family_name: str) -> list[int]:
    if family_name == "vix":
        return [0, 1, 2, 3, 4, 5]
    if family_name == "macro":
        return [-1, 0, 1, 2, 3, 4]
    if family_name == "earnings":
        return [30 * k for k in range(1, 14)]
    raise ValueError(f"Unknown family: {family_name}")


def build_key_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    """Compact per-horizon summary of the family-level within-pre drift."""

    rows: list[dict[str, object]] = []
    for family_result in results:
        family_name = str(family_result["family"])
        for placebo_output in family_result["placebo_outputs"]:
            shock_name = str(placebo_output["shock_name"])
            drift_bands = placebo_output["drift_bands"]
            horizons = set(drift_bands["horizon"].astype(int).tolist())
            for horizon in _key_horizons_for_family(family_name):
                if horizon not in horizons:
                    continue
                row = drift_bands.loc[drift_bands["horizon"] == horizon].iloc[0]
                horizon_label = str(horizon)
                if family_name == "earnings":
                    horizon_label = f"block_{horizon // 30}"
                lower = float(row["dark_share_change_bps_lower95"])
                upper = float(row["dark_share_change_bps_upper95"])
                rows.append(
                    {
                        "family": family_name,
                        "shock_name": shock_name,
                        "group": "full_panel",
                        "horizon": int(horizon),
                        "horizon_label": horizon_label,
                        "dark_share_change_bps_point": float(row["dark_share_change_bps_point"]),
                        "dark_share_change_bps_lower95": lower,
                        "dark_share_change_bps_upper95": upper,
                        "exclude_zero": bool((lower > 0.0) or (upper < 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def build_run_summary(results: list[dict[str, object]], *, n_draws: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_result in results:
        family_name = str(family_result["family"])
        for placebo_output in family_result["placebo_outputs"]:
            rows.append(
                {
                    "family": family_name,
                    "shock_name": str(placebo_output["shock_name"]),
                    "group": "full_panel",
                    "pre_a_start": PRE_A_WINDOW.start,
                    "pre_a_end": PRE_A_WINDOW.end,
                    "pre_b_start": PRE_B_WINDOW.start,
                    "pre_b_end": PRE_B_WINDOW.end,
                    "n_draws": int(n_draws),
                    "p_lags": int(BENCHMARK_P_LAGS),
                }
            )
    return pd.DataFrame(rows)

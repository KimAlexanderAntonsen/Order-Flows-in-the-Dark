"""Within-pre IRF stability test for H3.

The benchmark H3 design assumes that the regime-break difference
(IRF_post - IRF_pre) is comparable across the treated and 
matched-control groups. The level-based pretrend diagnostics in files
02-04 show mixed evidence of a level pretrend on dark share: the
linear trend does not reject at 5% (p = 0.067) but the placebo DiD
at the 2019-08-07 split does (p = 0.024). This tempers the benchmark
reading but does not test the actual counterfactual on IRFs. 
This module provides that IRF-level check.

We split the pre period at the same placebo split date used in the 
level diagnostics (2019-08-07). Fit the family-specific VARX 
separately in each pre half for each group. Build the IRF drift

    drift_group = IRF(pre_B)_group - IRF(pre_A)_group

for treated and for matched control, and the difference in difference placebo

    placebo_drift = drift_treated - drift_control

If `placebo_drift` is close to zero at the key H3 horizons, the 
IRF-level parallel-trends assumption is credible even though the 
level-based placebo DiD rejects. If `placebo_drift` is similar in sign 
and magnitude to the benchmark H3 estimate, a non-trivial share of the 
benchmark reading could be pre-trend rather than a genuine regime 
response.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

H1H2_DIR = H3_DIR.parents[0] / "05_Estimation_H1_H2"
if str(H1H2_DIR) not in sys.path:
    sys.path.insert(0, str(H1H2_DIR))

_config = importlib.import_module("01_h3_config")
_diagnostics = importlib.import_module("02_h3_pretrend_diagnostics")
_h1h2 = importlib.import_module("02_estimation_h1_h2")
_estimation_config = importlib.import_module("01_estimation_config")


BENCHMARK_P_LAGS = _config.BENCHMARK_P_LAGS
DEFAULT_RIDGE = _config.DEFAULT_RIDGE
BASE_SEED = _config.BASE_SEED
FAMILY_NAMES = _config.FAMILY_NAMES
PLACEBO_SPLIT_DATE = pd.Timestamp(_config.PLACEBO_SPLIT_DATE)
PRE_PERIOD_START = pd.Timestamp(_config.PRE_PERIOD_START)
PRE_PERIOD_END = pd.Timestamp(_config.PRE_PERIOD_END)

load_group_definitions = _diagnostics.load_group_definitions

FAMILY_SPECS = _h1h2.FAMILY_SPECS
fit_family_regime = _h1h2.fit_family_regime
estimate_regime_volume_means = _h1h2.estimate_regime_volume_means
simulate_difference_bands = _h1h2.simulate_difference_bands
_extract_point_frame = _h1h2._extract_point_frame
_simulate_regime_bands_from_shared_draws = _h1h2._simulate_regime_bands_from_shared_draws

RegimeWindow = _estimation_config.RegimeWindow


# Split the pre period at the same date used in the level-based 
# placebo DiD (file 02_h3_pretrend_diagnostics.py). This keeps the IRF 
# stability test and the placebo DiD on the same clock.
PRE_A_WINDOW = RegimeWindow(
    name="pre_a",
    start=PRE_PERIOD_START.isoformat(sep=" "),
    end=(PLACEBO_SPLIT_DATE - pd.Timedelta(minutes=1)).isoformat(sep=" "),
)
PRE_B_WINDOW = RegimeWindow(
    name="pre_b",
    start=PLACEBO_SPLIT_DATE.isoformat(sep=" "),
    end=PRE_PERIOD_END.isoformat(sep=" "),
)


def _key_horizons_for_family(family_name: str) -> list[int]:
    """Match the key-horizon selection used in the benchmark H3 summary."""

    if family_name == "vix":
        return [0, 1, 2, 3, 4, 5]
    if family_name == "macro":
        return [-1, 0, 1, 2, 3, 4]
    if family_name == "earnings":
        return [30 * k for k in range(1, 14)]
    raise ValueError(f"Unknown family: {family_name}")


def _fit_group_pre_halves(
    family_name: str,
    *,
    group_name: str,
    tickers: Iterable[str],
    n_draws: int,
    seed_offset: int,
) -> dict[str, object]:
    """Fit pre_A and pre_B VARXs for one group and build the within-pre drift."""

    family = FAMILY_SPECS[family_name]

    pre_a_result = fit_family_regime(
        family,
        tickers=tickers,
        window=PRE_A_WINDOW,
        p_lags=BENCHMARK_P_LAGS,
        ridge=DEFAULT_RIDGE,
    )
    pre_b_result = fit_family_regime(
        family,
        tickers=tickers,
        window=PRE_B_WINDOW,
        p_lags=BENCHMARK_P_LAGS,
        ridge=DEFAULT_RIDGE,
    )

    pre_a_volume_means = estimate_regime_volume_means(
        family, tickers=tickers, window=PRE_A_WINDOW
    )
    pre_b_volume_means = estimate_regime_volume_means(
        family, tickers=tickers, window=PRE_B_WINDOW
    )

    shock_outputs: list[dict[str, object]] = []
    for shock_index, (shock_name, shock_path) in enumerate(family.shock_builder()):
        pre_a_bands, _pre_a_diag, pre_a_draws = _simulate_regime_bands_from_shared_draws(
            base_result=pre_a_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=pre_a_volume_means["dark_volume_mean"],
            lit_volume_mean=pre_a_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + seed_offset + 100 * shock_index,
        )
        pre_b_bands, _pre_b_diag, pre_b_draws = _simulate_regime_bands_from_shared_draws(
            base_result=pre_b_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=pre_b_volume_means["dark_volume_mean"],
            lit_volume_mean=pre_b_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + seed_offset + 1000 + 100 * shock_index,
        )

        drift_bands = simulate_difference_bands(
            point_pre=_extract_point_frame(pre_a_bands),
            point_post=_extract_point_frame(pre_b_bands),
            pre_draw_irfs=pre_a_draws,
            post_draw_irfs=pre_b_draws,
        )
        drift_draws: list[pd.DataFrame] = []
        for draw_id, (a_draw, b_draw) in enumerate(
            zip(pre_a_draws, pre_b_draws, strict=True), start=1
        ):
            a_values = a_draw.drop(columns=["draw_id"], errors="ignore")
            b_values = b_draw.drop(columns=["draw_id"], errors="ignore")
            diff = _h1h2._subtract_irf_frames(b_values, a_values)
            diff.insert(0, "draw_id", draw_id)
            drift_draws.append(diff)

        shock_outputs.append(
            {
                "shock_name": shock_name,
                "pre_a_bands": pre_a_bands,
                "pre_b_bands": pre_b_bands,
                "drift_bands": drift_bands,
                "drift_draws": drift_draws,
            }
        )

    return {
        "group": group_name,
        "tickers": list(tickers),
        "shock_outputs": shock_outputs,
    }


def _add_identifiers(
    frame: pd.DataFrame,
    *,
    family_name: str,
    shock_name: str,
    group_name: str,
    regime_name: str,
) -> pd.DataFrame:
    """Attach identifier columns to a bands frame for saving."""

    out = frame.copy()
    out.insert(0, "regime", regime_name)
    out.insert(0, "group", group_name)
    out.insert(0, "shock_name", shock_name)
    out.insert(0, "family", family_name)
    return out


def run_family_pre_stability(
    family_name: str,
    *,
    treated_group: str = "treated",
    control_group: str = "matched_control",
    n_draws: int,
) -> dict[str, object]:
    """Run the within-pre IRF stability test for one family."""

    group_definitions = load_group_definitions()

    treated_result = _fit_group_pre_halves(
        family_name,
        group_name=treated_group,
        tickers=group_definitions[treated_group],
        n_draws=n_draws,
        seed_offset=0,
    )
    control_result = _fit_group_pre_halves(
        family_name,
        group_name=control_group,
        tickers=group_definitions[control_group],
        n_draws=n_draws,
        seed_offset=3000,
    )

    placebo_outputs: list[dict[str, object]] = []
    for treated_shock, control_shock in zip(
        treated_result["shock_outputs"],
        control_result["shock_outputs"],
        strict=True,
    ):
        shock_name = str(treated_shock["shock_name"])
        placebo_bands = simulate_difference_bands(
            point_pre=_extract_point_frame(control_shock["drift_bands"]),
            point_post=_extract_point_frame(treated_shock["drift_bands"]),
            pre_draw_irfs=control_shock["drift_draws"],
            post_draw_irfs=treated_shock["drift_draws"],
        )

        placebo_outputs.append(
            {
                "shock_name": shock_name,
                "treated_drift_bands": _add_identifiers(
                    treated_shock["drift_bands"],
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=treated_group,
                    regime_name="pre_b_minus_pre_a",
                ),
                "control_drift_bands": _add_identifiers(
                    control_shock["drift_bands"],
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=control_group,
                    regime_name="pre_b_minus_pre_a",
                ),
                "placebo_bands": _add_identifiers(
                    placebo_bands,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=f"{treated_group}_minus_{control_group}",
                    regime_name="within_pre_placebo",
                ),
            }
        )

    return {
        "family": family_name,
        "treated_group": treated_group,
        "control_group": control_group,
        "placebo_outputs": placebo_outputs,
    }


def _summarize_drift(
    family_name: str,
    shock_name: str,
    *,
    group_label: str,
    drift_bands: pd.DataFrame,
) -> pd.DataFrame:
    """Compact per-horizon summary of a within-pre drift object."""

    rows: list[dict[str, object]] = []
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
                "group": group_label,
                "horizon": int(horizon),
                "horizon_label": horizon_label,
                "dark_share_change_bps_point": float(row["dark_share_change_bps_point"]),
                "dark_share_change_bps_lower95": lower,
                "dark_share_change_bps_upper95": upper,
                "exclude_zero": bool((lower > 0.0) or (upper < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_key_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    """Stack the compact drift summary across families and groups."""

    tables: list[pd.DataFrame] = []
    for family_result in results:
        family_name = str(family_result["family"])
        treated_group = str(family_result["treated_group"])
        control_group = str(family_result["control_group"])
        for placebo_output in family_result["placebo_outputs"]:
            shock_name = str(placebo_output["shock_name"])
            tables.append(
                _summarize_drift(
                    family_name,
                    shock_name,
                    group_label=treated_group,
                    drift_bands=placebo_output["treated_drift_bands"],
                )
            )
            tables.append(
                _summarize_drift(
                    family_name,
                    shock_name,
                    group_label=control_group,
                    drift_bands=placebo_output["control_drift_bands"],
                )
            )
            tables.append(
                _summarize_drift(
                    family_name,
                    shock_name,
                    group_label=f"{treated_group}_minus_{control_group}",
                    drift_bands=placebo_output["placebo_bands"],
                )
            )

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def build_run_summary(results: list[dict[str, object]], *, n_draws: int) -> pd.DataFrame:
    """Compact identifier table describing the within-pre stability run."""

    rows: list[dict[str, object]] = []
    for family_result in results:
        family_name = str(family_result["family"])
        for placebo_output in family_result["placebo_outputs"]:
            rows.append(
                {
                    "family": family_name,
                    "shock_name": str(placebo_output["shock_name"]),
                    "treated_group": str(family_result["treated_group"]),
                    "control_group": str(family_result["control_group"]),
                    "pre_a_start": PRE_A_WINDOW.start,
                    "pre_a_end": PRE_A_WINDOW.end,
                    "pre_b_start": PRE_B_WINDOW.start,
                    "pre_b_end": PRE_B_WINDOW.end,
                    "n_draws": int(n_draws),
                    "p_lags": int(BENCHMARK_P_LAGS),
                }
            )
    return pd.DataFrame(rows)

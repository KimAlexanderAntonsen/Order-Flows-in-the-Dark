"""Benchmark H3 estimation on top of the finished H1/H2 and beta 
layers.

The goal of this module is:

1. Reuse the finished regime-estimation logic from H1/H2.
2. Estimate the pre/post model separately for the treated and 
   matched-control groups.
3. Combine those objects into the benchmark H3 estimand:

       (treated post - treated pre) - (control post - control pre)

We are extending the existing VARX workflow to a treatment-control 
comparison, which keeps the H3 layer easy to understand.
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


BENCHMARK_GROUPS = _config.BENCHMARK_GROUPS
BENCHMARK_P_LAGS = _config.BENCHMARK_P_LAGS
DEFAULT_RIDGE = _config.DEFAULT_RIDGE
BASE_SEED = _config.BASE_SEED
FAMILY_NAMES = _config.FAMILY_NAMES
N_SIMULATION_DRAWS = _config.N_SIMULATION_DRAWS

load_group_definitions = _diagnostics.load_group_definitions

FAMILY_SPECS = _h1h2.FAMILY_SPECS
PRE_WINDOW = _h1h2._local.PRE_WINDOW
POST_WINDOW = _h1h2._local.POST_WINDOW

fit_family_regime = _h1h2.fit_family_regime
estimate_regime_volume_means = _h1h2.estimate_regime_volume_means
simulate_difference_bands = _h1h2.simulate_difference_bands
_extract_point_frame = _h1h2._extract_point_frame
_simulate_regime_bands_from_shared_draws = _h1h2._simulate_regime_bands_from_shared_draws
_subtract_irf_frames = _h1h2._subtract_irf_frames


def _difference_draws(
    *,
    pre_draw_irfs: list[pd.DataFrame],
    post_draw_irfs: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Construct post-minus-pre draw paths from regime-specific draw tables."""

    diff_draws: list[pd.DataFrame] = []
    for draw_id, (pre_draw, post_draw) in enumerate(
        zip(pre_draw_irfs, post_draw_irfs, strict=True),
        start=1,
    ):
        pre_values = pre_draw.drop(columns=["draw_id"], errors="ignore")
        post_values = post_draw.drop(columns=["draw_id"], errors="ignore")
        diff = _subtract_irf_frames(post_values, pre_values)
        diff.insert(0, "draw_id", draw_id)
        diff_draws.append(diff)
    return diff_draws


def _add_band_identifiers(
    frame: pd.DataFrame,
    *,
    family_name: str,
    shock_name: str,
    group_name: str,
    regime_name: str,
) -> pd.DataFrame:
    """Add consistent identifier columns to a saved bands table."""

    out = frame.copy()
    out.insert(0, "regime", regime_name)
    out.insert(0, "group", group_name)
    out.insert(0, "shock_name", shock_name)
    out.insert(0, "family", family_name)
    return out


def _add_diagnostic_identifiers(
    frame: pd.DataFrame,
    *,
    family_name: str,
    shock_name: str,
    group_name: str,
    regime_name: str,
) -> pd.DataFrame:
    """Add consistent identifier columns to a draw-diagnostics table."""

    out = frame.copy()
    out.insert(0, "regime", regime_name)
    out.insert(0, "group", group_name)
    out.insert(0, "shock_name", shock_name)
    out.insert(0, "family", family_name)
    return out


def _key_horizons_for_family(family_name: str) -> list[int]:
    """Return a small set of horizons that are useful for quick H3 review."""

    if family_name == "vix":
        return [0, 1, 2, 3, 4, 5]
    if family_name == "macro":
        return [-1, 0, 1, 2, 3, 4]
    if family_name == "earnings":
        return [30 * k for k in range(1, 14)]
    raise ValueError(f"Unknown family: {family_name}")


def summarize_h3_bands(
    family_name: str,
    shock_name: str,
    h3_bands: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact review table from the final H3 bands table.

    We focus on dark-share changes in basis points because that is the 
    most intuitive treatment-control outcome for the write-up.
    """

    rows: list[dict[str, object]] = []
    available_horizons = set(h3_bands["horizon"].astype(int).tolist())
    for horizon in _key_horizons_for_family(family_name):
        if horizon not in available_horizons:
            continue
        row = h3_bands.loc[h3_bands["horizon"] == horizon].iloc[0]
        horizon_label = str(horizon)
        if family_name == "earnings":
            horizon_label = f"block_{horizon // 30}"

        lower = float(row["dark_share_change_bps_lower95"])
        upper = float(row["dark_share_change_bps_upper95"])
        rows.append(
            {
                "family": family_name,
                "shock_name": shock_name,
                "horizon": int(horizon),
                "horizon_label": horizon_label,
                "dark_share_change_bps_point": float(row["dark_share_change_bps_point"]),
                "dark_share_change_bps_lower95": lower,
                "dark_share_change_bps_upper95": upper,
                "exclude_zero": bool((lower > 0.0) or (upper < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _fit_group_regimes(
    family_name: str,
    *,
    group_name: str,
    tickers: Iterable[str],
    n_draws: int,
    p_lags: int | None = None,
    ridge: float | None = None,
    seed_offset: int = 0,
    family_spec_override=None,
) -> dict[str, object]:
    """Estimate one group's pre/post benchmark objects for one family.

    ``family_spec_override`` lets robustness passes inject a modified 
    family spec (e.g. the trend-controlled variant in the 
    trend-robustness block) without duplicating the rest of the fit 
    logic.
    """

    family = FAMILY_SPECS[family_name] if family_spec_override is None else family_spec_override
    active_p_lags = BENCHMARK_P_LAGS if p_lags is None else int(p_lags)

    pre_result = fit_family_regime(
        family,
        tickers=tickers,
        window=PRE_WINDOW,
        p_lags=active_p_lags,
        ridge=DEFAULT_RIDGE if ridge is None else float(ridge),
    )
    post_result = fit_family_regime(
        family,
        tickers=tickers,
        window=POST_WINDOW,
        p_lags=active_p_lags,
        ridge=DEFAULT_RIDGE if ridge is None else float(ridge),
    )

    pre_volume_means = estimate_regime_volume_means(family, tickers=tickers, window=PRE_WINDOW)
    post_volume_means = estimate_regime_volume_means(family, tickers=tickers, window=POST_WINDOW)

    shock_outputs: list[dict[str, object]] = []
    for shock_index, (shock_name, shock_path) in enumerate(family.shock_builder()):
        pre_bands, pre_diagnostics, pre_draw_irfs = _simulate_regime_bands_from_shared_draws(
            base_result=pre_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=pre_volume_means["dark_volume_mean"],
            lit_volume_mean=pre_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + seed_offset + 100 * shock_index,
        )
        post_bands, post_diagnostics, post_draw_irfs = _simulate_regime_bands_from_shared_draws(
            base_result=post_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=post_volume_means["dark_volume_mean"],
            lit_volume_mean=post_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + seed_offset + 1000 + 100 * shock_index,
        )

        change_bands = simulate_difference_bands(
            point_pre=_extract_point_frame(pre_bands),
            point_post=_extract_point_frame(post_bands),
            pre_draw_irfs=pre_draw_irfs,
            post_draw_irfs=post_draw_irfs,
        )
        change_draws = _difference_draws(
            pre_draw_irfs=pre_draw_irfs,
            post_draw_irfs=post_draw_irfs,
        )

        shock_outputs.append(
            {
                "shock_name": shock_name,
                "pre_bands": _add_band_identifiers(
                    pre_bands,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=group_name,
                    regime_name="pre",
                ),
                "post_bands": _add_band_identifiers(
                    post_bands,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=group_name,
                    regime_name="post",
                ),
                "change_bands": _add_band_identifiers(
                    change_bands,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=group_name,
                    regime_name="post_minus_pre",
                ),
                "pre_diagnostics": _add_diagnostic_identifiers(
                    pre_diagnostics,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=group_name,
                    regime_name="pre",
                ),
                "post_diagnostics": _add_diagnostic_identifiers(
                    post_diagnostics,
                    family_name=family_name,
                    shock_name=shock_name,
                    group_name=group_name,
                    regime_name="post",
                ),
                "change_draws": change_draws,
            }
        )

    return {
        "group": group_name,
        "tickers": list(tickers),
        "pre_result": pre_result,
        "post_result": post_result,
        "pre_volume_means": pre_volume_means,
        "post_volume_means": post_volume_means,
        "p_lags": active_p_lags,
        "shock_outputs": shock_outputs,
    }


def run_family_h3(
    family_name: str,
    *,
    treated_group: str = "treated",
    control_group: str = "matched_control",
    n_draws: int,
    p_lags: int | None = None,
    ridge: float | None = None,
    family_spec_override=None,
) -> dict[str, object]:
    """Run the benchmark H3 estimation for one urgency family.

    ``family_spec_override`` is forwarded to 
    :func:`_fit_group_regimes` and is used by the trend-controlled 
    robustness block to run the same pipeline with a ``day_index`` 
    covariate injected into the family spec.
    """

    group_definitions = load_group_definitions()
    active_p_lags = BENCHMARK_P_LAGS if p_lags is None else int(p_lags)

    treated_result = _fit_group_regimes(
        family_name,
        group_name=treated_group,
        tickers=group_definitions[treated_group],
        n_draws=n_draws,
        p_lags=active_p_lags,
        ridge=ridge,
        seed_offset=0,
        family_spec_override=family_spec_override,
    )
    control_result = _fit_group_regimes(
        family_name,
        group_name=control_group,
        tickers=group_definitions[control_group],
        n_draws=n_draws,
        p_lags=active_p_lags,
        ridge=ridge,
        seed_offset=2000,
        family_spec_override=family_spec_override,
    )

    h3_outputs: list[dict[str, object]] = []
    for treated_shock, control_shock in zip(
        treated_result["shock_outputs"],
        control_result["shock_outputs"],
        strict=True,
    ):
        shock_name = str(treated_shock["shock_name"])
        h3_bands = simulate_difference_bands(
            point_pre=_extract_point_frame(control_shock["change_bands"]),
            point_post=_extract_point_frame(treated_shock["change_bands"]),
            pre_draw_irfs=control_shock["change_draws"],
            post_draw_irfs=treated_shock["change_draws"],
        )
        h3_bands = _add_band_identifiers(
            h3_bands,
            family_name=family_name,
            shock_name=shock_name,
            group_name=f"{treated_group}_minus_{control_group}",
            regime_name="post_minus_pre",
        )
        h3_outputs.append(
            {
                "shock_name": shock_name,
                "h3_bands": h3_bands,
                "key_summary": summarize_h3_bands(family_name, shock_name, h3_bands),
            }
        )

    return {
        "family": family_name,
        "p_lags": active_p_lags,
        "treated_group": treated_group,
        "control_group": control_group,
        "treated_result": treated_result,
        "control_result": control_result,
        "h3_outputs": h3_outputs,
    }


def build_run_summary(results: list[dict[str, object]], *, n_draws: int) -> pd.DataFrame:
    """Create one compact benchmark summary row per family and shock."""

    rows: list[dict[str, object]] = []
    for family_result in results:
        family_name = str(family_result["family"])
        treated = family_result["treated_result"]
        control = family_result["control_result"]
        for h3_output in family_result["h3_outputs"]:
            rows.append(
                {
                    "family": family_name,
                    "shock_name": h3_output["shock_name"],
                    "treated_group": str(family_result.get("treated_group", "treated")),
                    "control_group": str(family_result.get("control_group", "matched_control")),
                    "benchmark_p_lags": int(family_result["p_lags"]),
                    "n_draws": int(n_draws),
                    "treated_pre_nobs": int(treated["pre_result"].design_info["nobs"]),
                    "treated_post_nobs": int(treated["post_result"].design_info["nobs"]),
                    "control_pre_nobs": int(control["pre_result"].design_info["nobs"]),
                    "control_post_nobs": int(control["post_result"].design_info["nobs"]),
                    "treated_pre_entities": int(treated["pre_result"].design_info["n_entities"]),
                    "treated_post_entities": int(treated["post_result"].design_info["n_entities"]),
                    "control_pre_entities": int(control["pre_result"].design_info["n_entities"]),
                    "control_post_entities": int(control["post_result"].design_info["n_entities"]),
                    "treated_n_tickers": len(treated["tickers"]),
                    "control_n_tickers": len(control["tickers"]),
                }
            )
    return pd.DataFrame(rows)


def build_key_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    """Stack the compact H3 review rows across families."""

    tables: list[pd.DataFrame] = []
    for family_result in results:
        for h3_output in family_result["h3_outputs"]:
            tables.append(h3_output["key_summary"])
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)

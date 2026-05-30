"""Helpers for H1/H2 estimation on top of the VARX engine.

The VARX folder already contains the baseline VARX, IRF, and inference 
code. This module keeps the H1/H2 layer thin by doing only three 
things:

1. Split the cleaned sample into pre and post windows.
2. Reuse the benchmark beta specification for each urgency family.
3. Construct regime-specific IRFs and a post-minus-pre comparison 
object.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Parallelism for the inner Monte Carlo IRF loop. Threading backend 
# keeps pickling simple given the dynamically-imported numbered 
# modules in this project.
_PARALLEL_N_JOBS = int(os.environ.get("VARX_MC_N_JOBS", "-1"))
_PARALLEL_BACKEND = os.environ.get("VARX_MC_BACKEND", "threading")


# The VARX files use numeric prefixes. We therefore put the VARX 
# directory on the import path and load the sibling modules through 
# importlib.
BETA_DIR = Path(__file__).resolve().parents[1] / "04_VARX"
if str(BETA_DIR) not in sys.path:
    sys.path.insert(0, str(BETA_DIR))

_config = importlib.import_module("02_beta_varx_config")
_data = importlib.import_module("04_beta_varx_data")
_model = importlib.import_module("06_beta_varx_model")
_panel = importlib.import_module("05_beta_varx_panel")
_irf = importlib.import_module("10_beta_varx_irf")
_inference = importlib.import_module("13_beta_varx_inference")
_local = importlib.import_module("01_estimation_config")


Y_COLS = _config.Y_COLS
VIX_X_COLS = _config.VIX_X_COLS
MACRO_X_COLS = _config.MACRO_X_COLS
MACRO_FOMC_X_COLS = _config.MACRO_FOMC_X_COLS
MACRO_INFLATION_X_COLS = _config.MACRO_INFLATION_X_COLS
EARNINGS_X_COLS = _config.EARNINGS_X_COLS

VIX_IRF_SHOCK_COLS = _config.VIX_IRF_SHOCK_COLS
MACRO_EVENT_SHOCK_SIZE = _config.MACRO_EVENT_SHOCK_SIZE
EARNINGS_EVENT_SHOCK_SIZE = _config.EARNINGS_EVENT_SHOCK_SIZE

load_vix_panel = _data.load_vix_panel

BENCHMARK_P_LAGS = _local.BENCHMARK_P_LAGS
DEFAULT_RIDGE = _local.DEFAULT_RIDGE
BASE_SEED = _local.BASE_SEED

RegimeWindow = _local.RegimeWindow

load_sp500_universe = _data.load_sp500_universe
BaselinePanelVARX = _model.BaselinePanelVARX
apply_two_way_clustered_covariance = _model.apply_two_way_clustered_covariance

iter_vix_panel_pieces = _panel.iter_vix_panel_pieces
iter_macro_panel_pieces = _panel.iter_macro_panel_pieces
iter_macro_fomc_panel_pieces = _panel.iter_macro_fomc_panel_pieces
iter_macro_inflation_panel_pieces = _panel.iter_macro_inflation_panel_pieces
iter_earnings_panel_pieces = _panel.iter_earnings_panel_pieces

build_unit_shock_path = _irf.build_unit_shock_path
build_macro_event_path = _irf.build_macro_event_path
build_macro_fomc_event_path = _irf.build_macro_fomc_event_path
build_macro_inflation_event_path = _irf.build_macro_inflation_event_path
build_earnings_event_path = _irf.build_earnings_event_path
estimate_baseline_volume_means = _irf.estimate_baseline_volume_means

coefficient_draw_generator = _inference.coefficient_draw_generator
clone_result_with_coefficients = _inference.clone_result_with_coefficients
finalize_irf_for_inference = _inference.finalize_irf_for_inference
summarize_irf_draws = _inference.summarize_irf_draws

simulate_irf_recursive = _irf.simulate_irf_recursive


@dataclass(frozen=True)
class FamilySpec:
    """Minimal specification needed for one urgency family."""

    name: str
    common_x_cols: tuple[str, ...]
    panel_x_cols: tuple[str, ...]
    horizon_end: int
    iterator_factory: Callable[[Iterable[str] | None], Iterator[pd.DataFrame]]
    shock_builder: Callable[[], list[tuple[str, pd.DataFrame]]]


_VIX_POS_SIGMA_CACHE: float | None = None


def get_vix_pos_innovation_sigma() -> float:
    """Return the sample standard deviation of ``dVIX_pos_inv``."""

    global _VIX_POS_SIGMA_CACHE
    if _VIX_POS_SIGMA_CACHE is not None:
        return _VIX_POS_SIGMA_CACHE

    cache_path = _local.OUTPUT_DIR / "vix_dvix_pos_sigma.txt"
    if cache_path.exists():
        _VIX_POS_SIGMA_CACHE = float(cache_path.read_text().strip())
        return _VIX_POS_SIGMA_CACHE

    vix = load_vix_panel()
    sigma = float(vix["dVIX_pos_inv"].to_numpy(dtype=float).std(ddof=1))
    _local.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(f"{sigma:.12g}\n")
    _VIX_POS_SIGMA_CACHE = sigma
    return sigma


def _build_vix_shocks() -> list[tuple[str, pd.DataFrame]]:
    """Return the calibrated VIX innovation shocks used in the beta build."""

    shock_size = get_vix_pos_innovation_sigma()
    return [
        (
            shock_col,
            build_unit_shock_path(
                VIX_X_COLS,
                shock_col=shock_col,
                shock_size=shock_size,
            ),
        )
        for shock_col in VIX_IRF_SHOCK_COLS
    ]


def _build_macro_shocks() -> list[tuple[str, pd.DataFrame]]:
    """Return the macro event path."""

    return [("macro_event_path", build_macro_event_path(shock_size=MACRO_EVENT_SHOCK_SIZE))]


def _build_macro_fomc_shocks() -> list[tuple[str, pd.DataFrame]]:
    """Return the FOMC-only macro event path."""

    return [
        (
            "macro_fomc_event_path",
            build_macro_fomc_event_path(shock_size=MACRO_EVENT_SHOCK_SIZE),
        )
    ]


def _build_macro_inflation_shocks() -> list[tuple[str, pd.DataFrame]]:
    """Return the CPI/PPI-only macro event path."""

    return [
        (
            "macro_inflation_event_path",
            build_macro_inflation_event_path(shock_size=MACRO_EVENT_SHOCK_SIZE),
        )
    ]


def _build_earnings_shocks() -> list[tuple[str, pd.DataFrame]]:
    """Return the earnings event path."""

    return [
        (
            "earnings_event_path",
            build_earnings_event_path(
                shock_size=EARNINGS_EVENT_SHOCK_SIZE,
                block_minutes=30,
            ),
        )
    ]


FAMILY_SPECS: dict[str, FamilySpec] = {
    "vix": FamilySpec(
        name="vix",
        common_x_cols=VIX_X_COLS,
        panel_x_cols=(),
        horizon_end=60,
        iterator_factory=iter_vix_panel_pieces,
        shock_builder=_build_vix_shocks,
    ),
    "macro": FamilySpec(
        name="macro",
        common_x_cols=MACRO_X_COLS,
        panel_x_cols=(),
        horizon_end=60,
        iterator_factory=iter_macro_panel_pieces,
        shock_builder=_build_macro_shocks,
    ),
    # FOMC-only and inflation-only refits used for the macro 
    # decomposition diagnostic. Same VARX engine as the combined macro 
    # family but with a different exogenous block.
    "macro_fomc": FamilySpec(
        name="macro_fomc",
        common_x_cols=MACRO_FOMC_X_COLS,
        panel_x_cols=(),
        horizon_end=60,
        iterator_factory=iter_macro_fomc_panel_pieces,
        shock_builder=_build_macro_fomc_shocks,
    ),
    "macro_inflation": FamilySpec(
        name="macro_inflation",
        common_x_cols=MACRO_INFLATION_X_COLS,
        panel_x_cols=(),
        horizon_end=60,
        iterator_factory=iter_macro_inflation_panel_pieces,
        shock_builder=_build_macro_inflation_shocks,
    ),
    "earnings": FamilySpec(
        name="earnings",
        common_x_cols=(),
        panel_x_cols=EARNINGS_X_COLS,
        horizon_end=450,
        iterator_factory=iter_earnings_panel_pieces,
        shock_builder=_build_earnings_shocks,
    ),
}


def filter_piece_to_window(piece: pd.DataFrame, window: RegimeWindow) -> pd.DataFrame:
    """Restrict one stock panel piece to a regime window.

    The data layer has already applied the global sample limits
    and the October exclusion-window exclusion. Here we only do the
    additional pre/post split needed for H1 and H2.
    """

    timestamp = pd.to_datetime(piece["timestamp"])
    mask = (timestamp >= pd.Timestamp(window.start)) & (timestamp <= pd.Timestamp(window.end))
    return piece.loc[mask].copy()


def iter_windowed_pieces(
    family: FamilySpec,
    *,
    tickers: Iterable[str] | None,
    window: RegimeWindow,
) -> Iterator[pd.DataFrame]:
    """Yield stock-minute panel pieces for one family and one regime."""

    for piece in family.iterator_factory(tickers):
        filtered = filter_piece_to_window(piece, window)
        if not filtered.empty:
            yield filtered


def fit_family_regime(
    family: FamilySpec,
    *,
    tickers: Iterable[str] | None,
    window: RegimeWindow,
    p_lags: int | None = None,
    ridge: float | None = None,
):
    """Fit the benchmark VARX for one urgency family and one regime.

    The returned result carries the two-way (stock * time) 
    Petersen-clustered parameter covariance as 
    ``parameter_covariance``. The classical Sigma_u ⊗ (X'X)^{-1} 
    estimate is preserved on the same object as
    ``parameter_covariance_classical`` for diagnostic comparison.
    """

    model = BaselinePanelVARX(
        p_lags=BENCHMARK_P_LAGS if p_lags is None else int(p_lags),
        ridge=DEFAULT_RIDGE if ridge is None else float(ridge),
        entity_fixed_effects=True,
    )
    materialized_tickers = None if tickers is None else list(tickers)

    def iterator_factory():
        return iter_windowed_pieces(family, tickers=materialized_tickers, window=window)

    result = model.fit_from_iterator(
        iterator_factory(),
        entity_col="asset",
        time_col="timestamp",
        y_cols=Y_COLS,
        common_x_cols=family.common_x_cols,
        panel_x_cols=family.panel_x_cols,
    )
    apply_two_way_clustered_covariance(
        model,
        iterator_factory,
        result=result,
        entity_col="asset",
        time_col="timestamp",
        y_cols=Y_COLS,
        common_x_cols=family.common_x_cols,
        panel_x_cols=family.panel_x_cols,
    )
    return result


def estimate_regime_volume_means(
    family: FamilySpec,
    *,
    tickers: Iterable[str] | None,
    window: RegimeWindow,
) -> dict[str, float]:
    """Estimate mean dark and lit volumes for one regime.

    The derived dark-share responses depend on regime-specific average 
    volume levels. We therefore estimate those levels separately in 
    the pre and post samples.
    """

    return estimate_baseline_volume_means(iter_windowed_pieces(family, tickers=tickers, window=window))


def _simulate_final_irf(
    result,
    *,
    shock_path: pd.DataFrame,
    horizon_end: int,
    dark_volume_mean: float,
    lit_volume_mean: float,
) -> pd.DataFrame:
    """Simulate one finalized IRF table for a fitted result."""

    raw = simulate_irf_recursive(result, shock_path, horizon_end=horizon_end)
    return finalize_irf_for_inference(
        raw,
        dark_volume_mean=dark_volume_mean,
        lit_volume_mean=lit_volume_mean,
    )


def _simulate_irf_draws_from_coefficients(
    *,
    base_result,
    coefficient_draws: list[np.ndarray],
    shock_path: pd.DataFrame,
    horizon_end: int,
    dark_volume_mean: float,
    lit_volume_mean: float,
) -> list[pd.DataFrame]:
    """Turn coefficient draws into finalized IRF draw tables.

    Step 3 already separates coefficient drawing from IRF simulation. 
    The H1/H2 layer reuses that separation so the same pre/post draws 
    can support both the regime-specific bands and the post-minus-pre 
    comparison.
    """

    def _one_draw(draw_id: int, draw_beta: np.ndarray) -> pd.DataFrame:
        draw_result = clone_result_with_coefficients(base_result, draw_beta)
        draw_irf = _simulate_final_irf(
            draw_result,
            shock_path=shock_path,
            horizon_end=horizon_end,
            dark_volume_mean=dark_volume_mean,
            lit_volume_mean=lit_volume_mean,
        )
        draw_irf.insert(0, "draw_id", draw_id)
        return draw_irf

    draw_irfs = Parallel(n_jobs=_PARALLEL_N_JOBS, backend=_PARALLEL_BACKEND)(
        delayed(_one_draw)(draw_id, draw_beta)
        for draw_id, draw_beta in enumerate(coefficient_draws, start=1)
    )
    return draw_irfs


def _simulate_regime_bands_from_shared_draws(
    *,
    base_result,
    shock_path: pd.DataFrame,
    horizon_end: int,
    dark_volume_mean: float,
    lit_volume_mean: float,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Build one regime's IRF bands while keeping the underlying draw 
    paths.

    This helper avoids duplicate work. The same simulated draw paths 
    can later be paired with the other regime to form the 
    post-minus-pre difference.
    """

    point_irf = _simulate_final_irf(
        base_result,
        shock_path=shock_path,
        horizon_end=horizon_end,
        dark_volume_mean=dark_volume_mean,
        lit_volume_mean=lit_volume_mean,
    )
    coefficient_draws, diagnostics = coefficient_draw_generator(
        base_result,
        n_draws=n_draws,
        seed=seed,
        require_stable=True,
    )
    draw_irfs = _simulate_irf_draws_from_coefficients(
        base_result=base_result,
        coefficient_draws=coefficient_draws,
        shock_path=shock_path,
        horizon_end=horizon_end,
        dark_volume_mean=dark_volume_mean,
        lit_volume_mean=lit_volume_mean,
    )
    summary = summarize_irf_draws(
        point_irf=point_irf,
        draw_irfs=draw_irfs,
        alpha=_local.SIMULATION_ALPHA,
    )
    return summary, diagnostics, draw_irfs


def _subtract_irf_frames(post_irf: pd.DataFrame, pre_irf: pd.DataFrame) -> pd.DataFrame:
    """Compute the post-minus-pre difference for matched IRF tables."""

    if list(post_irf["horizon"]) != list(pre_irf["horizon"]):
        raise ValueError("Pre and post IRFs do not share the same horizon grid.")

    diff = pd.DataFrame({"horizon": post_irf["horizon"].to_numpy(dtype=int)})
    value_cols = [col for col in post_irf.columns if col != "horizon"]
    for col in value_cols:
        diff[col] = (
            post_irf[col].to_numpy(dtype=float)
            - pre_irf[col].to_numpy(dtype=float)
        )
    return diff


def _extract_point_frame(summary_bands: pd.DataFrame) -> pd.DataFrame:
    """Recover the point-IRF table from a summary bands table.

    The regime-specific summaries are saved with `_point`, `_lower95`, 
    and `_upper95` suffixes. The difference calculation only needs the 
    point path.
    """

    point_cols = ["horizon"] + [col for col in summary_bands.columns if col.endswith("_point")]
    point = summary_bands[point_cols].copy()
    point = point.rename(columns={col: col[:-6] for col in point.columns if col.endswith("_point")})
    return point


def simulate_difference_bands(
    *,
    point_pre: pd.DataFrame,
    point_post: pd.DataFrame,
    pre_draw_irfs: list[pd.DataFrame],
    post_draw_irfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """Construct post-minus-pre confidence bands from already 
    simulated draws.

    The regime-specific band construction already simulated 
    independent draws for pre and post. We reuse those draws here 
    instead of drawing the same models a second time.
    """

    point_diff = _subtract_irf_frames(point_post, point_pre)

    diff_draws: list[pd.DataFrame] = []
    for draw_id, (pre_draw_irf, post_draw_irf) in enumerate(
        zip(pre_draw_irfs, post_draw_irfs, strict=True),
        start=1,
    ):
        # The stored draw tables carry their own draw_id column. The
        # difference object should be based only on the 
        # horizon-by-response paths, after which we add a single fresh 
        # draw_id column.
        pre_values = pre_draw_irf.drop(columns=["draw_id"], errors="ignore")
        post_values = post_draw_irf.drop(columns=["draw_id"], errors="ignore")
        diff_draw = _subtract_irf_frames(post_values, pre_values)
        diff_draw.insert(0, "draw_id", draw_id)
        diff_draws.append(diff_draw)

    diff_summary = summarize_irf_draws(
        point_irf=point_diff,
        draw_irfs=diff_draws,
        alpha=_local.SIMULATION_ALPHA,
    )
    return diff_summary


def run_family_h1_h2(
    family_name: str,
    *,
    tickers: Iterable[str] | None = None,
    n_draws: int,
    p_lags: int | None = None,
    ridge: float | None = None,
    family_spec_override: "FamilySpec | None" = None,
) -> dict[str, object]:
    """Run the full H1/H2 estimation for one urgency family.

    The return value is deliberately plain. The runner script can save 
    the pieces it needs without hiding the structure inside a custom 
    class.
    """

    family = FAMILY_SPECS[family_name] if family_spec_override is None else family_spec_override
    if tickers is None:
        tickers = load_sp500_universe()
    tickers = list(tickers)

    active_p_lags = BENCHMARK_P_LAGS if p_lags is None else int(p_lags)

    pre_result = fit_family_regime(
        family,
        tickers=tickers,
        window=_local.PRE_WINDOW,
        p_lags=active_p_lags,
        ridge=ridge,
    )
    post_result = fit_family_regime(
        family,
        tickers=tickers,
        window=_local.POST_WINDOW,
        p_lags=active_p_lags,
        ridge=ridge,
    )

    pre_volume_means = estimate_regime_volume_means(family, tickers=tickers, window=_local.PRE_WINDOW)
    post_volume_means = estimate_regime_volume_means(family, tickers=tickers, window=_local.POST_WINDOW)

    outputs: dict[str, object] = {
        "family": family_name,
        "pre_result": pre_result,
        "post_result": post_result,
        "pre_volume_means": pre_volume_means,
        "post_volume_means": post_volume_means,
        "p_lags": active_p_lags,
        "shock_outputs": [],
    }

    for shock_index, (shock_name, shock_path) in enumerate(family.shock_builder()):
        pre_bands, pre_diagnostics, pre_draw_irfs = _simulate_regime_bands_from_shared_draws(
            base_result=pre_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=pre_volume_means["dark_volume_mean"],
            lit_volume_mean=pre_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + 100 * shock_index,
        )
        post_bands, post_diagnostics, post_draw_irfs = _simulate_regime_bands_from_shared_draws(
            base_result=post_result,
            shock_path=shock_path,
            horizon_end=family.horizon_end,
            dark_volume_mean=post_volume_means["dark_volume_mean"],
            lit_volume_mean=post_volume_means["lit_volume_mean"],
            n_draws=n_draws,
            seed=BASE_SEED + 1000 + 100 * shock_index,
        )
        difference_bands = simulate_difference_bands(
            point_pre=_extract_point_frame(pre_bands),
            point_post=_extract_point_frame(post_bands),
            pre_draw_irfs=pre_draw_irfs,
            post_draw_irfs=post_draw_irfs,
        )

        for frame, regime_name in ((pre_bands, "pre"), (post_bands, "post"), (difference_bands, "post_minus_pre")):
            frame.insert(0, "shock_name", shock_name)
            frame.insert(0, "family", family_name)
            frame.insert(0, "regime", regime_name)

        for diagnostics, regime_name in (
            (pre_diagnostics, "pre"),
            (post_diagnostics, "post"),
        ):
            diagnostics.insert(0, "shock_name", shock_name)
            diagnostics.insert(0, "family", family_name)
            diagnostics.insert(0, "regime", regime_name)

        outputs["shock_outputs"].append(
            {
                "shock_name": shock_name,
                "pre_bands": pre_bands,
                "post_bands": post_bands,
                "difference_bands": difference_bands,
                "pre_diagnostics": pre_diagnostics,
                "post_diagnostics": post_diagnostics,
            }
        )

    return outputs


def build_run_summary(
    results: list[dict[str, object]],
    *,
    n_draws: int,
    p_lags: int | None = None,
) -> pd.DataFrame:
    """Create one compact summary row per family/shock/regime object."""

    rows: list[dict[str, object]] = []
    for family_result in results:
        family_name = str(family_result["family"])
        pre_result = family_result["pre_result"]
        post_result = family_result["post_result"]
        active_p_lags = int(family_result.get("p_lags", BENCHMARK_P_LAGS if p_lags is None else p_lags))
        for shock_output in family_result["shock_outputs"]:
            shock_name = str(shock_output["shock_name"])
            if family_name == "vix":
                shock_size = float(get_vix_pos_innovation_sigma())
                shock_units = "sigma(dVIX_pos_inv)"
            elif family_name == "macro":
                shock_size = float(MACRO_EVENT_SHOCK_SIZE)
                shock_units = "surprise_path_unit"
            elif family_name == "earnings":
                shock_size = float(EARNINGS_EVENT_SHOCK_SIZE)
                shock_units = "earnings_surprise_unit"
            else:
                shock_size = float("nan")
                shock_units = ""
            rows.append(
                {
                    "family": family_name,
                    "shock_name": shock_name,
                    "benchmark_p_lags": active_p_lags,
                    "n_draws": n_draws,
                    "shock_size": shock_size,
                    "shock_units": shock_units,
                    "pre_nobs": int(pre_result.design_info["nobs"]),
                    "post_nobs": int(post_result.design_info["nobs"]),
                    "pre_entities": int(pre_result.design_info["n_entities"]),
                    "post_entities": int(post_result.design_info["n_entities"]),
                    "pre_sample_start": pre_result.design_info.get("sample_start"),
                    "pre_sample_end": pre_result.design_info.get("sample_end"),
                    "post_sample_start": post_result.design_info.get("sample_start"),
                    "post_sample_end": post_result.design_info.get("sample_end"),
                }
            )
    return pd.DataFrame(rows)

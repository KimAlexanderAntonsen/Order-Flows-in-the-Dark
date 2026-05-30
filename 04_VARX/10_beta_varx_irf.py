"""Step 2 impulse-response helpers for the beta VARX build.

This module sits directly on top of the Step 1 baseline VARX. The goal 
is to keep the IRF logic readable:

1. Recover the estimated lag matrices and exogenous-response matrix.
2. Feed a chosen exogenous shock path through the system.
3. Compare a direct recursive simulation with a companion-form 
   simulation.
4. Derive a dark-share response from the simulated dark and lit volume 
   paths.

The model we simulate is the Step 1 baseline:

    y_t = Phi_1 y_{t-1} + ... + Phi_p y_{t-p} + Psi x_t

with stock fixed effects already removed during estimation.
"""

from __future__ import annotations

import importlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_config = importlib.import_module("02_beta_varx_config")
_model = importlib.import_module("06_beta_varx_model")

MACRO_X_COLS = _config.MACRO_X_COLS
MACRO_FOMC_X_COLS = _config.MACRO_FOMC_X_COLS
MACRO_INFLATION_X_COLS = _config.MACRO_INFLATION_X_COLS
EARNINGS_X_COLS = _config.EARNINGS_X_COLS
companion_matrix = _model.companion_matrix


def get_irf_x_cols(result) -> list[str]:
    """Return the exogenous columns in the order used by the fitted model."""

    return [*result.common_x_cols, *result.panel_x_cols]


def get_irf_psi_matrix(result) -> np.ndarray:
    """Return the combined exogenous-response matrix Psi.

    The rows correspond to endogenous variables and the columns 
    correspond to the exogenous variables returned by 
    :func:`get_irf_x_cols`.
    """

    blocks: list[np.ndarray] = []
    common = result.common_exog_matrix()
    panel = result.panel_exog_matrix()
    if common is not None:
        blocks.append(common)
    if panel is not None:
        blocks.append(panel)
    if not blocks:
        return np.zeros((len(result.y_cols), 0), dtype=float)
    return np.hstack(blocks)


def build_unit_shock_path(
    x_cols: Sequence[str],
    *,
    shock_col: str,
    shock_size: float = 1.0,
) -> pd.DataFrame:
    """Create a one-minute unit shock for a single exogenous regressor."""

    if shock_col not in x_cols:
        raise ValueError(f"{shock_col} is not in the fitted exogenous block.")

    row = {col: 0.0 for col in x_cols}
    row["horizon"] = 0
    row[shock_col] = float(shock_size)
    return pd.DataFrame([row])


def build_macro_event_path(*, shock_size: float = 1.0) -> pd.DataFrame:
    """Create the intended macro-news event path.

    The processed macro panel contains one dummy for the minute before 
    the scheduled release and five dummies covering the release minute 
    and the next four minutes. The path therefore starts at horizon 
    -1.
    """

    rows = [
        {"horizon": -1, "pre_news_1min": shock_size},
        {"horizon": 0, "post_news_0min": shock_size},
        {"horizon": 1, "post_news_1min": shock_size},
        {"horizon": 2, "post_news_2min": shock_size},
        {"horizon": 3, "post_news_3min": shock_size},
        {"horizon": 4, "post_news_4min": shock_size},
    ]
    frame = pd.DataFrame(rows)
    for col in MACRO_X_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[["horizon", *MACRO_X_COLS]].fillna(0.0)


def build_macro_fomc_event_path(*, shock_size: float = 1.0) -> pd.DataFrame:
    """Create the FOMC-only macro event path.

    Same -1 to +4 horizon structure as :func:`build_macro_event_path`,
    but using the rate-decision dummies. The shock is interpreted as 
    the FOMC contribution to the macro IRF and is the object reported 
    in the macro decomposition diagnostic.
    """

    rows = [
        {"horizon": -1, "pre_rate_1min": shock_size},
        {"horizon": 0, "post_rate_0min": shock_size},
        {"horizon": 1, "post_rate_1min": shock_size},
        {"horizon": 2, "post_rate_2min": shock_size},
        {"horizon": 3, "post_rate_3min": shock_size},
        {"horizon": 4, "post_rate_4min": shock_size},
    ]
    frame = pd.DataFrame(rows)
    for col in MACRO_FOMC_X_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[["horizon", *MACRO_FOMC_X_COLS]].fillna(0.0)


def build_macro_inflation_event_path(*, shock_size: float = 1.0) -> pd.DataFrame:
    """Create the CPI/PPI-only macro event path.

    Same -1 to +4 horizon structure as :func:`build_macro_event_path`,
    but using the inflation-release dummies. The shock is interpreted 
    as the CPI/PPI contribution to the macro IRF and is the foil to 
    the FOMC-only path.
    """

    rows = [
        {"horizon": -1, "pre_inflation_1min": shock_size},
        {"horizon": 0, "post_inflation_0min": shock_size},
        {"horizon": 1, "post_inflation_1min": shock_size},
        {"horizon": 2, "post_inflation_2min": shock_size},
        {"horizon": 3, "post_inflation_3min": shock_size},
        {"horizon": 4, "post_inflation_4min": shock_size},
    ]
    frame = pd.DataFrame(rows)
    for col in MACRO_INFLATION_X_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[["horizon", *MACRO_INFLATION_X_COLS]].fillna(0.0)


def build_earnings_event_path(
    *,
    shock_size: float = 1.0,
    block_minutes: int = 30,
) -> pd.DataFrame:
    """Create the intended minute-level earnings event path.

    In our earnings design, each `post_ea_k` regressor is active for 
    one half-hour block. This helper expands that design into the 
    minute grid used by the Step 1 baseline model.
    """

    rows: list[dict[str, float]] = []
    for block_idx, col in enumerate(EARNINGS_X_COLS):
        start = block_idx * block_minutes
        end = start + block_minutes
        for horizon in range(start, end):
            row = {name: 0.0 for name in EARNINGS_X_COLS}
            row["horizon"] = horizon
            row[col] = float(shock_size)
            rows.append(row)
    return pd.DataFrame(rows)[["horizon", *EARNINGS_X_COLS]]


def _prepare_shock_path(
    x_cols: Sequence[str],
    shock_path: pd.DataFrame,
    *,
    horizon_end: int,
) -> tuple[dict[int, np.ndarray], int, int]:
    """Convert a shock-path DataFrame into a horizon-to-vector mapping."""

    if "horizon" not in shock_path.columns:
        raise ValueError("shock_path must contain a 'horizon' column.")

    work = shock_path.copy()
    for col in x_cols:
        if col not in work.columns:
            work[col] = 0.0

    work = work[["horizon", *x_cols]].copy()
    work["horizon"] = work["horizon"].astype(int)
    work = work.groupby("horizon", as_index=False)[list(x_cols)].sum()

    start_horizon = min(0, int(work["horizon"].min())) if not work.empty else 0
    if horizon_end < 0:
        raise ValueError("horizon_end must be non-negative.")

    shock_map = {
        int(row["horizon"]): row[list(x_cols)].to_numpy(dtype=float)
        for _, row in work.iterrows()
    }
    return shock_map, start_horizon, int(horizon_end)


def simulate_irf_recursive(
    result,
    shock_path: pd.DataFrame,
    *,
    horizon_end: int,
) -> pd.DataFrame:
    """Simulate the IRF directly from the VARX recursion.

    This is the most transparent implementation:

        y_h = Phi_1 y_{h-1} + ... + Phi_p y_{h-p} + Psi x_h

    The simulation starts from a zero state, which is natural because 
    the within-transformed Step 1 model is centered around the 
    fixed-effect means.
    """

    y_cols = list(result.y_cols)
    x_cols = get_irf_x_cols(result)
    psi = get_irf_psi_matrix(result)
    lag_mats = result.lag_matrices()
    k_y = len(y_cols)

    shock_map, start_horizon, end_horizon = _prepare_shock_path(
        x_cols,
        shock_path,
        horizon_end=horizon_end,
    )

    zeros = np.zeros(k_y, dtype=float)
    responses: dict[int, np.ndarray] = {}

    for horizon in range(start_horizon, end_horizon + 1):
        y_h = np.zeros(k_y, dtype=float)
        for lag, phi in enumerate(lag_mats, start=1):
            y_h += phi @ responses.get(horizon - lag, zeros)

        x_h = shock_map.get(horizon)
        if x_h is not None and psi.size:
            y_h += psi @ x_h

        responses[horizon] = y_h

    rows = []
    for horizon in range(start_horizon, end_horizon + 1):
        row = {"horizon": horizon}
        row.update({col: responses[horizon][idx] for idx, col in enumerate(y_cols)})
        rows.append(row)

    return pd.DataFrame(rows)


def simulate_irf_companion(
    result,
    shock_path: pd.DataFrame,
    *,
    horizon_end: int,
) -> pd.DataFrame:
    """Simulate the same IRF in companion form.

    Menkveld describe the dynamic system through a companion 
    representation. We use the same logic here and then compare it to 
    the direct recursion above. If the two paths agree, we have a 
    strong implementation check.
    """

    y_cols = list(result.y_cols)
    x_cols = get_irf_x_cols(result)
    psi = get_irf_psi_matrix(result)
    lag_mats = result.lag_matrices()
    k_y = len(y_cols)
    p_lags = len(lag_mats)

    shock_map, start_horizon, end_horizon = _prepare_shock_path(
        x_cols,
        shock_path,
        horizon_end=horizon_end,
    )

    state_dim = k_y * p_lags
    transition = companion_matrix(lag_mats)
    shock_load = np.zeros((state_dim, len(x_cols)), dtype=float)
    if psi.size:
        shock_load[:k_y, :] = psi

    state_prev = np.zeros(state_dim, dtype=float)
    responses: dict[int, np.ndarray] = {}

    for horizon in range(start_horizon, end_horizon + 1):
        x_h = shock_map.get(horizon, np.zeros(len(x_cols), dtype=float))
        state_now = transition @ state_prev
        if shock_load.size:
            state_now += shock_load @ x_h
        responses[horizon] = state_now[:k_y]
        state_prev = state_now

    rows = []
    # Match the direct recursion exactly, including negative horizons 
    # when the shock path starts before the event minute.
    for horizon in range(start_horizon, end_horizon + 1):
        row = {"horizon": horizon}
        row.update({col: responses[horizon][idx] for idx, col in enumerate(y_cols)})
        rows.append(row)

    return pd.DataFrame(rows)


def validate_irf_paths(
    recursive_irf: pd.DataFrame,
    companion_irf: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    tolerance: float = 1e-10,
) -> dict[str, float | bool]:
    """Compare the two IRF implementations and report the largest gap."""

    merged = recursive_irf.merge(
        companion_irf,
        on="horizon",
        suffixes=("_recursive", "_companion"),
        how="inner",
    )

    max_abs_diff = 0.0
    for col in value_cols:
        diff = np.abs(merged[f"{col}_recursive"] - merged[f"{col}_companion"]).max()
        max_abs_diff = max(max_abs_diff, float(diff))

    return {
        "max_abs_diff": max_abs_diff,
        "passes_tolerance": bool(max_abs_diff <= tolerance),
        "tolerance": float(tolerance),
    }


def check_one_off_contemporaneous_response(
    result,
    recursive_irf: pd.DataFrame,
    *,
    shock_col: str,
    shock_size: float = 1.0,
) -> dict[str, float | bool]:
    """Check that the horizon-0 response matches the contemporaneous 
    coefficient.

    For a one-minute unit shock and zero initial state, the horizon-0 
    response should equal the corresponding column of Psi.
    """

    x_cols = get_irf_x_cols(result)
    psi = get_irf_psi_matrix(result)
    shock_idx = x_cols.index(shock_col)
    expected = psi[:, shock_idx] * float(shock_size)
    observed = recursive_irf.loc[recursive_irf["horizon"] == 0, list(result.y_cols)].iloc[0].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(observed - expected)))
    return {
        "max_abs_diff": max_abs_diff,
        "passes_tolerance": bool(max_abs_diff <= 1e-10),
        "tolerance": 1e-10,
    }


def estimate_baseline_volume_means(pieces: Iterable[pd.DataFrame]) -> dict[str, float]:
    """Estimate the average dark and lit trading volume used for dark-share IRFs."""

    dark_sum = 0.0
    lit_sum = 0.0
    nobs = 0

    for piece in pieces:
        dark = pd.to_numeric(piece["dark_volume_t"], errors="coerce").fillna(0.0)
        lit = pd.to_numeric(piece["lit_volume_t"], errors="coerce").fillna(0.0)
        dark_sum += float(dark.sum())
        lit_sum += float(lit.sum())
        nobs += int(len(piece))

    if nobs == 0:
        raise ValueError("No observations were available when estimating baseline volume means.")

    dark_mean = dark_sum / float(nobs)
    lit_mean = lit_sum / float(nobs)
    baseline_dark_share = dark_mean / (dark_mean + lit_mean)
    return {
        "dark_volume_mean": dark_mean,
        "lit_volume_mean": lit_mean,
        "baseline_dark_share": baseline_dark_share,
    }


def add_dark_share_response(
    irf: pd.DataFrame,
    *,
    dark_volume_mean: float,
    lit_volume_mean: float,
) -> pd.DataFrame:
    """Derive a dark-share path from the simulated log dark and lit 
    responses.

    A direct reconstruction of the volume levels can overflow if a 
    simulated log path becomes very large. We therefore work with the 
    equivalent ratio form

        D / (D + L) = 1 / (1 + (L_bar / D_bar) * exp(log L - log D))

    and clip the exponent argument for numerical stability.
    """

    work = irf.copy()
    baseline_dark_share = dark_volume_mean / (dark_volume_mean + lit_volume_mean)
    if dark_volume_mean <= 0.0 or lit_volume_mean < 0.0:
        raise ValueError("Baseline volume means must be non-negative, with dark volume strictly positive.")

    log_ratio = (
        np.log(lit_volume_mean / dark_volume_mean)
        + work["log_lit_volume_t"].to_numpy(dtype=float)
        - work["log_dark_volume_t"].to_numpy(dtype=float)
    )
    # Clipping keeps exp() finite while preserving the qualitative 
    # ranking of very large positive and negative differences.
    log_ratio = np.clip(log_ratio, -700.0, 700.0)
    work["dark_share_level"] = 1.0 / (1.0 + np.exp(log_ratio))
    work["dark_share_change"] = work["dark_share_level"] - baseline_dark_share
    work["dark_share_change_bps"] = 10000.0 * work["dark_share_change"]
    return work


def add_cumulative_columns(irf: pd.DataFrame, *, value_cols: Sequence[str]) -> pd.DataFrame:
    """Add cumulative versions of the selected IRF columns."""

    work = irf.copy()
    for col in value_cols:
        work[f"cum_{col}"] = work[col].cumsum()
    return work

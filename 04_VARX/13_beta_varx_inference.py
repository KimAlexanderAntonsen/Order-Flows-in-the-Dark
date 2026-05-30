"""Step 3 confidence-band helpers for the beta VARX build.

This module adds the inference layer on top of the Step 1 baseline and
the Step 2 IRF implementation. The logic follows the Menkveld 
simulation idea:

1. Treat the point estimate as the center of the parameter 
   distribution.
2. Use the estimated parameter covariance matrix from the fitted VARX.
3. Draw parameter vectors from a multivariate normal distribution.
4. Recompute the IRF for each accepted draw.
5. Form 95% confidence bands from the simulated IRF distribution.

To keep the bands economically meaningful, we reject draws that imply 
an unstable autoregressive system.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_model = importlib.import_module("06_beta_varx_model")
_irf = importlib.import_module("10_beta_varx_irf")

companion_matrix = _model.companion_matrix
add_cumulative_columns = _irf.add_cumulative_columns
add_dark_share_response = _irf.add_dark_share_response
simulate_irf_recursive = _irf.simulate_irf_recursive

# Number of parallel workers for the IRF Monte Carlo loop. Set via env
# var or default to all cores. 
_PARALLEL_N_JOBS = int(os.environ.get("VARX_MC_N_JOBS", "-1"))
_PARALLEL_BACKEND = os.environ.get("VARX_MC_BACKEND", "threading")


@dataclass
class DrawVARXResults:
    """Lightweight result container for one simulated parameter draw.

    The Step 2 IRF code only needs a handful of attributes and helper 
    methods, so this small class mirrors the relevant parts of the 
    Step 1 result object.
    """

    y_cols: tuple[str, ...]
    common_x_cols: tuple[str, ...]
    panel_x_cols: tuple[str, ...]
    regressor_names: list[str]
    coefficients: np.ndarray

    def lag_matrices(self) -> list[np.ndarray]:
        """Return the autoregressive matrices implied by the drawn parameters."""

        k_y = len(self.y_cols)
        lag_names = [name for name in self.regressor_names if name.startswith("Y_L")]
        n_lags = len(lag_names) // k_y
        matrices: list[np.ndarray] = []
        for lag in range(1, n_lags + 1):
            cols = [self.regressor_names.index(f"Y_L{lag}.{col}") for col in self.y_cols]
            matrices.append(self.coefficients[cols, :].T)
        return matrices

    def common_exog_matrix(self) -> np.ndarray | None:
        """Return the coefficient matrix on common exogenous variables."""

        if not self.common_x_cols:
            return None
        cols = [self.regressor_names.index(col) for col in self.common_x_cols]
        return self.coefficients[cols, :].T

    def panel_exog_matrix(self) -> np.ndarray | None:
        """Return the coefficient matrix on firm-specific exogenous variables."""

        if not self.panel_x_cols:
            return None
        cols = [self.regressor_names.index(col) for col in self.panel_x_cols]
        return self.coefficients[cols, :].T


def regularize_parameter_covariance(covariance: np.ndarray, *, jitter: float = 1e-12) -> np.ndarray:
    """Symmetrize and regularize a covariance matrix before simulation draws."""

    cov = 0.5 * (covariance + covariance.T)
    min_eig = float(np.min(np.linalg.eigvalsh(cov)))
    if min_eig < 0.0:
        cov = cov + (-min_eig + jitter) * np.eye(cov.shape[0], dtype=float)
    return cov


def clone_result_with_coefficients(base_result, coefficients: np.ndarray) -> DrawVARXResults:
    """Create a Step-2-compatible result object from a drawn coefficient matrix."""

    return DrawVARXResults(
        y_cols=tuple(base_result.y_cols),
        common_x_cols=tuple(base_result.common_x_cols),
        panel_x_cols=tuple(base_result.panel_x_cols),
        regressor_names=list(base_result.regressor_names),
        coefficients=np.asarray(coefficients, dtype=float),
    )


def coefficient_draw_generator(
    base_result,
    *,
    n_draws: int,
    seed: int,
    require_stable: bool = True,
    max_attempt_multiplier: int = 10,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Draw coefficient matrices and keep only those with stable dynamics."""

    if base_result.parameter_covariance is None:
        raise ValueError("The fitted result does not contain a parameter covariance matrix.")

    rng = np.random.default_rng(seed)
    mean_vector = base_result.coefficients.reshape(-1, order="F")
    covariance = regularize_parameter_covariance(base_result.parameter_covariance)

    accepted: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, float | int | bool]] = []
    max_attempts = int(max_attempt_multiplier) * int(n_draws)

    attempts = 0
    unstable_rejections = 0
    while len(accepted) < n_draws and attempts < max_attempts:
        attempts += 1
        draw_vector = rng.multivariate_normal(mean_vector, covariance)
        draw_beta = draw_vector.reshape(base_result.coefficients.shape, order="F")
        draw_result = clone_result_with_coefficients(base_result, draw_beta)

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion_matrix(draw_result.lag_matrices())))))
        is_stable = bool(spectral_radius < 1.0)
        if require_stable and not is_stable:
            unstable_rejections += 1
            continue

        accepted.append(draw_beta)
        diagnostic_rows.append(
            {
                "draw_number": len(accepted),
                "attempt_number": attempts,
                "spectral_radius": spectral_radius,
                "is_stable": is_stable,
            }
        )

    if len(accepted) < n_draws:
        raise RuntimeError(
            f"Only {len(accepted)} stable draws were accepted out of {attempts} attempts. "
            "Increase max_attempt_multiplier or revisit the covariance approximation."
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.attrs["attempts"] = attempts
    diagnostics.attrs["unstable_rejections"] = unstable_rejections
    return accepted, diagnostics


def finalize_irf_for_inference(
    irf: pd.DataFrame,
    *,
    dark_volume_mean: float,
    lit_volume_mean: float,
) -> pd.DataFrame:
    """Apply the same post-processing used in Step 2 before summarizing bands."""

    out = add_dark_share_response(
        irf,
        dark_volume_mean=dark_volume_mean,
        lit_volume_mean=lit_volume_mean,
    )
    return add_cumulative_columns(
        out,
        value_cols=(
            "log_dark_volume_t",
            "log_lit_volume_t",
            "log_total_realized_variance_t",
            "dark_share_change",
            "dark_share_change_bps",
        ),
    )


def summarize_irf_draws(
    *,
    point_irf: pd.DataFrame,
    draw_irfs: list[pd.DataFrame],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize simulated IRFs into point estimates and confidence bands."""

    if not draw_irfs:
        raise ValueError("At least one simulated IRF draw is required.")

    value_cols = [col for col in point_irf.columns if col != "horizon"]
    draw_stack = np.stack([frame[value_cols].to_numpy(dtype=float) for frame in draw_irfs], axis=0)

    lower = np.quantile(draw_stack, alpha / 2.0, axis=0)
    upper = np.quantile(draw_stack, 1.0 - alpha / 2.0, axis=0)

    summary = pd.DataFrame({"horizon": point_irf["horizon"].to_numpy()})
    for idx, col in enumerate(value_cols):
        summary[f"{col}_point"] = point_irf[col].to_numpy(dtype=float)
        summary[f"{col}_lower95"] = lower[:, idx]
        summary[f"{col}_upper95"] = upper[:, idx]
    return summary


def simulate_irf_bands(
    *,
    base_result,
    shock_path: pd.DataFrame,
    horizon_end: int,
    dark_volume_mean: float,
    lit_volume_mean: float,
    n_draws: int,
    seed: int,
    require_stable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 95% IRF confidence bands by parameter simulation."""

    point_raw = simulate_irf_recursive(base_result, shock_path, horizon_end=horizon_end)
    point_irf = finalize_irf_for_inference(
        point_raw,
        dark_volume_mean=dark_volume_mean,
        lit_volume_mean=lit_volume_mean,
    )

    coefficient_draws, diagnostics = coefficient_draw_generator(
        base_result,
        n_draws=n_draws,
        seed=seed,
        require_stable=require_stable,
    )

    def _one_draw(draw_id: int, draw_beta: np.ndarray) -> pd.DataFrame:
        draw_result = clone_result_with_coefficients(base_result, draw_beta)
        draw_raw = simulate_irf_recursive(draw_result, shock_path, horizon_end=horizon_end)
        draw_final = finalize_irf_for_inference(
            draw_raw,
            dark_volume_mean=dark_volume_mean,
            lit_volume_mean=lit_volume_mean,
        )
        draw_final.insert(0, "draw_id", draw_id)
        return draw_final

    draw_irfs = Parallel(n_jobs=_PARALLEL_N_JOBS, backend=_PARALLEL_BACKEND)(
        delayed(_one_draw)(draw_id, draw_beta)
        for draw_id, draw_beta in enumerate(coefficient_draws, start=1)
    )

    summary = summarize_irf_draws(point_irf=point_irf, draw_irfs=draw_irfs, alpha=0.05)
    return summary, diagnostics

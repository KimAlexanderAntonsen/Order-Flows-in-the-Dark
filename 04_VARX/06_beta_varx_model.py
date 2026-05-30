"""Lean baseline panel VARX estimator for Step 1.

This module focuses only on the baseline model:

    y_{j,t} = alpha_j + sum_{ell=1}^p Phi_ell y_{j,t-ell} + B c_t + G f_{j,t} + eps_{j,t}

where:
    - y_{j,t} is the endogenous market vector,
    - c_t are common exogenous urgency variables,
    - f_{j,t} are firm-specific exogenous urgency variables,
    - alpha_j is a stock fixed effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class BaselineVARXResults:
    """Container for baseline VARX output."""

    y_cols: tuple[str, ...]
    common_x_cols: tuple[str, ...]
    panel_x_cols: tuple[str, ...]
    regressor_names: list[str]
    coefficients: np.ndarray
    coefficient_table: pd.DataFrame
    fitted_values: pd.DataFrame | None
    residuals: pd.DataFrame | None
    xtx: np.ndarray | None
    xtx_inverse: np.ndarray | None
    residual_covariance: np.ndarray | None
    parameter_covariance: np.ndarray | None
    design_info: dict[str, object]
    # Populated by a second-pass clustering routine; left at None until then.
    parameter_covariance_classical: np.ndarray | None = None
    parameter_covariance_entity: np.ndarray | None = None
    parameter_covariance_time: np.ndarray | None = None
    parameter_covariance_hc0: np.ndarray | None = None
    cluster_diagnostics: dict[str, object] | None = None

    def lag_matrices(self) -> list[np.ndarray]:
        """Return the autoregressive coefficient matrices Phi_1, ..., Phi_p."""

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


class BaselinePanelVARX:
    """Simple pooled panel VARX with stock fixed effects.

    The estimator uses a within transformation to remove stock fixed 
    effects. 
    """

    def __init__(self, *, p_lags: int = 2, ridge: float = 1e-8, entity_fixed_effects: bool = True):
        self.p_lags = int(p_lags)
        self.ridge = float(ridge)
        self.entity_fixed_effects = bool(entity_fixed_effects)

    @staticmethod
    def _compute_covariance_objects(
        *,
        xtx: np.ndarray,
        residual_sscp: np.ndarray,
        nobs: int,
        ridge: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute residual and parameter covariance estimates.

        With a common regressor matrix across equations, multivariate 
        OLS implies

            Var(vec(B_hat)) = Sigma_u \\kron (X'X)^{-1}.

        The tiny ridge penalty used in the build is only there for 
        numerical stability, so we use the ridge-adjusted 
        cross-product matrix in the inverse as a practical 
        approximation.
        """

        k_x = xtx.shape[0]
        dof = max(int(nobs) - int(k_x), 1)
        ridge_matrix = ridge * np.eye(k_x, dtype=float) if ridge > 0.0 else 0.0
        xtx_inverse = np.linalg.inv(xtx + ridge_matrix)
        residual_covariance = residual_sscp / float(dof)
        parameter_covariance = np.kron(residual_covariance, xtx_inverse)
        return xtx_inverse, residual_covariance, dof

    def _prepare_piece_sample(
        self,
        piece: pd.DataFrame,
        *,
        entity_col: str,
        time_col: str,
        y_cols: Sequence[str],
        common_x_cols: Sequence[str],
        panel_x_cols: Sequence[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """Create one stock's lagged estimation sample.

        This helper keeps the iterator-based fit and the 
        iterator-based diagnostics perfectly aligned. Any 
        transformation used during fitting should also be used during 
        the later residual and stability checks.
        """

        y_cols = list(y_cols)
        common_x_cols = list(common_x_cols)
        panel_x_cols = list(panel_x_cols)

        work = piece.sort_values([entity_col, time_col]).copy()
        lag_cols: list[str] = []
        for lag in range(1, self.p_lags + 1):
            lagged = work[y_cols].shift(lag)
            lagged.columns = [f"Y_L{lag}.{col}" for col in y_cols]
            work = pd.concat([work, lagged], axis=1)
            lag_cols.extend(lagged.columns.tolist())

        regressor_names = lag_cols + common_x_cols + panel_x_cols
        sample_cols = [entity_col, time_col, *y_cols, *regressor_names]
        sample = work[sample_cols].dropna().copy()

        if sample.empty:
            return sample, regressor_names

        # Each iterator piece is one stock. Demeaning within the piece 
        # is therefore equivalent to removing that stock's fixed 
        # effect.
        if self.entity_fixed_effects:
            sample[y_cols] = sample[y_cols] - sample[y_cols].mean()
            sample[regressor_names] = sample[regressor_names] - sample[regressor_names].mean()

        return sample, regressor_names

    def fit(
        self,
        df: pd.DataFrame,
        *,
        entity_col: str,
        time_col: str,
        y_cols: Sequence[str],
        common_x_cols: Sequence[str] = (),
        panel_x_cols: Sequence[str] = (),
    ) -> BaselineVARXResults:
        """Estimate the baseline panel VARX.

        Exogenous variables enter contemporaneously, matching the 
        compact Menkveld-style formulation used in Step 1.
        """

        y_cols = list(y_cols)
        common_x_cols = list(common_x_cols)
        panel_x_cols = list(panel_x_cols)
        work = df.sort_values([entity_col, time_col]).copy()

        # Create lagged endogenous regressors within each stock.
        grouped = work.groupby(entity_col, sort=False)
        lag_cols: list[str] = []
        for lag in range(1, self.p_lags + 1):
            lagged = grouped[y_cols].shift(lag)
            lagged.columns = [f"Y_L{lag}.{col}" for col in y_cols]
            work = pd.concat([work, lagged], axis=1)
            lag_cols.extend(lagged.columns.tolist())

        regressor_names = lag_cols + common_x_cols + panel_x_cols
        sample_cols = [entity_col, time_col, *y_cols, *regressor_names]
        sample = work[sample_cols].dropna().copy()

        if sample.empty:
            raise ValueError("No observations remain after applying lags and dropping missing values.")

        # Remove stock fixed effects by demeaning within entity.
        if self.entity_fixed_effects:
            sample[y_cols] = sample[y_cols] - sample.groupby(entity_col)[y_cols].transform("mean")
            sample[regressor_names] = sample[regressor_names] - sample.groupby(entity_col)[regressor_names].transform("mean")

        x = sample[regressor_names].to_numpy(dtype=float)
        y = sample[y_cols].to_numpy(dtype=float)
        xtx = x.T @ x

        # A tiny ridge penalty keeps the system numerically stable.
        if self.ridge > 0.0:
            ridge_block = np.sqrt(self.ridge) * np.eye(x.shape[1], dtype=float)
            x_aug = np.vstack([x, ridge_block])
            y_aug = np.vstack([y, np.zeros((x.shape[1], y.shape[1]), dtype=float)])
            beta, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)

        fitted = x @ beta
        residuals = y - fitted
        residual_sscp = residuals.T @ residuals
        xtx_inverse, residual_covariance, covariance_dof = self._compute_covariance_objects(
            xtx=xtx,
            residual_sscp=residual_sscp,
            nobs=len(sample),
            ridge=self.ridge,
        )

        coefficient_table = pd.DataFrame(
            beta,
            index=regressor_names,
            columns=y_cols,
        ).reset_index(names="regressor")

        fitted_values = pd.DataFrame(fitted, columns=y_cols, index=sample.index)
        fitted_values.insert(0, time_col, sample[time_col].to_numpy())
        fitted_values.insert(0, entity_col, sample[entity_col].to_numpy())

        residual_frame = pd.DataFrame(residuals, columns=y_cols, index=sample.index)
        residual_frame.insert(0, time_col, sample[time_col].to_numpy())
        residual_frame.insert(0, entity_col, sample[entity_col].to_numpy())

        design_info = {
            "nobs": int(len(sample)),
            "n_entities": int(sample[entity_col].nunique()),
            "k_endogenous": int(len(y_cols)),
            "k_regressors": int(len(regressor_names)),
            "p_lags": int(self.p_lags),
            "entity_fixed_effects": bool(self.entity_fixed_effects),
            "ridge": float(self.ridge),
            "covariance_dof": int(covariance_dof),
        }

        return BaselineVARXResults(
            y_cols=tuple(y_cols),
            common_x_cols=tuple(common_x_cols),
            panel_x_cols=tuple(panel_x_cols),
            regressor_names=regressor_names,
            coefficients=beta,
            coefficient_table=coefficient_table,
            fitted_values=fitted_values,
            residuals=residual_frame,
            xtx=xtx,
            xtx_inverse=xtx_inverse,
            residual_covariance=residual_covariance,
            parameter_covariance=np.kron(residual_covariance, xtx_inverse),
            design_info=design_info,
        )

    def fit_from_iterator(
        self,
        pieces: Iterable[pd.DataFrame],
        *,
        entity_col: str,
        time_col: str,
        y_cols: Sequence[str],
        common_x_cols: Sequence[str] = (),
        panel_x_cols: Sequence[str] = (),
    ) -> BaselineVARXResults:
        """
        Estimate the baseline VARX from stock-by-stock panel pieces.

        This method is mathematically equivalent to fitting on one 
        combined DataFrame, but it avoids materializing the full stock
        panel at once. 
        """

        y_cols = list(y_cols)
        common_x_cols = list(common_x_cols)
        panel_x_cols = list(panel_x_cols)

        k_y = len(y_cols)
        regressor_names: list[str] | None = None
        xtx: np.ndarray | None = None
        xty: np.ndarray | None = None
        yty: np.ndarray | None = None

        nobs = 0
        n_entities = 0
        first_timestamp = None
        last_timestamp = None

        for piece in pieces:
            sample, piece_regressors = self._prepare_piece_sample(
                piece,
                entity_col=entity_col,
                time_col=time_col,
                y_cols=y_cols,
                common_x_cols=common_x_cols,
                panel_x_cols=panel_x_cols,
            )
            if sample.empty:
                continue

            if regressor_names is None:
                regressor_names = piece_regressors
                k_x = len(regressor_names)
                xtx = np.zeros((k_x, k_x), dtype=float)
                xty = np.zeros((k_x, k_y), dtype=float)
                yty = np.zeros((k_y, k_y), dtype=float)

            x_i = sample[regressor_names].to_numpy(dtype=float)
            y_i = sample[y_cols].to_numpy(dtype=float)

            xtx += x_i.T @ x_i
            xty += x_i.T @ y_i
            yty += y_i.T @ y_i

            nobs += len(sample)
            n_entities += 1
            piece_first = sample[time_col].iloc[0]
            piece_last = sample[time_col].iloc[-1]
            first_timestamp = piece_first if first_timestamp is None else min(first_timestamp, piece_first)
            last_timestamp = piece_last if last_timestamp is None else max(last_timestamp, piece_last)

        if nobs == 0:
            raise ValueError("No observations remain after applying lags and dropping missing values.")

        assert regressor_names is not None
        assert xtx is not None
        assert xty is not None
        assert yty is not None
        k_x = len(regressor_names)
        ridge_matrix = self.ridge * np.eye(k_x, dtype=float) if self.ridge > 0.0 else 0.0
        beta = np.linalg.solve(xtx + ridge_matrix, xty)
        residual_sscp = yty - xty.T @ beta - beta.T @ xty + beta.T @ xtx @ beta
        xtx_inverse, residual_covariance, covariance_dof = self._compute_covariance_objects(
            xtx=xtx,
            residual_sscp=residual_sscp,
            nobs=nobs,
            ridge=self.ridge,
        )

        coefficient_table = pd.DataFrame(
            beta,
            index=regressor_names,
            columns=y_cols,
        ).reset_index(names="regressor")

        design_info = {
            "nobs": int(nobs),
            "n_entities": int(n_entities),
            "k_endogenous": int(k_y),
            "k_regressors": int(k_x),
            "p_lags": int(self.p_lags),
            "entity_fixed_effects": bool(self.entity_fixed_effects),
            "ridge": float(self.ridge),
            "sample_start": None if first_timestamp is None else str(first_timestamp),
            "sample_end": None if last_timestamp is None else str(last_timestamp),
            "fit_method": "iterator_crossproducts",
            "covariance_dof": int(covariance_dof),
        }

        return BaselineVARXResults(
            y_cols=tuple(y_cols),
            common_x_cols=tuple(common_x_cols),
            panel_x_cols=tuple(panel_x_cols),
            regressor_names=regressor_names,
            coefficients=beta,
            coefficient_table=coefficient_table,
            fitted_values=None,
            residuals=None,
            xtx=xtx,
            xtx_inverse=xtx_inverse,
            residual_covariance=residual_covariance,
            parameter_covariance=np.kron(residual_covariance, xtx_inverse),
            design_info=design_info,
        )

    def compute_two_way_clustered_covariance(
        self,
        pieces: Iterable[pd.DataFrame],
        *,
        result: BaselineVARXResults,
        entity_col: str,
        time_col: str,
        y_cols: Sequence[str],
        common_x_cols: Sequence[str] = (),
        panel_x_cols: Sequence[str] = (),
    ) -> dict[str, object]:
        """Second-pass Petersen (2009) two-way clustered parameter 
        covariance.

        We cluster by entity (stock) and by time (minute), and build 
        the Cameron-Gelbach-Miller (2011) two-way estimator

            V_two_way = V_entity + V_time - V_intersection

        where V_intersection uses the (stock, minute) cells as 
        singletons.

        The routine consumes its own iterator, so callers must supply 
        a fresh factory. Residuals are recomputed from the fitted 
        coefficients using the same within-entity demeaning used 
        during estimation, which keeps the meat matrices consistent 
        with the bread ``(X'X)^{-1}``.
        """

        y_cols = list(y_cols)
        common_x_cols = list(common_x_cols)
        panel_x_cols = list(panel_x_cols)

        if result.xtx_inverse is None:
            raise ValueError("Fitted result is missing xtx_inverse; cannot build clustered covariance.")
        if result.coefficients is None:
            raise ValueError("Fitted result is missing coefficients.")

        regressor_names = list(result.regressor_names)
        beta = np.asarray(result.coefficients, dtype=float)
        k_x = beta.shape[0]
        k_y = beta.shape[1]
        km = k_x * k_y
        invxx = np.asarray(result.xtx_inverse, dtype=float)
        bread = np.kron(np.eye(k_y, dtype=float), invxx)

        omega_entity = np.zeros((km, km), dtype=float)
        omega_hc0 = np.zeros((km, km), dtype=float)
        time_sums: dict[np.int64, np.ndarray] = {}

        nobs = 0
        n_entities = 0

        for piece in pieces:
            sample, piece_regressors = self._prepare_piece_sample(
                piece,
                entity_col=entity_col,
                time_col=time_col,
                y_cols=y_cols,
                common_x_cols=common_x_cols,
                panel_x_cols=panel_x_cols,
            )
            if sample.empty:
                continue
            if piece_regressors != regressor_names:
                raise ValueError(
                    "Regressor layout differs between fit and cluster pass; "
                    "check that the same family spec is used."
                )

            x_g = sample[regressor_names].to_numpy(dtype=float)
            y_g = sample[y_cols].to_numpy(dtype=float)
            u_g = y_g - x_g @ beta  # N_g x M
            n_g = x_g.shape[0]

            # Entity meat: s_g = X_g' U_g; column-major vec -> outer product.
            s_entity = x_g.T @ u_g
            s_entity_vec = s_entity.reshape(-1, order="F")
            omega_entity += np.outer(s_entity_vec, s_entity_vec)

            # Row-wise Kronecker (u_i ⊗ x_i), so each row of q_g 
            # equals vec(x_i u_i') in Fortran order. q_g has shape 
            # (N_g, k_y * k_x).
            q_g = (u_g[:, :, None] * x_g[:, None, :]).reshape(n_g, km)
            omega_hc0 += q_g.T @ q_g

            ts_values = pd.to_datetime(sample[time_col]).to_numpy()
            ts_int = ts_values.astype("datetime64[ns]").view("i8")
            frame = pd.DataFrame(q_g)
            frame.insert(0, "__ts__", ts_int)
            grouped = frame.groupby("__ts__", sort=False, as_index=True).sum()
            grouped_values = grouped.to_numpy()
            for key, row in zip(grouped.index.to_numpy(), grouped_values, strict=True):
                existing = time_sums.get(key)
                if existing is None:
                    time_sums[key] = row.copy()
                else:
                    existing += row

            nobs += n_g
            n_entities += 1

        if nobs == 0:
            raise ValueError("No observations available for clustered covariance computation.")

        omega_time = np.zeros((km, km), dtype=float)
        for vec in time_sums.values():
            omega_time += np.outer(vec, vec)

        g_entity = max(int(n_entities), 1)
        g_time = max(int(len(time_sums)), 1)
        # After demeaning, stock fixed effects absorb g_entity degrees of freedom.
        k_full = k_x + g_entity
        dof_scale = (nobs - 1) / max(nobs - k_full, 1)

        c_entity = (g_entity / max(g_entity - 1, 1)) * dof_scale
        c_time = (g_time / max(g_time - 1, 1)) * dof_scale
        c_hc0 = (nobs / max(nobs - 1, 1)) * dof_scale  # singleton intersection

        v_entity = c_entity * bread @ omega_entity @ bread
        v_time = c_time * bread @ omega_time @ bread
        v_hc0 = c_hc0 * bread @ omega_hc0 @ bread

        v_two_way = v_entity + v_time - v_hc0
        # Cameron-Gelbach-Miller (2011) eigenvalue repair: clip 
        # negative eigenvalues to zero so the resulting covariance is 
        # PSD.
        eigvals, eigvecs = np.linalg.eigh(0.5 * (v_two_way + v_two_way.T))
        clipped = np.clip(eigvals, a_min=0.0, a_max=None)
        v_two_way_psd = (eigvecs * clipped) @ eigvecs.T
        min_eig_raw = float(eigvals.min())
        n_negative_eigs = int(np.sum(eigvals < 0.0))

        diagnostics = {
            "nobs": int(nobs),
            "n_entities": int(g_entity),
            "n_time": int(g_time),
            "k_regressors": int(k_x),
            "k_full_with_fe": int(k_full),
            "dof_scale_factor": float(dof_scale),
            "min_eig_raw_two_way": min_eig_raw,
            "n_negative_eigs_two_way": n_negative_eigs,
            "psd_repaired": bool(n_negative_eigs > 0),
        }

        return {
            "parameter_covariance_two_way": v_two_way_psd,
            "parameter_covariance_two_way_raw": v_two_way,
            "parameter_covariance_entity": v_entity,
            "parameter_covariance_time": v_time,
            "parameter_covariance_hc0": v_hc0,
            "cluster_diagnostics": diagnostics,
        }

    def diagnostics_from_iterator(
        self,
        pieces: Iterable[pd.DataFrame],
        *,
        result: BaselineVARXResults,
        entity_col: str,
        time_col: str,
        y_cols: Sequence[str],
        common_x_cols: Sequence[str] = (),
        panel_x_cols: Sequence[str] = (),
    ) -> dict[str, object]:
        """
        Compute post-estimation diagnostics on an iterator-based fit.

        The two diagnostics we care about before Step 2 are:

        1. A lag-selection sanity check using information criteria.
        2. A stability check using the eigenvalues of the companion 
           matrix.
        """

        y_cols = list(y_cols)
        common_x_cols = list(common_x_cols)
        panel_x_cols = list(panel_x_cols)

        k_y = len(y_cols)
        resid_sscp = np.zeros((k_y, k_y), dtype=float)
        nobs = 0

        for piece in pieces:
            sample, regressor_names = self._prepare_piece_sample(
                piece,
                entity_col=entity_col,
                time_col=time_col,
                y_cols=y_cols,
                common_x_cols=common_x_cols,
                panel_x_cols=panel_x_cols,
            )
            if sample.empty:
                continue

            x_i = sample[regressor_names].to_numpy(dtype=float)
            y_i = sample[y_cols].to_numpy(dtype=float)
            resid_i = y_i - x_i @ result.coefficients

            resid_sscp += resid_i.T @ resid_i
            nobs += len(sample)

        if nobs == 0:
            raise ValueError("No observations remain for diagnostics.")

        sigma = resid_sscp / float(nobs)
        sign, logdet = np.linalg.slogdet(sigma)

        # A tiny diagonal jitter prevents numerical edge cases from 
        # producing undefined log-determinants in otherwise 
        # well-behaved samples.
        if sign <= 0:
            sigma = sigma + 1e-12 * np.eye(k_y, dtype=float)
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                raise ValueError("Residual covariance matrix is not positive definite.")

        loglik = -0.5 * nobs * (k_y * np.log(2.0 * np.pi) + logdet + k_y)

        n_regression_params = len(result.regressor_names) * k_y
        n_covariance_params = k_y * (k_y + 1) // 2
        n_params_total = n_regression_params + n_covariance_params

        aic = -2.0 * loglik + 2.0 * n_params_total
        bic = -2.0 * loglik + np.log(float(nobs)) * n_params_total

        companion = companion_matrix(result.lag_matrices())
        eigvals = np.linalg.eigvals(companion)
        spectral_radius = float(np.max(np.abs(eigvals)))

        return {
            "diagnostic_nobs": int(nobs),
            "loglik": float(loglik),
            "aic": float(aic),
            "bic": float(bic),
            "resid_logdet": float(logdet),
            "spectral_radius": spectral_radius,
            "is_stable": bool(spectral_radius < 1.0),
        }


def apply_two_way_clustered_covariance(
    model: "BaselinePanelVARX",
    pieces_factory: Callable[[], Iterable[pd.DataFrame]],
    *,
    result: BaselineVARXResults,
    entity_col: str,
    time_col: str,
    y_cols: Sequence[str],
    common_x_cols: Sequence[str] = (),
    panel_x_cols: Sequence[str] = (),
) -> BaselineVARXResults:
    """Replace the classical parameter covariance with the two-way 
    clustered one.

    The classical Sigma_u ⊗ (X'X)^{-1} estimate remains available on 
    theresult object as ``parameter_covariance_classical`` so it can 
    be inspected for comparison, but ``parameter_covariance`` is 
    updated in place so that downstream inference (Monte Carlo 
    coefficient draws) picks up the Petersen (2009) / 
    Cameron-Gelbach-Miller (2011) standard errors by default.
    """

    cluster_out = model.compute_two_way_clustered_covariance(
        pieces_factory(),
        result=result,
        entity_col=entity_col,
        time_col=time_col,
        y_cols=y_cols,
        common_x_cols=common_x_cols,
        panel_x_cols=panel_x_cols,
    )
    if result.parameter_covariance_classical is None:
        result.parameter_covariance_classical = result.parameter_covariance
    result.parameter_covariance = cluster_out["parameter_covariance_two_way"]
    result.parameter_covariance_entity = cluster_out["parameter_covariance_entity"]
    result.parameter_covariance_time = cluster_out["parameter_covariance_time"]
    result.parameter_covariance_hc0 = cluster_out["parameter_covariance_hc0"]
    result.cluster_diagnostics = cluster_out["cluster_diagnostics"]
    return result


def companion_matrix(lag_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Build the VAR companion matrix from Phi_1, ..., Phi_p.

    A stable VAR requires all eigenvalues of this companion matrix to 
    lie strictly inside the unit circle. Which we check before the
    impulse responses are interpreted.
    """

    if not lag_matrices:
        raise ValueError("At least one lag matrix is required.")

    k_y = lag_matrices[0].shape[0]
    p_lags = len(lag_matrices)
    top = np.hstack(lag_matrices)
    if p_lags == 1:
        return top

    lower = np.zeros((k_y * (p_lags - 1), k_y * p_lags), dtype=float)
    lower[:, :-k_y] = np.eye(k_y * (p_lags - 1), dtype=float)
    return np.vstack([top, lower])

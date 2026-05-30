"""Presentation helpers for H1/H2 estimation results.

This module sits on top of the saved H1/H2 outputs and turns them into
Menkveld-inspired tables and figures.

The goal is to:

1. Load the already-saved regime-specific IRF bands.
2. Build a small number of summary tables that are easy to inspect.
3. Plot full-sample figures in a style that is close to Menkveld.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ESTIMATION_DIR = Path(__file__).resolve().parent
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

_config = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")


H1H2_OUTPUT_DIR = _config.OUTPUT_DIR
PRESENTATION_DIR = ESTIMATION_DIR / "output" / "presentation"
TABLE_DIR = PRESENTATION_DIR / "tables"
FIGURE_DIR = PRESENTATION_DIR / "figures"
ROBUSTNESS_DIR = ESTIMATION_DIR / "output" / "robustness_p3"

PRE_WINDOW = _config.PRE_WINDOW
POST_WINDOW = _config.POST_WINDOW

FAMILY_SPECS = _helpers.FAMILY_SPECS
FAMILY_NAMES = _config.FAMILY_NAMES
fit_family_regime = _helpers.fit_family_regime
load_sp500_universe = _helpers.load_sp500_universe


def ensure_output_dirs() -> None:
    """Create the presentation output folders if they are missing."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def apply_menkveld_style() -> None:
    """Use a simple publication-like plotting style, like Menkveld."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def load_band_table(family: str, shock_name: str, regime: str) -> pd.DataFrame:
    """Load one saved IRF-bands table."""

    path = H1H2_OUTPUT_DIR / f"h1h2_{family}_{shock_name}_{regime}_bands.csv"
    return pd.read_csv(path)


def load_run_summary() -> pd.DataFrame:
    """Load the saved H1/H2 run summary."""

    return pd.read_csv(H1H2_OUTPUT_DIR / "h1h2_run_summary.csv")


def _lit_level_from_dark(frame: pd.DataFrame) -> pd.DataFrame:
    """Create lit-share levels and changes from the saved dark-share 
    outputs.

    Our reduced system saves dark-share outputs directly. Lit share is 
    the natural two-venue complement, so we derive it here.
    """

    out = frame.copy()
    out["lit_share_level_point"] = 1.0 - out["dark_share_level_point"]
    out["lit_share_level_lower95"] = 1.0 - out["dark_share_level_upper95"]
    out["lit_share_level_upper95"] = 1.0 - out["dark_share_level_lower95"]

    out["lit_share_change_bps_point"] = -out["dark_share_change_bps_point"]
    out["lit_share_change_bps_lower95"] = -out["dark_share_change_bps_upper95"]
    out["lit_share_change_bps_upper95"] = -out["dark_share_change_bps_lower95"]
    return out


def _steady_state_dark_share(frame: pd.DataFrame) -> float:
    """Recover the regime-specific steady-state dark share.

    The saved levels satisfy

        level = steady_state + change,

    so we back out the steady-state level from the saved point paths.
    """

    baseline = frame["dark_share_level_point"] - frame["dark_share_change_point"]
    return float(baseline.mean())


def _plot_spec(family: str) -> dict[str, object]:
    """Return the family-specific plotting choices."""

    if family == "vix":
        return {
            "shock_name": "dVIX_pos_inv",
            "horizons": list(range(0, 6)),
            "xlabel": "Minutes after shock",
            "title": "VIX shock",
        }
    if family == "macro":
        return {
            "shock_name": "macro_event_path",
            "horizons": list(range(-1, 5)),
            "xlabel": "Minutes relative to announcement",
            "title": "Macro news release",
        }
    if family == "earnings":
        return {
            "shock_name": "earnings_event_path",
            "horizons": list(range(1, 14)),
            "xlabel": "Time-of-day (in half-hour intervals)",
            "title": "Earnings surprise",
        }
    raise ValueError(f"Unknown family: {family}")


def aggregate_earnings_to_blocks(frame: pd.DataFrame, *, block_minutes: int = 30) -> pd.DataFrame:
    """Convert minute-level earnings paths into the 13 half-hour 
    blocks.

    Menkveld discuss earnings responses in half-hour blocks rather 
    than in every minute. Our Step 2 and Step 3 code simulate the 
    minute grid, so the presentation layer averages within each 
    half-hour block to recover a Menkveld-like display object.
    """

    work = frame.copy()
    work["horizon"] = work["horizon"].astype(int)
    block_horizons = [k * block_minutes for k in range(1, 14)]
    work = work[work["horizon"].isin(block_horizons)].copy()
    work["block"] = (work["horizon"] // block_minutes).astype(int)
    work = work.drop(columns=["horizon"]).rename(columns={"block": "horizon"})
    work = work.sort_values("horizon").reset_index(drop=True)
    return work


def prepare_plot_frame(family: str, regime: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and reshape one family/regime output for plotting."""

    spec = _plot_spec(family)
    frame = load_band_table(family, str(spec["shock_name"]), regime)
    if family == "earnings":
        frame = aggregate_earnings_to_blocks(frame)
    frame = frame[frame["horizon"].isin(spec["horizons"])].copy()
    frame = _lit_level_from_dark(frame)
    return frame, spec


def _format_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    """Turn a share into percentage points for plotting."""

    return 100.0 * np.asarray(values, dtype=float)


def _bar_with_errors(
    ax,
    x: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Draw Menkveld-inspired white bars with blue outlines and black errors."""

    ax.bar(
        x,
        point,
        color="white",
        edgecolor="blue",
        linewidth=0.8,
        width=0.72,
        zorder=2,
    )
    yerr = np.vstack([point - lower, upper - point])
    ax.errorbar(
        x,
        point,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=1.8,
        zorder=3,
    )


def _set_level_axis_limits(
    ax,
    *,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    steady_level: float,
) -> None:
    """Zoom the y-axis around the steady state and confidence band.

    Menkveld's level figures do not start from zero. They zoom around 
    the relevant share range so the response is visible. We follow 
    that principle here instead of forcing a zero-based axis.
    """

    values = np.concatenate([point, lower, upper, np.array([steady_level], dtype=float)])
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1e-6)
    margin = max(0.15 * span, 0.08)
    ax.set_ylim(low - margin, high + margin)


def _set_difference_axis_limits(
    ax,
    *,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Zoom the difference figure around the simulated confidence band."""

    values = np.concatenate([point, lower, upper, np.array([0.0], dtype=float)])
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1e-6)
    margin = max(0.12 * span, 0.1)
    ax.set_ylim(low - margin, high + margin)


def plot_pre_post_levels(family: str) -> list[Path]:
    """Plot pre/post level figures in a Menkveld-inspired layout.

    We use two rows instead of Menkveld's three because our reduced 
    system is built around dark share and its lit-share complement.
    """

    pre_frame, spec = prepare_plot_frame(family, "pre")
    post_frame, _ = prepare_plot_frame(family, "post")

    apply_menkveld_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    frames = [pre_frame, post_frame]
    titles = ["(a) Pre-window", "(b) Post-window"]
    row_specs = [
        ("Dark shares", "dark_share_level"),
        ("Lit shares", "lit_share_level"),
    ]

    for col_idx, frame in enumerate(frames):
        x = frame["horizon"].to_numpy(dtype=int)
        steady_dark = _steady_state_dark_share(frame)
        steady_levels = {
            "dark_share_level": 100.0 * steady_dark,
            "lit_share_level": 100.0 * (1.0 - steady_dark),
        }
        for row_idx, (ylabel, prefix) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            point = _format_percent(frame[f"{prefix}_point"])
            lower = _format_percent(frame[f"{prefix}_lower95"])
            upper = _format_percent(frame[f"{prefix}_upper95"])
            _bar_with_errors(ax, x, point, lower, upper)
            ax.axhline(steady_levels[prefix], color="black", linestyle=(0, (4, 3)), linewidth=0.8)
            _set_level_axis_limits(
                ax,
                point=point,
                lower=lower,
                upper=upper,
                steady_level=steady_levels[prefix],
            )
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=12, fontweight="bold", pad=6)
            ax.set_xlabel(str(spec["xlabel"]))
            ax.text(0.02, 0.98, "Volume share, %", transform=ax.transAxes, ha="left", va="top", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xlim(float(x.min()) - 0.6, float(x.max()) + 0.6)

    base = FIGURE_DIR / f"h1h2_{family}_pre_post_levels"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def plot_post_minus_pre_difference(family: str) -> list[Path]:
    """Plot Menkveld-style difference figures for the regime comparison."""

    spec = _plot_spec(family)
    diff_frame = load_band_table(family, str(spec["shock_name"]), "post_minus_pre")
    if family == "earnings":
        diff_frame = aggregate_earnings_to_blocks(diff_frame)
    diff_frame = diff_frame[diff_frame["horizon"].isin(spec["horizons"])].copy()
    diff_frame = _lit_level_from_dark(diff_frame)

    apply_menkveld_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    panels = [
        ("Post minus pre: Dark shares", "dark_share_change_bps"),
        ("Post minus pre: Lit shares", "lit_share_change_bps"),
    ]

    x = diff_frame["horizon"].to_numpy(dtype=int)
    for ax, (title, prefix) in zip(axes, panels, strict=True):
        point = diff_frame[f"{prefix}_point"].to_numpy(dtype=float)
        lower = diff_frame[f"{prefix}_lower95"].to_numpy(dtype=float)
        upper = diff_frame[f"{prefix}_upper95"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color="0.8", alpha=1.0, zorder=1)
        ax.plot(x, point, color="black", linewidth=1.0, zorder=2)
        ax.axhline(0.0, color="black", linestyle=(0, (4, 3)), linewidth=0.8)
        _set_difference_axis_limits(
            ax,
            point=point,
            lower=lower,
            upper=upper,
        )
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel(str(spec["xlabel"]))
        ax.text(0.02, 0.98, "Sensitivity difference, bps", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xlim(float(x.min()) - 0.25, float(x.max()) + 0.25)

    base = FIGURE_DIR / f"h1h2_{family}_post_minus_pre_difference"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def _stars_from_pvalue(pvalue: float) -> str:
    """Convert p-values to Menkveld-style significance stars."""

    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _pretty_regressor_name(name: str) -> str:
    """Map internal regressor names to readable table labels."""

    replacements = {
        "log_dark_volume_t": "LogDark",
        "log_lit_volume_t": "LogLit",
        "log_total_realized_variance_t": "LogRealVar",
        "dVIX_pos_inv": "dVIX+",
        "dVIX_neg_inv": "dVIX-",
        "VIX_close": "VIX",
        "pre_news_1min": "PreNews1min",
        "post_news_0min": "PostNews0min",
        "post_news_1min": "PostNews1min",
        "post_news_2min": "PostNews2min",
        "post_news_3min": "PostNews3min",
        "post_news_4min": "PostNews4min",
    }
    if name.startswith("Y_L1."):
        return f"{_pretty_regressor_name(name.split('.', 1)[1])} (-1)"
    if name.startswith("Y_L2."):
        return f"{_pretty_regressor_name(name.split('.', 1)[1])} (-2)"
    if name.startswith("post_ea_"):
        suffix = name.split("_")[-1]
        return f"PostEA{suffix}"
    return replacements.get(name, name)


def build_parameter_inference_table(result, *, family: str, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tidy and a formatted VARX coefficient table.

    The formatted version is meant to resemble a compact 
    Menkveld-style coefficient table. The tidy version keeps the raw 
    inference objects for later use.
    """

    regressor_names = list(result.regressor_names)
    y_cols = list(result.y_cols)
    k_x = len(regressor_names)
    k_y = len(y_cols)

    if result.parameter_covariance is None:
        raise ValueError("Result does not carry a parameter covariance matrix.")

    beta = np.asarray(result.coefficients, dtype=float)
    diag = np.diag(np.asarray(result.parameter_covariance, dtype=float))
    std_err = np.sqrt(diag).reshape((k_x, k_y), order="F")
    t_stat = beta / std_err

    dof = int(result.design_info.get("covariance_dof", 1))
    p_value = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df=dof))

    tidy_rows: list[dict[str, object]] = []
    formatted = pd.DataFrame({"regressor": [_pretty_regressor_name(name) for name in regressor_names]})
    for col_idx, y_col in enumerate(y_cols):
        values = []
        for row_idx, reg_name in enumerate(regressor_names):
            stars = _stars_from_pvalue(float(p_value[row_idx, col_idx]))
            values.append(f"{beta[row_idx, col_idx]:.3f}{stars}")
            tidy_rows.append(
                {
                    "family": family,
                    "regime": regime,
                    "regressor": reg_name,
                    "regressor_label": _pretty_regressor_name(reg_name),
                    "equation": y_col,
                    "coefficient": float(beta[row_idx, col_idx]),
                    "std_error": float(std_err[row_idx, col_idx]),
                    "t_stat": float(t_stat[row_idx, col_idx]),
                    "p_value": float(p_value[row_idx, col_idx]),
                    "stars": stars,
                }
            )
        formatted[y_col] = values

    return pd.DataFrame(tidy_rows), formatted


def build_all_varx_tables() -> tuple[pd.DataFrame, list[Path]]:
    """Refit the regime models and save compact coefficient tables.

    The saved H1/H2 coefficient CSVs only contain point estimates. To 
    build Menkveld-style tables with significance markers, we refit 
    the six regime models here and compute the standard inference 
    objects from the stored parameter covariance matrix.
    """

    tickers = load_sp500_universe()
    outputs: list[Path] = []
    long_tables: list[pd.DataFrame] = []
    for family_name in FAMILY_NAMES:
        family_spec = FAMILY_SPECS[family_name]
        for regime_name, window in (("pre", PRE_WINDOW), ("post", POST_WINDOW)):
            result = fit_family_regime(family_spec, tickers=tickers, window=window)
            tidy, formatted = build_parameter_inference_table(
                result,
                family=family_name,
                regime=regime_name,
            )
            tidy_path = TABLE_DIR / f"h1h2_{family_name}_{regime_name}_varx_inference_tidy.csv"
            wide_path = TABLE_DIR / f"h1h2_{family_name}_{regime_name}_varx_table.csv"
            tidy.to_csv(tidy_path, index=False)
            formatted.to_csv(wide_path, index=False)
            outputs.extend([tidy_path, wide_path])
            long_tables.append(tidy)

    combined = pd.concat(long_tables, ignore_index=True)
    combined_path = TABLE_DIR / "h1h2_varx_inference_long.csv"
    combined.to_csv(combined_path, index=False)
    outputs.append(combined_path)
    return combined, outputs


def build_key_irf_summary_table() -> Path:
    """Build one compact table of the main IRF objects used for interpretation."""

    rows: list[pd.DataFrame] = []
    for family in FAMILY_NAMES:
        spec = _plot_spec(family)
        for regime in ("pre", "post", "post_minus_pre"):
            frame = load_band_table(family, str(spec["shock_name"]), regime)
            if family == "earnings":
                frame = aggregate_earnings_to_blocks(frame)
            frame = frame[frame["horizon"].isin(spec["horizons"])].copy()
            frame = _lit_level_from_dark(frame)
            keep = [
                "regime",
                "family",
                "shock_name",
                "horizon",
                "dark_share_level_point",
                "dark_share_level_lower95",
                "dark_share_level_upper95",
                "lit_share_level_point",
                "lit_share_level_lower95",
                "lit_share_level_upper95",
                "dark_share_change_bps_point",
                "dark_share_change_bps_lower95",
                "dark_share_change_bps_upper95",
                "lit_share_change_bps_point",
                "lit_share_change_bps_lower95",
                "lit_share_change_bps_upper95",
            ]
            rows.append(frame[keep])

    summary = pd.concat(rows, ignore_index=True)
    path = TABLE_DIR / "h1h2_key_irf_summary.csv"
    summary.to_csv(path, index=False)
    return path


def build_estimation_sample_table() -> Path:
    """Save a lightly cleaned version of the benchmark run summary."""

    summary = load_run_summary().copy()
    summary = summary.rename(
        columns={
            "family": "urgency_family",
            "shock_name": "shock_design",
            "pre_nobs": "pre_observations",
            "post_nobs": "post_observations",
            "pre_entities": "pre_entities",
            "post_entities": "post_entities",
        }
    )
    path = TABLE_DIR / "h1h2_estimation_sample_summary.csv"
    summary.to_csv(path, index=False)
    return path


def _evaluate_sign_and_bands(
    frame: pd.DataFrame,
    *,
    horizons: list[int],
    predicted_sign: str,
) -> dict[str, object]:
    """Evaluate sign direction and 95% bands over a key-horizon 
    window.

    This is a compact reading aid where the goal is to summarize 
    whether the headline pattern appears in the direction predicted by 
    the hypothesis and whether the 95% bands exclude zero over the 
    family-specific horizons used in the write-up.
    """

    window = frame[frame["horizon"].isin(horizons)].copy()
    if window.empty:
        raise ValueError("No rows remain after applying the key-horizon window.")

    point = window["dark_share_change_bps_point"].to_numpy(dtype=float)
    lower = window["dark_share_change_bps_lower95"].to_numpy(dtype=float)
    upper = window["dark_share_change_bps_upper95"].to_numpy(dtype=float)

    if predicted_sign == "negative":
        sign_match = point < 0.0
        band_match = upper < 0.0
    elif predicted_sign == "positive":
        sign_match = point > 0.0
        band_match = lower > 0.0
    else:
        raise ValueError(f"Unknown predicted sign: {predicted_sign}")

    return {
        "n_horizons": int(len(window)),
        "n_sign_match": int(sign_match.sum()),
        "n_band_match": int(band_match.sum()),
        "all_sign_match": bool(sign_match.all()),
        "all_band_match": bool(band_match.all()),
    }


def build_hypothesis_support_table() -> Path:
    """Create a compact H1/H2 support table using the benchmark 95% 
    bands.

    The table is a reading aid and summarizes whether the relevant 
    horizons line up with the predicted sign and whether the 95% 
    simulation bands exclude zero in that direction.
    """

    benchmark = pd.read_csv(TABLE_DIR / "h1h2_key_irf_summary.csv")

    p3_summary_path = ROBUSTNESS_DIR / "h1h2_p3_key_irf_summary.csv"
    p3_summary = pd.read_csv(p3_summary_path) if p3_summary_path.exists() else None

    family_windows = {
        "vix": {
            "h1": [0, 1, 2],
            "h2": [0, 1, 2],
        },
        "macro": {
            "h1": [0, 1, 2, 3, 4],
            "h2": [0, 1, 2, 3, 4],
        },
        "earnings": {
            "h1": list(range(1, 13)),
            "h2": list(range(1, 13)),
        },
    }

    row_notes = {
        "vix": {
            "h1_note": "Pre-window dark share falls on impact, which matches the Menkveld-style pecking-order prediction.",
            "h2_note": "Post-window dark share becomes slightly more dark-negative than in the pre-window, so the attenuation prediction is not supported.",
        },
        "macro": {
            "h1_note": "Pre-window macro releases are associated with a clear dark-share decline over the key event window.",
            "h2_note": "The post-minus-pre response is positive throughout the main event window, which is the strongest H2 result in the project.",
        },
        "earnings": {
            "h1_note": "Pre-window earnings responses are dark-positive in the reduced system, so the Menkveld-style dark-negative H1 prediction is not met.",
            "h2_note": "Post-window earnings responses are attenuated, but not in a way that cleanly supports the strict Menkveld-style H2 prediction.",
        },
    }

    rows: list[dict[str, object]] = []
    for family in FAMILY_NAMES:
        h1_frame = benchmark[(benchmark["family"] == family) & (benchmark["regime"] == "pre")].copy()
        h2_frame = benchmark[(benchmark["family"] == family) & (benchmark["regime"] == "post_minus_pre")].copy()

        h1_eval = _evaluate_sign_and_bands(
            h1_frame,
            horizons=family_windows[family]["h1"],
            predicted_sign="negative",
        )
        h2_eval = _evaluate_sign_and_bands(
            h2_frame,
            horizons=family_windows[family]["h2"],
            predicted_sign="positive",
        )

        h1_assessment = "Supported" if h1_eval["all_band_match"] else "Not supported"
        h2_assessment = "Supported" if h2_eval["all_band_match"] else "Not supported"

        p3_consistency = "Not run"
        if p3_summary is not None:
            p3_h1 = p3_summary[(p3_summary["family"] == family) & (p3_summary["regime"] == "pre")].copy()
            p3_h2 = p3_summary[(p3_summary["family"] == family) & (p3_summary["regime"] == "post_minus_pre")].copy()
            p3_h1_eval = _evaluate_sign_and_bands(
                p3_h1,
                horizons=family_windows[family]["h1"],
                predicted_sign="negative",
            )
            p3_h2_eval = _evaluate_sign_and_bands(
                p3_h2,
                horizons=family_windows[family]["h2"],
                predicted_sign="positive",
            )
            same_h1 = bool(p3_h1_eval["all_band_match"] == h1_eval["all_band_match"])
            same_h2 = bool(p3_h2_eval["all_band_match"] == h2_eval["all_band_match"])
            p3_consistency = "Yes" if same_h1 and same_h2 else "No"

        rows.append(
            {
                "family": family,
                "h1_key_horizons": ",".join(str(h) for h in family_windows[family]["h1"]),
                "h1_predicted_sign": "negative",
                "h1_all_key_horizons_match_sign": "Yes" if h1_eval["all_sign_match"] else "No",
                "h1_95pct_bands_exclude_zero_in_predicted_direction": "Yes" if h1_eval["all_band_match"] else "No",
                "h1_assessment": h1_assessment,
                "h1_note": row_notes[family]["h1_note"],
                "h2_key_horizons": ",".join(str(h) for h in family_windows[family]["h2"]),
                "h2_predicted_sign": "positive",
                "h2_all_key_horizons_match_sign": "Yes" if h2_eval["all_sign_match"] else "No",
                "h2_95pct_bands_exclude_zero_in_predicted_direction": "Yes" if h2_eval["all_band_match"] else "No",
                "h2_assessment": h2_assessment,
                "h2_note": row_notes[family]["h2_note"],
                "p3_robustness_same_assessment": p3_consistency,
                "important_caveat": (
                    "This is a summary classification based on family-specific key horizons."
                ),
            }
        )

    summary = pd.DataFrame(rows)
    path = TABLE_DIR / "h1h2_hypothesis_support_summary.csv"
    summary.to_csv(path, index=False)
    return path


def run_presentation_layer() -> dict[str, list[Path] | Path]:
    """Create the first Menkveld-inspired tables and figures for H1/H2."""

    ensure_output_dirs()

    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    table_paths.append(build_estimation_sample_table())
    table_paths.append(build_key_irf_summary_table())
    table_paths.append(build_hypothesis_support_table())
    _, coef_paths = build_all_varx_tables()
    table_paths.extend(coef_paths)

    for family in FAMILY_NAMES:
        figure_paths.extend(plot_pre_post_levels(family))
        figure_paths.extend(plot_post_minus_pre_difference(family))

    return {
        "tables": table_paths,
        "figures": figure_paths,
    }

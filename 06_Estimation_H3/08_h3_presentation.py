"""Presentation helpers for the H3 estimation layer.

This module sits on top of the saved H3 benchmark outputs and turns 
them into figures and summary tables.

The design follows the same basic style as the H1/H2 presentation 
layer:

1. keep the same plotting style,
2. separate main-text objects from appendix-style detail,
3. stay transparent about what is benchmark evidence and what is 
   interpretation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

_config = importlib.import_module("01_h3_config")


ESTIMATION_OUTPUT_DIR = _config.ESTIMATION_OUTPUT_DIR
PRESENTATION_TABLE_DIR = _config.PRESENTATION_TABLE_DIR
PRESENTATION_FIGURE_DIR = _config.PRESENTATION_FIGURE_DIR


def ensure_output_dirs() -> None:
    """Create the presentation output folders if they are missing."""

    PRESENTATION_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def apply_menkveld_style() -> None:
    """Use the same publication-like plotting style as in H1/H2."""

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


def _lit_from_dark(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive lit-share levels and changes from the saved dark-share outputs."""

    out = frame.copy()
    if "lit_share_change_bps_point" not in out:
        out["lit_share_change_bps_point"] = -out["dark_share_change_bps_point"]
        out["lit_share_change_bps_lower95"] = -out["dark_share_change_bps_upper95"]
        out["lit_share_change_bps_upper95"] = -out["dark_share_change_bps_lower95"]
    if "lit_share_level_point" not in out and "dark_share_level_point" in out:
        out["lit_share_level_point"] = 1.0 - out["dark_share_level_point"]
        out["lit_share_level_lower95"] = 1.0 - out["dark_share_level_upper95"]
        out["lit_share_level_upper95"] = 1.0 - out["dark_share_level_lower95"]
    return out


def aggregate_earnings_to_blocks(frame: pd.DataFrame, *, block_minutes: int = 30) -> pd.DataFrame:
    """Reduce minute-level earnings paths to the 13 half-hour block boundaries.

    Block ``k`` is the point estimate at minute ``k * block_minutes`` (block 1 =
    h=30, ..., block 13 = h=390). 
    """

    work = frame.copy()
    work["horizon"] = work["horizon"].astype(int)
    block_horizons = [k * block_minutes for k in range(1, 14)]
    work = work[work["horizon"].isin(block_horizons)].copy()
    work["block"] = (work["horizon"] // block_minutes).astype(int)
    work = work.drop(columns=["horizon"]).rename(columns={"block": "horizon"})
    work = work.sort_values("horizon").reset_index(drop=True)
    return work


def _load_group_change_table(family: str, group_name: str) -> pd.DataFrame:
    """Load one group's post-minus-pre bands table for a family."""

    shock_name = str(_plot_spec(family)["shock_name"])
    path = ESTIMATION_OUTPUT_DIR / f"h3_{family}_{shock_name}_{group_name}_post_minus_pre_bands.csv"
    frame = pd.read_csv(path)
    if family == "earnings":
        frame = aggregate_earnings_to_blocks(frame)
    frame = frame[frame["horizon"].isin(_plot_spec(family)["horizons"])].copy()
    return _lit_from_dark(frame)


def _load_h3_table(family: str) -> pd.DataFrame:
    """Load the final treated-minus-control post-minus-pre bands table."""

    shock_name = str(_plot_spec(family)["shock_name"])
    path = (
        ESTIMATION_OUTPUT_DIR
        / f"h3_{family}_{shock_name}_treated_minus_control_post_minus_pre_bands.csv"
    )
    frame = pd.read_csv(path)
    if family == "earnings":
        frame = aggregate_earnings_to_blocks(frame)
    frame = frame[frame["horizon"].isin(_plot_spec(family)["horizons"])].copy()
    return _lit_from_dark(frame)


def _bar_with_errors(
    ax,
    x: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Draw white bars with blue outlines and thin black error bars."""

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


def _set_difference_axis_limits(
    ax,
    *,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Zoom the y-axis around the relevant confidence band."""

    values = np.concatenate([point, lower, upper, np.array([0.0], dtype=float)])
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1e-6)
    margin = max(0.12 * span, 0.1)
    ax.set_ylim(low - margin, high + margin)


def plot_group_change_components(family: str) -> list[Path]:
    """Plot the treated and matched-control post-minus-pre responses.

    This figure is most useful for the appendix because it shows the 
    two building blocks of the H3 difference in difference object.
    """

    spec = _plot_spec(family)
    treated = _load_group_change_table(family, "treated")
    control = _load_group_change_table(family, "matched_control")

    apply_menkveld_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    frames = [treated, control]
    titles = ["(a) Treated", "(b) Matched control"]
    row_specs = [
        ("Dark shares", "dark_share_change_bps"),
        ("Lit shares", "lit_share_change_bps"),
    ]

    for col_idx, frame in enumerate(frames):
        x = frame["horizon"].to_numpy(dtype=int)
        for row_idx, (ylabel, prefix) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            point = frame[f"{prefix}_point"].to_numpy(dtype=float)
            lower = frame[f"{prefix}_lower95"].to_numpy(dtype=float)
            upper = frame[f"{prefix}_upper95"].to_numpy(dtype=float)
            _bar_with_errors(ax, x, point, lower, upper)
            ax.axhline(0.0, color="black", linestyle=(0, (4, 3)), linewidth=0.8)
            _set_difference_axis_limits(ax, point=point, lower=lower, upper=upper)
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=12, fontweight="bold", pad=6)
            ax.set_xlabel(str(spec["xlabel"]))
            ax.text(
                0.02,
                0.98,
                "Sensitivity difference, bps",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xlim(float(x.min()) - 0.6, float(x.max()) + 0.6)

    base = PRESENTATION_FIGURE_DIR / f"h3_{family}_group_post_minus_pre_components"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def plot_h3_triple_difference(family: str) -> list[Path]:
    """Plot the benchmark H3 difference in difference response."""

    spec = _plot_spec(family)
    frame = _load_h3_table(family)

    apply_menkveld_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    panels = [
        ("Treated minus control: Dark shares", "dark_share_change_bps"),
        ("Treated minus control: Lit shares", "lit_share_change_bps"),
    ]

    x = frame["horizon"].to_numpy(dtype=int)
    for ax, (title, prefix) in zip(axes, panels, strict=True):
        point = frame[f"{prefix}_point"].to_numpy(dtype=float)
        lower = frame[f"{prefix}_lower95"].to_numpy(dtype=float)
        upper = frame[f"{prefix}_upper95"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color="0.8", alpha=1.0, zorder=1)
        ax.plot(x, point, color="black", linewidth=1.0, zorder=2)
        ax.axhline(0.0, color="black", linestyle=(0, (4, 3)), linewidth=0.8)
        _set_difference_axis_limits(ax, point=point, lower=lower, upper=upper)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel(str(spec["xlabel"]))
        ax.text(
            0.02,
            0.98,
            "Triple difference, bps",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.set_xticks(x)
        ax.set_xlim(float(x.min()) - 0.25, float(x.max()) + 0.25)

    base = PRESENTATION_FIGURE_DIR / f"h3_{family}_triple_difference"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def build_estimation_sample_table() -> Path:
    """Save a compact benchmark H3 sample table for the write-up."""

    summary = pd.read_csv(ESTIMATION_OUTPUT_DIR / "h3_run_summary.csv")
    rows = [
        {
            "family": "VIX",
            "shock_design": "+/- dVIX innovations",
            "row": summary[summary["family"] == "vix"].iloc[0],
        },
        {
            "family": "Macro",
            "shock_design": "Scheduled macro event path",
            "row": summary[summary["family"] == "macro"].iloc[0],
        },
        {
            "family": "Earnings",
            "shock_design": "1% earnings-surprise path",
            "row": summary[summary["family"] == "earnings"].iloc[0],
        },
    ]

    clean_rows: list[dict[str, object]] = []
    for item in rows:
        row = item["row"]
        clean_rows.append(
            {
                "family": item["family"],
                "shock_design": item["shock_design"],
                "benchmark_p_lags": int(row["benchmark_p_lags"]),
                "n_draws": int(row["n_draws"]),
                "treated_pre_observations": int(row["treated_pre_nobs"]),
                "treated_post_observations": int(row["treated_post_nobs"]),
                "control_pre_observations": int(row["control_pre_nobs"]),
                "control_post_observations": int(row["control_post_nobs"]),
            }
        )

    path = PRESENTATION_TABLE_DIR / "h3_estimation_sample_summary.csv"
    pd.DataFrame(clean_rows).to_csv(path, index=False)
    return path


def build_key_triple_difference_table() -> Path:
    """Save a cleaned copy of the key benchmark H3 difference in difference rows."""

    frame = pd.read_csv(ESTIMATION_OUTPUT_DIR / "h3_key_triple_difference_summary.csv").copy()
    path = PRESENTATION_TABLE_DIR / "h3_key_triple_difference_summary.csv"
    frame.to_csv(path, index=False)
    return path


def build_group_change_key_summary() -> Path:
    """Save a compact appendix table of treated and control post-minus-pre paths."""

    rows: list[pd.DataFrame] = []
    for family in ("vix", "macro", "earnings"):
        for group in ("treated", "matched_control"):
            frame = _load_group_change_table(family, group)
            keep = [
                "family",
                "shock_name",
                "group",
                "regime",
                "horizon",
                "dark_share_change_bps_point",
                "dark_share_change_bps_lower95",
                "dark_share_change_bps_upper95",
                "lit_share_change_bps_point",
                "lit_share_change_bps_lower95",
                "lit_share_change_bps_upper95",
            ]
            rows.append(frame[keep])

    summary = pd.concat(rows, ignore_index=True)
    path = PRESENTATION_TABLE_DIR / "h3_group_change_key_summary.csv"
    summary.to_csv(path, index=False)
    return path


def build_benchmark_reading_table() -> Path:
    """Create the structural reading guide for the benchmark H3 
    results.

    Only records the key horizon window and the planned presentation 
    role per family. The numeric verdict is left to 
    ``h3_robustness_dashboard.csv`` so the two sources cannot drift 
    apart.
    """

    rows = [
        {
            "family": "VIX",
            "key_window": "2-4 min",
            "presentation_role": "Short main-text discussion or appendix figure",
            "appendix_role": "Triple-difference figure, component figure, and key summary table",
        },
        {
            "family": "Macro",
            "key_window": "3 min",
            "presentation_role": "Short main-text discussion or appendix figure",
            "appendix_role": "Triple-difference figure, component figure, and key summary table",
        },
        {
            "family": "Earnings",
            "key_window": "Blocks 1-5 and 8-10",
            "presentation_role": "Headline main-text figure",
            "appendix_role": "Component figure, key summary table, and robustness tables",
        },
    ]
    path = PRESENTATION_TABLE_DIR / "h3_benchmark_reading_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_presentation_layer() -> dict[str, list[Path]]:
    """Build the H3 presentation outputs and return their paths."""

    ensure_output_dirs()

    table_paths = [
        build_estimation_sample_table(),
        build_benchmark_reading_table(),
        build_key_triple_difference_table(),
        build_group_change_key_summary(),
    ]

    figure_paths: list[Path] = []
    for family in ("vix", "macro", "earnings"):
        figure_paths.extend(plot_h3_triple_difference(family))
        figure_paths.extend(plot_group_change_components(family))

    return {"tables": table_paths, "figures": figure_paths}

"""Helpers for the p=4 H1/H2 robustness pass."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parent
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

_config = importlib.import_module("01_estimation_config")
_helpers = importlib.import_module("02_estimation_h1_h2")


ROBUSTNESS_P_LAGS = 4
ROBUSTNESS_OUTPUT_DIR = _config.ESTIMATION_DIR / "output" / "robustness_p4"

FAMILY_SPECS = _helpers.FAMILY_SPECS
FAMILY_NAMES = _config.FAMILY_NAMES
load_sp500_universe = _helpers.load_sp500_universe
run_family_h1_h2 = _helpers.run_family_h1_h2
build_run_summary = _helpers.build_run_summary


def ensure_output_dir() -> None:
    """Create the p=4 robustness output folder."""

    ROBUSTNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_family_outputs(family_result: dict[str, object]) -> None:
    """Save one family's p=4 outputs with transparent filenames."""

    family = str(family_result["family"])
    pre_result = family_result["pre_result"]
    post_result = family_result["post_result"]

    pre_result.coefficient_table.to_csv(
        ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_pre_coefficients.csv",
        index=False,
    )
    post_result.coefficient_table.to_csv(
        ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_post_coefficients.csv",
        index=False,
    )

    for shock_output in family_result["shock_outputs"]:
        shock_name = str(shock_output["shock_name"])

        shock_output["pre_bands"].to_csv(
            ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_pre_bands.csv",
            index=False,
        )
        shock_output["post_bands"].to_csv(
            ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_post_bands.csv",
            index=False,
        )
        shock_output["difference_bands"].to_csv(
            ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_post_minus_pre_bands.csv",
            index=False,
        )
        shock_output["pre_diagnostics"].to_csv(
            ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_pre_draw_diagnostics.csv",
            index=False,
        )
        shock_output["post_diagnostics"].to_csv(
            ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_post_draw_diagnostics.csv",
            index=False,
        )


def _plot_spec(family: str) -> dict[str, object]:
    """Keep the same key-horizon choices as the benchmark presentation layer."""

    if family == "vix":
        return {"shock_name": "dVIX_pos_inv", "horizons": list(range(0, 6))}
    if family == "macro":
        return {"shock_name": "macro_event_path", "horizons": list(range(-1, 5))}
    if family == "earnings":
        return {"shock_name": "earnings_event_path", "horizons": list(range(1, 14))}
    raise ValueError(f"Unknown family: {family}")


def _lit_level_from_dark(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive lit-share levels and changes from the saved dark-share outputs."""

    out = frame.copy()
    out["lit_share_level_point"] = 1.0 - out["dark_share_level_point"]
    out["lit_share_level_lower95"] = 1.0 - out["dark_share_level_upper95"]
    out["lit_share_level_upper95"] = 1.0 - out["dark_share_level_lower95"]

    out["lit_share_change_bps_point"] = -out["dark_share_change_bps_point"]
    out["lit_share_change_bps_lower95"] = -out["dark_share_change_bps_upper95"]
    out["lit_share_change_bps_upper95"] = -out["dark_share_change_bps_lower95"]
    return out


def _aggregate_earnings_to_blocks(frame: pd.DataFrame, *, block_minutes: int = 30) -> pd.DataFrame:
    """Average minute-level earnings outputs into 13 half-hour blocks."""

    work = frame.copy()
    work["horizon"] = work["horizon"].astype(int)
    work["block"] = (work["horizon"] // block_minutes) + 1

    numeric_cols = [
        col
        for col in work.columns
        if col not in {"regime", "family", "shock_name", "horizon", "block"}
    ]
    grouped = work.groupby("block", as_index=False)[numeric_cols].mean()
    grouped = grouped.rename(columns={"block": "horizon"})
    grouped.insert(0, "shock_name", str(work["shock_name"].iloc[0]))
    grouped.insert(0, "family", str(work["family"].iloc[0]))
    grouped.insert(0, "regime", str(work["regime"].iloc[0]))
    return grouped


def load_band_table(family: str, shock_name: str, regime: str) -> pd.DataFrame:
    """Load one saved p=4 band table."""

    path = ROBUSTNESS_OUTPUT_DIR / f"h1h2_p4_{family}_{shock_name}_{regime}_bands.csv"
    return pd.read_csv(path)


def build_key_irf_summary_table() -> Path:
    """Build the same compact key-horizon IRF table used in the benchmark."""

    rows: list[pd.DataFrame] = []
    for family in FAMILY_NAMES:
        spec = _plot_spec(family)
        for regime in ("pre", "post", "post_minus_pre"):
            frame = load_band_table(family, str(spec["shock_name"]), regime)
            if family == "earnings":
                frame = _aggregate_earnings_to_blocks(frame)
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
    path = ROBUSTNESS_OUTPUT_DIR / "h1h2_p4_key_irf_summary.csv"
    summary.to_csv(path, index=False)
    return path


def build_benchmark_comparison_table() -> Path:
    """Compare the benchmark p=2 key outputs with the p=4 robustness outputs."""

    benchmark = pd.read_csv(
        _config.ESTIMATION_DIR / "output" / "presentation" / "tables" / "h1h2_key_irf_summary.csv"
    )
    robustness = pd.read_csv(ROBUSTNESS_OUTPUT_DIR / "h1h2_p4_key_irf_summary.csv")

    join_cols = ["regime", "family", "shock_name", "horizon"]
    compare_cols = [
        "dark_share_change_bps_point",
        "dark_share_change_bps_lower95",
        "dark_share_change_bps_upper95",
        "lit_share_change_bps_point",
        "lit_share_change_bps_lower95",
        "lit_share_change_bps_upper95",
    ]

    merged = benchmark[join_cols + compare_cols].merge(
        robustness[join_cols + compare_cols],
        on=join_cols,
        suffixes=("_p2", "_p4"),
    )
    for col in compare_cols:
        merged[f"{col}_delta_p4_minus_p2"] = merged[f"{col}_p4"] - merged[f"{col}_p2"]

    path = ROBUSTNESS_OUTPUT_DIR / "h1h2_p4_vs_p2_comparison.csv"
    merged.to_csv(path, index=False)
    return path

"""Build the H1/H2 robustness point-estimate matrix LaTeX table.

For every benchmark rejection in H1 (regime ``pre``) and H2 (regime
``post_minus_pre``), this module assembles the dark-share-change point
estimate at the corresponding horizon under each of the four robustness
checks described in Section 4.5 (longer-lag $p{=}3$, longer-lag $p{=}4$,
linear time trend in the endogenous block, and within-pre placebo) and
writes a longtable snippet for the appendix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parent
PRESENTATION_TABLE_DIR = ESTIMATION_DIR / "output" / "presentation" / "tables"
OUTPUT_DIR = ESTIMATION_DIR / "output"
APPENDIX_DIR = ESTIMATION_DIR / "output" / "presentation" / "latex" / "appendix"
WRITING_RENDERED_DIR = (
    ESTIMATION_DIR.parent / "07_Thesis" / "latex" / "appendices" / "rendered"
)

FAMILY_SHOCK = {
    "vix": "dVIX_pos_inv",
    "macro": "macro_event_path",
    "earnings": "earnings_event_path",
}
FAMILY_LABEL = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}


def _format_bps(value: float) -> str:
    """Format a basis-point number with thousands separator and 1 decimal."""

    if pd.isna(value):
        return "--"
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))
    if magnitude >= 1000:
        whole = int(magnitude)
        thousand = whole // 1000
        rest = magnitude - thousand * 1000
        return f"{sign}{thousand}{{,}}{rest:05.1f}"
    return f"{sign}{magnitude:.1f}"


def _load_bench() -> pd.DataFrame:
    """Load the benchmark key IRF summary."""

    frame = pd.read_csv(PRESENTATION_TABLE_DIR / "h1h2_key_irf_summary.csv")
    return frame


def _load_robust_summary(tag: str) -> pd.DataFrame:
    """Load a p3 or p4 robustness key IRF summary."""

    return pd.read_csv(
        OUTPUT_DIR / f"robustness_{tag}" / f"h1h2_{tag}_key_irf_summary.csv"
    )


def _load_trend_bands(family: str, regime_path: str) -> pd.DataFrame:
    """Load the trend-robustness bands file for a (family, regime) cell."""

    shock_name = FAMILY_SHOCK[family]
    return pd.read_csv(
        OUTPUT_DIR
        / "robustness_trend"
        / f"h1h2_trend_{family}_{shock_name}_{regime_path}_bands.csv"
    )


def _load_placebo() -> pd.DataFrame:
    """Load the within-pre placebo (drift IRF) key summary."""

    return pd.read_csv(
        OUTPUT_DIR / "pre_stability" / "h1h2_pre_stability_key_summary.csv"
    )


def _file_horizon(family: str, horizon: int) -> int:
    """Map the benchmark horizon onto the minute-indexed file horizon column.

    Earnings uses block index (1..13) in the benchmark, but the placebo
    summary and the trend bands files store earnings horizons as minutes
    (30, 60, ..., 390). VIX and Macro use the same horizon convention
    everywhere.
    """

    if family == "earnings":
        return int(horizon) * 30
    return int(horizon)


def _rejection_rows(bench: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Return the benchmark rows that reject zero in the chapter-reported set.

    For VIX and Macro the test direction is the pecking-order sign (a
    negative dark-share-change in H1, a positive post-minus-pre difference
    in H2). For Earnings the chapter reports rejections in either
    direction because the empirical sign is opposite to the prediction;
    the filter therefore accepts any band that excludes zero.
    """

    excludes_zero = (bench["dark_share_change_bps_lower95"] > 0) | (
        bench["dark_share_change_bps_upper95"] < 0
    )
    if regime == "pre":
        directional = bench["dark_share_change_bps_upper95"] < 0
    else:
        directional = bench["dark_share_change_bps_lower95"] > 0
    family_mask = bench["family"].isin(["vix", "macro"])
    keep = excludes_zero & ((family_mask & directional) | (~family_mask))
    return bench[(bench["regime"] == regime) & keep].copy()


def _same_direction_rejects(row: pd.Series, direction: int) -> bool:
    """True iff the 95% band excludes zero on the benchmark side."""

    lo = float(row["dark_share_change_bps_lower95"])
    hi = float(row["dark_share_change_bps_upper95"])
    return (hi < 0) if direction < 0 else (lo > 0)


def _trend_cell(family: str, regime: str, horizon: int, direction: int) -> tuple[float, bool]:
    """Return (point, passes) for the trend check at one horizon."""

    regime_path = "pre" if regime == "pre" else "post_minus_pre"
    frame = _load_trend_bands(family, regime_path)
    h = _file_horizon(family, horizon)
    match = frame[frame["horizon"] == h]
    if match.empty:
        return float("nan"), False
    row = match.iloc[0]
    return float(row["dark_share_change_bps_point"]), _same_direction_rejects(row, direction)


def _robust_cell(
    frame: pd.DataFrame, regime: str, family: str, horizon: int, direction: int
) -> tuple[float, bool]:
    """Return (point, passes) for a p3/p4 check at one horizon."""

    match = frame[
        (frame["regime"] == regime)
        & (frame["family"] == family)
        & (frame["horizon"] == horizon)
    ]
    if match.empty:
        return float("nan"), False
    row = match.iloc[0]
    return float(row["dark_share_change_bps_point"]), _same_direction_rejects(row, direction)


def _placebo_cell(
    frame: pd.DataFrame, family: str, horizon: int, direction: int
) -> tuple[float, bool]:
    """Return (point, passes) for the within-pre placebo at one horizon.

    The placebo passes when the drift IRF band covers zero at this
    horizon. The criterion is direction-blind: any within-pre band that
    excludes zero, in either direction, is treated as a placebo failure
    because it means the pre-window already produces a comparable signal
    on its own.
    """

    h = _file_horizon(family, horizon)
    match = frame[
        (frame["family"] == family)
        & (frame["shock_name"] == FAMILY_SHOCK[family])
        & (frame["horizon"] == h)
    ]
    if match.empty:
        return float("nan"), False
    row = match.iloc[0]
    lo = float(row["dark_share_change_bps_lower95"])
    hi = float(row["dark_share_change_bps_upper95"])
    drift_excludes_zero = (lo > 0) or (hi < 0)
    return float(row["dark_share_change_bps_point"]), not drift_excludes_zero


def _assemble_rows() -> list[dict[str, object]]:
    """Walk every benchmark rejection and gather the four robustness cells."""

    bench = _load_bench()
    p3 = _load_robust_summary("p3")
    p4 = _load_robust_summary("p4")
    placebo = _load_placebo()

    rows: list[dict[str, object]] = []

    for hypothesis, regime in [("H1", "pre"), ("H2", "post_minus_pre")]:
        rejections = _rejection_rows(bench, regime)
        for family in ("vix", "macro", "earnings"):
            family_rows = rejections[rejections["family"] == family]
            family_rows = family_rows.sort_values("horizon")
            for _, row in family_rows.iterrows():
                horizon = int(row["horizon"])
                point = float(row["dark_share_change_bps_point"])
                direction = -1 if point < 0 else +1
                p3_point, p3_pass = _robust_cell(p3, regime, family, horizon, direction)
                p4_point, p4_pass = _robust_cell(p4, regime, family, horizon, direction)
                tr_point, tr_pass = _trend_cell(family, regime, horizon, direction)
                pb_point, pb_pass = _placebo_cell(placebo, family, horizon, direction)
                rows.append(
                    {
                        "hypothesis": hypothesis,
                        "family": FAMILY_LABEL[family],
                        "horizon": horizon,
                        "benchmark": point,
                        "p3": (p3_point, p3_pass),
                        "p4": (p4_point, p4_pass),
                        "trend": (tr_point, tr_pass),
                        "placebo": (pb_point, pb_pass),
                    }
                )

    return rows


def _format_cell(cell: tuple[float, bool]) -> str:
    """Render a (point, passes) tuple, marking failures with a star."""

    point, passes = cell
    body = _format_bps(point)
    if passes:
        return body
    return body + r"$^{*}$"


def _render_table(rows: list[dict[str, object]]) -> str:
    """Render the assembled rows as a longtable LaTeX snippet."""

    body_lines: list[str] = []
    for row in rows:
        body_lines.append(
            " & ".join(
                [
                    str(row["hypothesis"]),
                    str(row["family"]),
                    str(row["horizon"]),
                    _format_bps(row["benchmark"]),
                    _format_cell(row["p3"]),
                    _format_cell(row["p4"]),
                    _format_cell(row["trend"]),
                    _format_cell(row["placebo"]),
                ]
            )
            + r" \\"
        )

    header = (
        "Hypothesis & Family & Horizon & Benchmark & $p{=}3$ & $p{=}4$ & "
        "Trend & Placebo \\\\"
    )

    content = (
        "% Requires packages: booktabs, longtable\n"
        r"""\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{longtable}{@{\extracolsep{\fill}}llrrrrrr@{}}
\caption{Robustness point estimates at every H1 and H2 benchmark rejection.}
\label{tab:h1h2-robustness-matrix} \\
\toprule
"""
        + header
        + r"""
\midrule
\endfirsthead

\multicolumn{8}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
"""
        + header
        + r"""
\midrule
\endhead

\midrule
\multicolumn{8}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\addlinespace[2pt]
\multicolumn{8}{@{}p{\linewidth}@{}}{\footnotesize Note: Robustness point estimates (basis points) at every benchmark-rejection horizon for H1 and H2. The Benchmark column is the panel VARX at $p=2$; the four columns to its right refit the model under one robustness check each, defined in Section~\ref{sec:robustness-h1h2} and recapped at the start of this appendix. A star ($^{*}$) marks a failed check at that horizon: for $p{=}3$, $p{=}4$, and Trend, the refit 95\% band no longer excludes zero in the benchmark direction; for Placebo, the within-pre drift IRF band excludes zero in either direction.} \\
\endlastfoot
"""
        + "\n".join(body_lines)
        + r"""
\end{longtable}
"""
    )
    return content


def build_robustness_matrix_table() -> list[Path]:
    """Render the matrix and write it to the pipeline + writing folder."""

    rows = _assemble_rows()
    content = _render_table(rows)

    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    WRITING_RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = APPENDIX_DIR / "h1h2_robustness_matrix.tex"
    writing_path = WRITING_RENDERED_DIR / "h1h2_robustness_matrix.tex"
    pipeline_path.write_text(content)
    writing_path.write_text(content)
    return [pipeline_path, writing_path]


if __name__ == "__main__":
    paths = build_robustness_matrix_table()
    print("Wrote robustness matrix snippet to:")
    for path in paths:
        print(f"- {path}")

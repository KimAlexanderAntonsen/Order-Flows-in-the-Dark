"""Build the H3 robustness point-estimate matrix LaTeX table.

For every benchmark rejection of the H3 difference in difference, this module
assembles the point estimate at the same horizon under each of the four
robustness checks shared with the H1/H2 battery (longer-lag $p{=}3$,
longer-lag $p{=}4$, linear time trend in the endogenous block, and
within-pre placebo). The parallel-trends diagnostics that the
difference in difference design also requires are sample-level checks that do
not slot into a per-horizon matrix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ESTIMATION_DIR / "output"
WRITING_RENDERED_DIR = (
    ESTIMATION_DIR.parent / "07_Thesis" / "latex" / "appendices" / "rendered"
)
APPENDIX_DIR = ESTIMATION_DIR / "output" / "presentation" / "latex" / "appendix"

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


def _same_direction_rejects(row: pd.Series, direction: int) -> bool:
    """True iff the 95% band excludes zero on the benchmark side."""

    lo = float(row["dark_share_change_bps_lower95"])
    hi = float(row["dark_share_change_bps_upper95"])
    return (hi < 0) if direction < 0 else (lo > 0)


def _block_label(family: str, horizon: int) -> str:
    """Display horizon as a block index for Earnings, otherwise as-is."""

    if family == "earnings":
        return str(int(horizon) // 30)
    return str(int(horizon))


def _load_bench() -> pd.DataFrame:
    """Load the benchmark difference in difference summary."""

    return pd.read_csv(OUTPUT_DIR / "h3_estimation" / "h3_key_triple_difference_summary.csv")


def _load_robust_summary(tag: str) -> pd.DataFrame:
    """Load a $p=3$ or $p=4$ difference in difference summary."""

    return pd.read_csv(
        OUTPUT_DIR / f"robustness_{tag}" / f"h3_{tag}_key_triple_difference_summary.csv"
    )


def _load_trend_summary() -> pd.DataFrame:
    """Load the trend-controlled difference in difference summary."""

    return pd.read_csv(
        OUTPUT_DIR / "robustness_trend" / "h3_trend_key_triple_difference_summary.csv"
    )


def _load_placebo() -> pd.DataFrame:
    """Load the within-pre treated-minus-control placebo summary."""

    return pd.read_csv(
        OUTPUT_DIR / "pre_stability" / "h3_pre_stability_key_summary.csv"
    )


def _check_cell(
    frame: pd.DataFrame, family: str, horizon: int, direction: int
) -> tuple[float, bool]:
    """Read (point, passes) at one horizon from a difference in difference summary."""

    match = frame[(frame["family"] == family) & (frame["horizon"] == horizon)]
    if match.empty:
        return float("nan"), False
    row = match.iloc[0]
    return float(row["dark_share_change_bps_point"]), _same_direction_rejects(row, direction)


def _placebo_cell(
    frame: pd.DataFrame, family: str, horizon: int, direction: int
) -> tuple[float, bool]:
    """Read the placebo (within-pre difference in difference) cell at one horizon."""

    match = frame[
        (frame["family"] == family)
        & (frame["shock_name"] == FAMILY_SHOCK[family])
        & (frame["group"] == "treated_minus_matched_control")
        & (frame["horizon"] == horizon)
    ]
    if match.empty:
        return float("nan"), False
    row = match.iloc[0]
    lo = float(row["dark_share_change_bps_lower95"])
    hi = float(row["dark_share_change_bps_upper95"])
    drift_excludes_zero = (lo > 0) or (hi < 0)
    return float(row["dark_share_change_bps_point"]), not drift_excludes_zero


def _assemble_rows() -> list[dict[str, object]]:
    """Walk every H3 benchmark rejection and gather the four robustness cells."""

    bench = _load_bench()
    p3 = _load_robust_summary("p3")
    p4 = _load_robust_summary("p4")
    trend = _load_trend_summary()
    placebo = _load_placebo()

    excludes_zero = (bench["dark_share_change_bps_lower95"] > 0) | (
        bench["dark_share_change_bps_upper95"] < 0
    )
    rejections = bench[excludes_zero].sort_values(["family", "horizon"])

    rows: list[dict[str, object]] = []
    for _, row in rejections.iterrows():
        family = str(row["family"])
        horizon = int(row["horizon"])
        point = float(row["dark_share_change_bps_point"])
        direction = -1 if point < 0 else +1
        rows.append(
            {
                "family": FAMILY_LABEL[family],
                "horizon": _block_label(family, horizon),
                "benchmark": point,
                "p3": _check_cell(p3, family, horizon, direction),
                "p4": _check_cell(p4, family, horizon, direction),
                "trend": _check_cell(trend, family, horizon, direction),
                "placebo": _placebo_cell(placebo, family, horizon, direction),
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
        "Family & Horizon & Benchmark & $p{=}3$ & $p{=}4$ & Trend & Placebo \\\\"
    )

    content = (
        "% Requires packages: booktabs, longtable\n"
        r"""\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{longtable}{@{\extracolsep{\fill}}lrrrrrr@{}}
\caption{Robustness point estimates at every H3 benchmark rejection.}
\label{tab:h3-robustness-matrix} \\
\toprule
"""
        + header
        + r"""
\midrule
\endfirsthead

\multicolumn{7}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
"""
        + header
        + r"""
\midrule
\endhead

\midrule
\multicolumn{7}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\addlinespace[2pt]
\multicolumn{7}{@{}p{\linewidth}@{}}{\footnotesize Note: Robustness point estimates (basis points) at every benchmark H3 rejection. The Benchmark column is the H3 difference-in-difference statistic at the $p=2$ benchmark; column conventions and the star convention are identical to Table~\ref{tab:h1h2-robustness-matrix}. The Placebo column reports the within-pre treated-minus-matched-control drift IRF rather than the panel-wide drift.} \\
\endlastfoot
"""
        + "\n".join(body_lines)
        + r"""
\end{longtable}
"""
    )
    return content


def build_robustness_matrix_table() -> list[Path]:
    """Render the H3 matrix and write it to the pipeline + writing folder."""

    rows = _assemble_rows()
    content = _render_table(rows)

    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    WRITING_RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = APPENDIX_DIR / "h3_robustness_matrix.tex"
    writing_path = WRITING_RENDERED_DIR / "h3_robustness_matrix.tex"
    pipeline_path.write_text(content)
    writing_path.write_text(content)
    return [pipeline_path, writing_path]


if __name__ == "__main__":
    paths = build_robustness_matrix_table()
    print("Wrote H3 robustness matrix snippet to:")
    for path in paths:
        print(f"- {path}")

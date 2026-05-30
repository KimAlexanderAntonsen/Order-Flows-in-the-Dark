"""Create LaTeX tables for the H1/H2 layer.

This module turns the saved CSV outputs into LaTeX snippets with 
captions, labels, and simple formatting so they can be reused 
downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parent
PRESENTATION_TABLE_DIR = ESTIMATION_DIR / "output" / "presentation" / "tables"
H1H2_OUTPUT_DIR = ESTIMATION_DIR / "output" / "h1_h2"
LATEX_DIR = ESTIMATION_DIR / "output" / "presentation" / "latex"
MAIN_TEXT_DIR = LATEX_DIR / "main_text"
APPENDIX_DIR = LATEX_DIR / "appendix"


def ensure_output_dirs() -> None:
    """Create the LaTeX output folders if they are missing."""

    MAIN_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(filename: str) -> pd.DataFrame:
    """Read one saved presentation CSV table."""

    return pd.read_csv(PRESENTATION_TABLE_DIR / filename)


def _latex_escape(value: object) -> str:
    """Escape the most common LaTeX special characters."""

    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _format_int(value: object) -> str:
    """Format an integer-like value with thousands separators."""

    return f"{int(value):,}"


def _format_float(value: object, digits: int = 3) -> str:
    """Format a float-like value with a fixed number of decimals."""

    return f"{float(value):.{digits}f}"


def _write_table(path: Path, content: str) -> Path:
    """Write a LaTeX snippet to disk."""

    path.write_text(content)
    return path


def _header_comment(packages: str) -> str:
    """Return a short package note at the top of each snippet."""

    return "% Requires packages: " + packages + "\n"


def _compress_horizon_string(text: str) -> str:
    """Turn comma-separated horizons into short range labels."""

    values = [int(part.strip()) for part in str(text).split(",") if str(part).strip()]
    if not values:
        return ""
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(value) for value in values)


def build_sample_summary_table() -> Path:
    """Build the compact sample table recommended for the main text."""

    frame = _read_csv("h1h2_estimation_sample_summary.csv").copy()

    # The VIX family appears twice because the benchmark stores 
    # separate positive and negative innovation shocks. For the 
    # summary sample table, one row is enough because the sample 
    # itself is identical.
    vix_row = frame[frame["urgency_family"] == "vix"].iloc[0]
    if "shock_size" in vix_row.index and pd.notna(vix_row["shock_size"]):
        vix_shock_label = (
            r"1$\sigma$(dVIX$^{+}$) $\approx$ "
            + _format_float(vix_row["shock_size"], 3)
        )
    else:
        vix_shock_label = r"1$\sigma$(dVIX$^{+}$) innovation"
    family_rows = [
        {
            "family": "VIX",
            "shock": vix_shock_label,
            "row": vix_row,
        },
        {
            "family": "Macro",
            "shock": "Scheduled macro event path",
            "row": frame[frame["urgency_family"] == "macro"].iloc[0],
        },
        {
            "family": "Earnings",
            "shock": r"1\% earnings-surprise path",
            "row": frame[frame["urgency_family"] == "earnings"].iloc[0],
        },
    ]

    body_rows: list[str] = []
    for item in family_rows:
        row = item["row"]
        # ``shock`` may contain deliberate LaTeX (math mode, \sigma) 
        # and must therefore not be passed through ``_latex_escape``; 
        # the builder above controls its content directly.
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(item["family"]),
                    str(item["shock"]),
                    _format_int(row["benchmark_p_lags"]),
                    _format_int(row["n_draws"]),
                    _format_int(row["pre_observations"]),
                    _format_int(row["post_observations"]),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, graphicx, caption")
        + r"""\begin{table}[htbp]
\centering
\caption{Benchmark H1/H2 estimation sample.}
\label{tab:h1h2-estimation-sample}
\begin{tabular}{llrrrr}
\toprule
Family & Shock design & $p$ & Draws & Pre obs. & Post obs. \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Note: The benchmark sample uses the constant-membership S\&P 500 universe of 488 firms (502 broad 2019 constituents minus the 14 that left the index during the sample window via mergers, acquisitions, or index reconstitution; see \texttt{sp500\_constant\_sample\_exclusions.csv}). The pre-window panel uses all 488 firms; the post-window entity count, reported in the table body, is slightly smaller because of corporate actions in the post-window. The VIX impulse responses are reported at a one-standard-deviation shock to the positive-innovation series $\mathrm{dVIX}^{+}$ estimated on the full analysis sample, following Menkveld (2017); the macro and earnings families continue to use their calibrated event paths. The pre-window runs from 2019-06-10 to 2019-09-30 and the post-window runs from 2019-10-11 to 2020-02-19. The October 1 to October 10, 2019 exclusion window is excluded throughout.}
\end{table}
"""
    )
    return _write_table(MAIN_TEXT_DIR / "h1h2_estimation_sample_summary.tex", content)


def build_hypothesis_support_table() -> Path:
    """Build the compact H1/H2 support table for the main text."""

    frame = _read_csv("h1h2_hypothesis_support_summary.csv").copy()
    family_map = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        key_horizons = _compress_horizon_string(row["h1_key_horizons"])
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(family_map.get(row["family"], str(row["family"]))),
                    _latex_escape(key_horizons),
                    _latex_escape(row["h1_assessment"]),
                    _latex_escape(row["h2_assessment"]),
                    _latex_escape(row["p3_robustness_same_assessment"]),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, graphicx, caption")
        + r"""\begin{table}[htbp]
\centering
\caption{Benchmark support for H1 and H2 by urgency family.}
\label{tab:h1h2-hypothesis-support}
\begin{tabular}{lcccc}
\toprule
Shock family & Key window & H1 (pre-window) & H2 (post vs.\ pre) & Unchanged under $p=3$ \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Note: The table summarizes the benchmark 95\% simulation-band reading over family-specific key windows. ``Supported'' means that the response has the predicted sign and the 95\% band excludes zero in that direction over the key window. This is a transparent summary aid, not a formal joint test of the full impulse-response path.}
\end{table}
"""
    )
    return _write_table(MAIN_TEXT_DIR / "h1h2_hypothesis_support_summary.tex", content)


def build_key_irf_summary_table() -> Path:
    """Build a detailed appendix table from the key IRF summary CSV."""

    frame = _read_csv("h1h2_key_irf_summary.csv").copy()
    family_map = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}
    regime_map = {"pre": "Pre", "post": "Post", "post_minus_pre": "Post - Pre"}

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        dark_ci = (
            "["
            + _format_float(row["dark_share_change_bps_lower95"], 1)
            + ", "
            + _format_float(row["dark_share_change_bps_upper95"], 1)
            + "]"
        )
        lit_ci = (
            "["
            + _format_float(row["lit_share_change_bps_lower95"], 1)
            + ", "
            + _format_float(row["lit_share_change_bps_upper95"], 1)
            + "]"
        )
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(regime_map.get(row["regime"], str(row["regime"]))),
                    _latex_escape(family_map.get(row["family"], str(row["family"]))),
                    _format_int(row["horizon"]),
                    _format_float(row["dark_share_change_bps_point"], 1),
                    _latex_escape(dark_ci),
                    _format_float(row["lit_share_change_bps_point"], 1),
                    _latex_escape(lit_ci),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, longtable")
        + r"""\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\tabcolsep}{2pt}
\footnotesize
\begin{longtable}{llrllll}
\caption{Detailed key IRF summary for H1 and H2.}
\label{tab:h1h2-key-irf-summary} \\
\toprule
Regime & Family & Horizon & $\Delta$ Dark share (bps) & 95\% CI & $\Delta$ Lit share (bps) & 95\% CI \\
\midrule
\endfirsthead

\multicolumn{7}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
Regime & Family & Horizon & $\Delta$ Dark share (bps) & 95\% CI & $\Delta$ Lit share (bps) & 95\% CI \\
\midrule
\endhead

\midrule
\multicolumn{7}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\addlinespace[2pt]
\multicolumn{7}{@{}p{\linewidth}@{}}{\footnotesize Note: Tabular form of the impulse-response paths plotted in Chapter~\ref{ch:results}. Each row is one (regime, family, horizon) triple, with the dark-share IRF point estimate (basis points) and its 95\% simulation band, followed by the lit-share IRF and its band (the mirror image, since shares sum to one). ``Pre'' and ``Post'' rows are within-regime IRFs; ``Post - Pre'' rows are the post-minus-pre difference at the same horizon. Horizons are in minutes for VIX and Macro and in 30-minute blocks of the post-announcement trading day for Earnings.} \\
\endlastfoot
"""
        + "\n".join(body_rows)
        + r"""
\end{longtable}
"""
    )
    return _write_table(APPENDIX_DIR / "h1h2_key_irf_summary.tex", content)


def _combined_varx_frame(family: str) -> pd.DataFrame:
    """Combine the pre and post compact VARX tables side by side."""

    pre = _read_csv(f"h1h2_{family}_pre_varx_table.csv").copy()
    post = _read_csv(f"h1h2_{family}_post_varx_table.csv").copy()

    merged = pre.merge(post, on="regressor", suffixes=("_pre", "_post"))
    merged = merged.rename(
        columns={
            "log_dark_volume_t_pre": "pre_dark",
            "log_lit_volume_t_pre": "pre_lit",
            "log_total_realized_variance_t_pre": "pre_realvar",
            "log_dark_volume_t_post": "post_dark",
            "log_lit_volume_t_post": "post_lit",
            "log_total_realized_variance_t_post": "post_realvar",
        }
    )
    return merged


_FAMILY_VARX_EXOG_NOTE = {
    "vix": (
        r"The exogenous rows are the positive one-minute VIX innovation "
        r"$\Delta\mathrm{VIX}^{*+}$, its negative counterpart "
        r"$\Delta\mathrm{VIX}^{*-}$, and the VIX level (semi-elasticities)."
    ),
    "macro": (
        r"The exogenous rows are the six unit dummies on the macroeconomic "
        r"event path: $\mathrm{PreNews1min}$ at the pre-announcement minute and "
        r"$\mathrm{PostNews}k\mathrm{min}$ at the announcement minute plus $k$ "
        r"for $k=0,\dots,4$ (semi-elasticities)."
    ),
    "earnings": (
        r"The thirteen $\mathrm{PostEA}k$ rows are half-hour block dummies on "
        r"the post-announcement trading day ($k=1,\dots,13$), each scaled by "
        r"the firm's standardised earnings surprise as defined in "
        r"Section~\ref{sec:earnings-urgency} (semi-elasticities)."
    ),
}


def build_family_varx_table(family: str) -> Path:
    """Build one appendix VARX table with pre/post columns."""

    frame = _combined_varx_frame(family)
    family_title = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}[family]

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(row["regressor"]),
                    _latex_escape(row["pre_dark"]),
                    _latex_escape(row["pre_lit"]),
                    _latex_escape(row["pre_realvar"]),
                    _latex_escape(row["post_dark"]),
                    _latex_escape(row["post_lit"]),
                    _latex_escape(row["post_realvar"]),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, graphicx, caption")
        + r"""\begin{table}[H]
\centering
\caption{"""
        + family_title
        + r""" benchmark VARX coefficients: pre-window and post-window.}
\label{tab:h1h2-"""
        + family
        + r"""-varx}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Pre-window} & \multicolumn{3}{c}{Post-window} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Regressor & LogDark & LogLit & LogRealVar & LogDark & LogLit & LogRealVar \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
}
\caption*{\footnotesize Note: Coefficients of the panel VARX in equation~\eqref{eq:varx}, estimated separately on the pre and post windows. Rows are regressors; the three columns under each regime header are the coefficients in the equations for log dark volume, log lit volume, and log realised variance. The first six rows are the lagged endogenous block (elasticities). """
        + _FAMILY_VARX_EXOG_NOTE[family]
        + r""" Standard errors are two-way cluster-robust on stock and minute (see Section~\ref{sec:varx-model}); $^{*}$ marks $p<0.05$, $^{**}$ marks $p<0.01$.}
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / f"h1h2_{family}_varx_table.tex", content)


def build_lag_selection_table() -> Path:
    """Build the BIC lag-selection appendix table."""

    frame = pd.read_csv(H1H2_OUTPUT_DIR / "h1h2_lag_selection_bic.csv").copy()
    family_map = {"vix": "VIX", "macro": "Macro", "earnings": "Earnings"}
    regime_map = {"pre": "Pre", "post": "Post"}

    frame = frame.sort_values(["family", "regime", "p_lags"]).reset_index(drop=True)

    body_rows: list[str] = []
    current_family: str | None = None
    current_regime: str | None = None
    for _, row in frame.iterrows():
        family_name = family_map.get(row["family"], str(row["family"]))
        regime_name = regime_map.get(row["regime"], str(row["regime"]))
        bic_cell = _format_float(row["bic"], 1)
        aic_cell = _format_float(row["aic"], 1)
        if bool(row["bic_is_min"]):
            bic_cell = r"\textbf{" + bic_cell + "}"
        if bool(row["aic_is_min"]):
            aic_cell = r"\textbf{" + aic_cell + "}"
        family_cell = family_name if family_name != current_family else ""
        regime_cell = (
            regime_name
            if (regime_name != current_regime or family_name != current_family)
            else ""
        )
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(family_cell),
                    _latex_escape(regime_cell),
                    _format_int(row["p_lags"]),
                    _format_int(row["nobs"]),
                    _format_int(row["k_regressors"]),
                    _format_float(row["loglik"], 1),
                    aic_cell,
                    bic_cell,
                    _format_float(row["spectral_radius"], 3),
                ]
            )
            + r" \\"
        )
        current_family = family_name
        current_regime = regime_name

    content = (
        _header_comment("booktabs, graphicx, caption")
        + r"""\begin{table}[htbp]
\centering
\caption{Lag-order selection for the benchmark H1/H2 VARX.}
\label{tab:h1h2-lag-selection}
\small
\begin{tabular}{llrrrrrrr}
\toprule
Family & Regime & $p$ & $N$ & $k$ & Log-lik & AIC & BIC & $\rho(\Phi)$ \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Note: Information criteria for the panel VARX estimated at $p \in \{1,2,3,4\}$ separately for each urgency family and each regime. $N$ is the number of stacked panel observations after within-demeaning and lagging; $k$ is the number of regressors per equation; Log-lik is the panel log-likelihood; AIC and BIC are the standard information criteria; $\rho(\Phi)$ is the companion-form spectral radius, reported as a stationarity diagnostic ($\rho < 1$ for stationarity). The minimum AIC and minimum BIC within each family-regime block are shown in \textbf{bold}. The benchmark specification reports results at $p = 2$ following \citet{menkveld2017shades}, as discussed in Section~\ref{sec:varx-model}.}
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / "h1h2_lag_selection.tex", content)


def build_latex_tables() -> list[Path]:
    """Create the full set of LaTeX tables."""

    ensure_output_dirs()
    paths = [
        build_sample_summary_table(),
        build_hypothesis_support_table(),
        build_key_irf_summary_table(),
        build_family_varx_table("vix"),
        build_family_varx_table("macro"),
        build_family_varx_table("earnings"),
        build_lag_selection_table(),
    ]
    return paths

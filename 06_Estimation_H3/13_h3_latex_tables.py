"""Create LaTeX tables for the H3 layer.

This module mirrors the H1/H2 LaTeX export layer. It turns the saved 
CSV outputs into readable LaTeX snippets that can be included in the 
write-up or compiled directly inside the project.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

_config = importlib.import_module("01_h3_config")

VARX_DIR = _config.PROJECT_ROOT / "04_VARX"
if str(VARX_DIR) not in sys.path:
    sys.path.insert(0, str(VARX_DIR))

_varx_config = importlib.import_module("02_beta_varx_config")
_varx_utils = importlib.import_module("03_beta_varx_utils")
_varx_data = importlib.import_module("04_beta_varx_data")


PRETREND_TABLE_DIR = _config.TABLE_DIR
PRESENTATION_TABLE_DIR = _config.PRESENTATION_TABLE_DIR
ROBUSTNESS_P3_DIR = _config.ROBUSTNESS_P3_DIR
ROBUSTNESS_REFERENCE_DIR = _config.ROBUSTNESS_REFERENCE_DIR

LATEX_DIR = _config.PRESENTATION_DIR / "latex"
MAIN_TEXT_DIR = LATEX_DIR / "main_text"
APPENDIX_DIR = LATEX_DIR / "appendix"


FAMILY_LABELS = {
    "vix": "VIX",
    "macro": "Macro",
    "earnings": "Earnings",
}

OUTCOME_LABELS = {
    "dark_share": "Dark share",
    "lit_share": "Lit share",
    "log_total_volume": "Log total volume",
    "log_total_realized_variance": "Log realized variance",
}

GROUP_LABELS = {
    "treated": "Treated",
    "matched_control": "Matched control",
}

SHOCK_LABELS = {
    "dVIX_pos_inv": "Positive VIX innovation",
    "dVIX_neg_inv": "Negative VIX innovation",
    "macro_event_path": "Macro event path",
    "earnings_event_path": "Earnings event path",
}

PRIMARY_SHOCK = {
    "vix": "dVIX_pos_inv",
    "macro": "macro_event_path",
    "earnings": "earnings_event_path",
}

REGULAR_SESSION = _varx_config.REGULAR_SESSION
VIX_X_COLS = _varx_config.VIX_X_COLS
MACRO_X_COLS = _varx_config.MACRO_X_COLS
EARNINGS_X_COLS = _varx_config.EARNINGS_X_COLS

filter_session = _varx_utils.filter_session
load_minute_bar = _varx_data.load_minute_bar
load_vix_panel = _varx_data.load_vix_panel
load_macro_panel = _varx_data.load_macro_panel
load_earnings_panel = _varx_data.load_earnings_panel


TABLE_1_CSV_PATH = PRESENTATION_TABLE_DIR / "h3_table1_average_venue_shares.csv"
TABLE_2_CSV_PATH = PRESENTATION_TABLE_DIR / "h3_table2_variable_descriptions.csv"
TABLE_3_CSV_PATH = PRESENTATION_TABLE_DIR / "h3_table3_summary_statistics.csv"

TABLE_GROUP_COLUMNS = (
    ("benchmark_pooled", "Benchmark pooled"),
    ("treated", "Treated"),
    ("matched_control", "Matched control"),
    ("least_retail_reference", "Least-retail reference"),
)

GROUP_TICKER_PATHS = {
    "treated": _config.TREATED_PATH,
    "matched_control": _config.MATCHED_CONTROL_PATH,
    "least_retail_reference": _config.REFERENCE_PATH,
}

_GROUP_PANEL_CACHE: dict[str, pd.DataFrame] = {}
_GROUP_TICKER_CACHE: dict[str, list[str]] = {}


def ensure_output_dirs() -> None:
    """Create the LaTeX output folders if they are missing."""

    MAIN_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one saved CSV table."""

    return pd.read_csv(path)


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


def _format_bps(value: object) -> str:
    """Format a basis-point value with one decimal place."""

    return f"{float(value):.1f}"


def _bool_to_label(value: object) -> str:
    """Render booleans as readable yes/no labels."""

    return "Yes" if bool(value) else "No"


def _header_comment(packages: str) -> str:
    """Return a short package note at the top of each snippet."""

    return "% Requires packages: " + packages + "\n"


def _write_table(path: Path, content: str) -> Path:
    """Write a LaTeX snippet to disk."""

    path.write_text(content)
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write a CSV table to disk and return the path."""

    frame.to_csv(path, index=False)
    return path


def _compress_numbers(values: list[int]) -> str:
    """Compress a sorted list of integers into readable ranges."""

    if not values:
        return ""

    ranges: list[str] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = value
        prev = value
    ranges.append(f"{start}-{prev}" if start != prev else str(start))

    if len(ranges) == 1:
        return ranges[0]
    return ", ".join(ranges[:-1]) + " and " + ranges[-1]


def _humanize_reading(text: str) -> str:
    """Turn the saved dashboard text into cleaner display wording."""

    text = str(text).strip()
    if text == "No clear effect":
        return text

    match = re.match(r"^(Negative|Positive) at (.+)$", text)
    if not match:
        return text.replace("block_", "block ")

    direction = match.group(1)
    raw_tokens = [part.strip() for part in match.group(2).split(",") if part.strip()]
    block_mode = any(token.startswith("block_") for token in raw_tokens)
    numbers = []
    for token in raw_tokens:
        cleaned = token.replace("block_", "")
        if cleaned.isdigit():
            numbers.append(int(cleaned))

    compressed = _compress_numbers(sorted(numbers))
    if block_mode:
        return f"{direction} in blocks {compressed}"
    return f"{direction} at {compressed}"


def _horizon_label(row: pd.Series) -> str:
    """Create a readable horizon label for one row."""

    if "horizon_label" in row.index:
        raw_label = str(row["horizon_label"])
    else:
        horizon_value = int(row["horizon"])
        if row["family"] == "earnings":
            block = horizon_value if horizon_value <= 13 else max(1, horizon_value // 30)
            raw_label = f"block_{block}"
        else:
            raw_label = str(horizon_value)

    if row["family"] == "earnings":
        return raw_label.replace("block_", "Block ")
    return raw_label


def _build_ci(lower: object, upper: object, digits: int = 1) -> str:
    """Create a compact confidence-interval string."""

    return f"[{_format_float(lower, digits)}, {_format_float(upper, digits)}]"


def _load_group_tickers(group: str) -> list[str]:
    """Load the tickers that define one H3 group."""

    if group in _GROUP_TICKER_CACHE:
        return _GROUP_TICKER_CACHE[group]

    frame = pd.read_csv(GROUP_TICKER_PATHS[group])
    tickers = sorted(frame["ticker"].astype(str).str.upper().tolist())
    _GROUP_TICKER_CACHE[group] = tickers
    return tickers


def _build_group_minute_panel(group: str) -> pd.DataFrame:
    """Build the cleaned regular-hours minute panel for one H3 group."""

    if group in _GROUP_PANEL_CACHE:
        return _GROUP_PANEL_CACHE[group]

    frames: list[pd.DataFrame] = []
    for ticker in _load_group_tickers(group):
        frame = load_minute_bar(ticker)
        frame = filter_session(
            frame,
            timestamp_col="timestamp",
            session_start=REGULAR_SESSION.start,
            session_end=REGULAR_SESSION.end,
        )
        frame["total_volume"] = frame["dark_volume"] + frame["lit_volume"]
        frame = frame.loc[frame["total_volume"] > 0].copy()
        frame["dark_share_pct"] = 100.0 * frame["dark_volume"] / frame["total_volume"]
        frame["lit_share_pct"] = 100.0 * frame["lit_volume"] / frame["total_volume"]
        frame["dark_volume_k"] = frame["dark_volume"] / 1_000.0
        frame["lit_volume_k"] = frame["lit_volume"] / 1_000.0
        frame["total_volume_k"] = frame["total_volume"] / 1_000.0
        frame["total_realized_variance"] = (
            frame["dark_realized_variance"] + frame["lit_realized_variance"]
        )
        frames.append(
            frame[
                [
                    "asset",
                    "timestamp",
                    "dark_volume_k",
                    "lit_volume_k",
                    "total_volume_k",
                    "dark_share_pct",
                    "lit_share_pct",
                    "total_realized_variance",
                ]
            ]
        )

    panel = pd.concat(frames, ignore_index=True)
    _GROUP_PANEL_CACHE[group] = panel
    return panel


def _build_benchmark_pooled_panel() -> pd.DataFrame:
    """Return the benchmark H3 stock-minute panel pooled across treated and matched control."""

    return pd.concat(
        [
            _build_group_minute_panel("treated"),
            _build_group_minute_panel("matched_control"),
        ],
        ignore_index=True,
    )


def _mean_std_min_max(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    """Compute the usual summary statistics on one numeric vector."""

    array = pd.Series(values).dropna().astype(float)
    return float(array.mean()), float(array.std(ddof=1)), float(array.min()), float(array.max())


def _flatten_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Flatten a block of columns into one numeric vector."""

    return frame.loc[:, list(columns)].to_numpy(dtype=float).ravel()


def _round_numeric_columns(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round only the numeric columns of a DataFrame."""

    work = frame.copy()
    numeric_cols = work.select_dtypes(include=["number"]).columns
    work[numeric_cols] = work[numeric_cols].round(digits)
    return work


def _latex_number(value: object, digits: int = 3) -> str:
    """Format one numeric cell for LaTeX tables."""

    return f"{float(value):.{digits}f}"


def _summary_digits(label: str) -> int:
    """Choose a readable number of decimals for one summary-stat row."""

    if label.startswith("TotalRealizedVariance"):
        return 6
    if label.startswith("PreNews") or label.startswith("PostNews") or label.startswith("PostEA"):
        return 4
    return 3


def _summary_cell(label: str, value: object) -> str:
    """Format one summary-statistics cell with row-specific precision."""

    digits = _summary_digits(label)
    numeric_value = float(value)
    scientific_labels = (
        label.startswith("TotalRealizedVariance")
        or label.startswith("PreNews")
        or label.startswith("PostNews")
        or label.startswith("PostEA")
    )
    if scientific_labels and numeric_value != 0.0 and abs(numeric_value) < 10 ** (-digits):
        return f"{numeric_value:.2e}"
    return _latex_number(numeric_value, digits)


def build_table1_average_venue_shares() -> tuple[Path, Path]:
    """Build the adapted Menkveld-style H3 venue-share table."""

    benchmark_panel = _build_benchmark_pooled_panel()
    treated_panel = _build_group_minute_panel("treated")
    matched_panel = _build_group_minute_panel("matched_control")
    reference_panel = _build_group_minute_panel("least_retail_reference")

    panels = {
        "benchmark_pooled": benchmark_panel,
        "treated": treated_panel,
        "matched_control": matched_panel,
        "least_retail_reference": reference_panel,
    }

    row_specs = [
        ("DarkShare [percent]", "dark_volume_k"),
        ("LitShare [percent]", "lit_volume_k"),
    ]

    records: list[dict[str, object]] = []
    for label, volume_col in row_specs:
        record: dict[str, object] = {"variable": label}
        for group_key, panel in panels.items():
            numerator = float(panel[volume_col].sum())
            denominator = float(panel["total_volume_k"].sum())
            record[group_key] = 100.0 * numerator / denominator
        records.append(record)

    frame = _round_numeric_columns(pd.DataFrame(records), digits=2)
    csv_path = _write_csv(TABLE_1_CSV_PATH, frame)

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(str(row["variable"])),
                    _latex_number(row["benchmark_pooled"], 2),
                    _latex_number(row["treated"], 2),
                    _latex_number(row["matched_control"], 2),
                    _latex_number(row["least_retail_reference"], 2),
                ]
            )
            + r" \\"
        )

    tex_content = (
        _header_comment("booktabs, threeparttable, caption")
        + r"""\begin{table}[htbp]
\centering
\small
\begin{threeparttable}
\caption{Average venue shares in the H3 benchmark sample.}
\label{tab:h3-table1-venue-shares}
\begin{tabular}{lrrrr}
\toprule
Venue share & Benchmark pooled & Treated & Matched control & Least-retail reference \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Note: The benchmark pooled column combines the treated and matched-control stocks. Shares are volume-weighted over the cleaned regular-hours minute-bar sample. The least-retail reference group is shown for orientation, but it is not the benchmark H3 control.}
\end{threeparttable}
\end{table}
"""
    )
    tex_path = _write_table(MAIN_TEXT_DIR / "h3_table1_average_venue_shares.tex", tex_content)
    return csv_path, tex_path


def build_table2_variable_descriptions() -> tuple[Path, Path]:
    """Build the adapted Menkveld-style variable-description table."""

    rows = [
        {
            "panel": "Panel A: Endogenous variables",
            "type": "Y",
            "variable": r"$\log(\mathrm{DarkVol}_{i,t})$",
            "definition": "Log dark volume in stock $i$ and minute $t$.",
        },
        {
            "panel": "Panel A: Endogenous variables",
            "type": "Y",
            "variable": r"$\log(\mathrm{LitVol}_{i,t})$",
            "definition": "Log lit volume in stock $i$ and minute $t$.",
        },
        {
            "panel": "Panel A: Endogenous variables",
            "type": "Y",
            "variable": r"$\log(\mathrm{TRV}_{i,t})$",
            "definition": "Log total realized variance, computed as dark plus lit realized variance in stock $i$ and minute $t$.",
        },
        {
            "panel": "Panel B: Common exogenous variables",
            "type": "Z",
            "variable": r"$dVIX_t^{+}$",
            "definition": "Positive VIX innovation, constructed as the positive part of the AR(1) residual in minute-by-minute VIX changes.",
        },
        {
            "panel": "Panel B: Common exogenous variables",
            "type": "Z",
            "variable": r"$dVIX_t^{-}$",
            "definition": "Negative VIX innovation, constructed as the absolute value of the negative AR(1) residual in minute-by-minute VIX changes.",
        },
        {
            "panel": "Panel B: Common exogenous variables",
            "type": "Z",
            "variable": r"$VIX_t$",
            "definition": "VIX level included as a common control variable.",
        },
        {
            "panel": "Panel B: Common exogenous variables",
            "type": "Z",
            "variable": r"$PreNews1min_t,\; PostNews0min_t,\ldots,PostNews4min_t$",
            "definition": "Minute-level macro-announcement path indicators spanning one minute before through four minutes after the scheduled release.",
        },
        {
            "panel": "Panel C: Firm-specific exogenous variables",
            "type": "Z",
            "variable": r"$PostEA1_{i,t},\ldots,PostEA13_{i,t}$",
            "definition": "Firm-specific earnings-surprise path, scaled by the absolute earnings surprise and mapped into the 13 half-hour trading blocks after the announcement.",
        },
        {
            "panel": "Panel D: Derived H3 objects",
            "type": "Derived",
            "variable": r"$DarkShare_{i,t}$",
            "definition": r"Dark-share outcome derived from the volume system, $100\times DarkVol_{i,t}/(DarkVol_{i,t}+LitVol_{i,t})$.",
            "definition_latex": r"Dark-share outcome derived from the volume system, $100\times DarkVol_{i,t}/(DarkVol_{i,t}+LitVol_{i,t})$.",
        },
        {
            "panel": "Panel D: Derived H3 objects",
            "type": "Derived",
            "variable": r"$LitShare_{i,t}$",
            "definition": r"Lit-share outcome derived as the complement to dark share, $100\times LitVol_{i,t}/(DarkVol_{i,t}+LitVol_{i,t})$.",
            "definition_latex": r"Lit-share outcome derived as the complement to dark share, $100\times LitVol_{i,t}/(DarkVol_{i,t}+LitVol_{i,t})$.",
        },
        {
            "panel": "Panel D: Derived H3 objects",
            "type": "Groups",
            "variable": r"$Treated_i,\; Control_i,\; Reference_i$",
            "definition": "Retail-treated, matched-control, and least-retail reference group indicators used in the H3 treatment-control extension.",
        },
    ]
    frame = pd.DataFrame(rows)
    csv_path = _write_csv(TABLE_2_CSV_PATH, frame)

    body_rows: list[str] = []
    current_panel = None
    for _, row in frame.iterrows():
        if row["panel"] != current_panel:
            current_panel = row["panel"]
            body_rows.append(
                rf"    \multicolumn{{3}}{{@{{}}l}}{{\textit{{{_latex_escape(current_panel)}}}}} \\"
            )
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(str(row["type"])),
                    str(row["variable"]),
                    (
                        str(row["definition_latex"])
                        if "definition_latex" in row.index and pd.notna(row["definition_latex"])
                        else _latex_escape(str(row["definition"]))
                    ),
                ]
            )
            + r" \\"
        )

    tex_content = (
        _header_comment("booktabs, tabularx, array, threeparttable, caption")
        + r"""\begin{table}[htbp]
\centering
\scriptsize
\begin{threeparttable}
\caption{Variable descriptions for the H3 treatment-control VARX design.}
\label{tab:h3-table2-variable-descriptions}
\renewcommand{\arraystretch}{1.06}
\begin{tabularx}{\textwidth}{@{}ll>{\RaggedRight\arraybackslash}X@{}}
\toprule
Type & Variable & Description \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabularx}
\caption*{\footnotesize Note: Subscript $i$ indexes stocks and subscript $t$ indexes minute intervals. The benchmark H3 design reuses the finished VARX specification from Folder 04 and then compares the post-minus-pre impulse responses of the treated and matched-control groups.}
\end{threeparttable}
\end{table}
"""
    )
    tex_path = _write_table(MAIN_TEXT_DIR / "h3_table2_variable_descriptions.tex", tex_content)
    return csv_path, tex_path


def build_table3_summary_statistics() -> tuple[Path, Path]:
    """Build the adapted Menkveld-style H3 summary-statistics table."""

    benchmark_panel = _build_benchmark_pooled_panel()
    treated_panel = _build_group_minute_panel("treated")
    matched_panel = _build_group_minute_panel("matched_control")
    reference_panel = _build_group_minute_panel("least_retail_reference")
    vix_panel = load_vix_panel()
    macro_panel = load_macro_panel()
    earnings_panel = load_earnings_panel()

    benchmark_tickers = set(_load_group_tickers("treated")) | set(_load_group_tickers("matched_control"))
    treated_tickers = set(_load_group_tickers("treated"))
    matched_tickers = set(_load_group_tickers("matched_control"))
    reference_tickers = set(_load_group_tickers("least_retail_reference"))

    def _earnings_vector(ticker_set: set[str]) -> np.ndarray:
        subset = earnings_panel.loc[earnings_panel["asset"].isin(ticker_set)]
        return _flatten_columns(subset, EARNINGS_X_COLS)

    records = []

    stock_rows = [
        ("DarkVolume [1k shares]", benchmark_panel["dark_volume_k"], treated_panel["dark_volume_k"], matched_panel["dark_volume_k"], reference_panel["dark_volume_k"]),
        ("LitVolume [1k shares]", benchmark_panel["lit_volume_k"], treated_panel["lit_volume_k"], matched_panel["lit_volume_k"], reference_panel["lit_volume_k"]),
        ("DarkShare [percent]", benchmark_panel["dark_share_pct"], treated_panel["dark_share_pct"], matched_panel["dark_share_pct"], reference_panel["dark_share_pct"]),
        ("LitShare [percent]", benchmark_panel["lit_share_pct"], treated_panel["lit_share_pct"], matched_panel["lit_share_pct"], reference_panel["lit_share_pct"]),
        ("TotalVolume [1k shares]", benchmark_panel["total_volume_k"], treated_panel["total_volume_k"], matched_panel["total_volume_k"], reference_panel["total_volume_k"]),
        ("TotalRealizedVariance [squared returns]", benchmark_panel["total_realized_variance"], treated_panel["total_realized_variance"], matched_panel["total_realized_variance"], reference_panel["total_realized_variance"]),
    ]

    for label, full_values, treated_values, matched_values, reference_values in stock_rows:
        mean_full, std_full, min_full, max_full = _mean_std_min_max(full_values)
        records.append(
            {
                "variable": label,
                "mean_full": mean_full,
                "std_full": std_full,
                "min_full": min_full,
                "max_full": max_full,
                "mean_treated": float(pd.Series(treated_values).mean()),
                "mean_matched": float(pd.Series(matched_values).mean()),
                "mean_reference": float(pd.Series(reference_values).mean()),
            }
        )

    mean_full, std_full, min_full, max_full = _mean_std_min_max(vix_panel["dVIX_pos_inv"])
    records.append(
        {
            "variable": "dVIX+ [index points]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": mean_full,
            "mean_matched": mean_full,
            "mean_reference": mean_full,
        }
    )
    mean_full, std_full, min_full, max_full = _mean_std_min_max(vix_panel["dVIX_neg_inv"])
    records.append(
        {
            "variable": "dVIX- [index points]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": mean_full,
            "mean_matched": mean_full,
            "mean_reference": mean_full,
        }
    )
    mean_full, std_full, min_full, max_full = _mean_std_min_max(vix_panel["VIX_close"])
    records.append(
        {
            "variable": "VIX [index level]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": mean_full,
            "mean_matched": mean_full,
            "mean_reference": mean_full,
        }
    )

    pre_vec = macro_panel["pre_news_1min"]
    mean_full, std_full, min_full, max_full = _mean_std_min_max(pre_vec)
    records.append(
        {
            "variable": "PreNews1min [1/0]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": mean_full,
            "mean_matched": mean_full,
            "mean_reference": mean_full,
        }
    )
    post_vec = _flatten_columns(macro_panel, MACRO_X_COLS[1:])
    mean_full, std_full, min_full, max_full = _mean_std_min_max(post_vec)
    records.append(
        {
            "variable": "PostNews0min,...,PostNews4min [1/0]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": mean_full,
            "mean_matched": mean_full,
            "mean_reference": mean_full,
        }
    )

    earnings_full = _earnings_vector(benchmark_tickers)
    earnings_treated = _earnings_vector(treated_tickers)
    earnings_matched = _earnings_vector(matched_tickers)
    earnings_reference = _earnings_vector(reference_tickers)
    mean_full, std_full, min_full, max_full = _mean_std_min_max(earnings_full)
    records.append(
        {
            "variable": "PostEA1,...,PostEA13 [scaled surprise]",
            "mean_full": mean_full,
            "std_full": std_full,
            "min_full": min_full,
            "max_full": max_full,
            "mean_treated": float(pd.Series(earnings_treated).mean()),
            "mean_matched": float(pd.Series(earnings_matched).mean()),
            "mean_reference": float(pd.Series(earnings_reference).mean()),
        }
    )

    frame = pd.DataFrame(records)
    csv_path = _write_csv(TABLE_3_CSV_PATH, _round_numeric_columns(frame, digits=3))

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(str(row["variable"])),
                    _summary_cell(str(row["variable"]), row["mean_full"]),
                    _summary_cell(str(row["variable"]), row["std_full"]),
                    _summary_cell(str(row["variable"]), row["min_full"]),
                    _summary_cell(str(row["variable"]), row["max_full"]),
                    _summary_cell(str(row["variable"]), row["mean_treated"]),
                    _summary_cell(str(row["variable"]), row["mean_matched"]),
                    _summary_cell(str(row["variable"]), row["mean_reference"]),
                ]
            )
            + r" \\"
        )

    tex_content = (
        _header_comment("booktabs, array, threeparttable, caption")
        + r"""\begin{table}[htbp]
\centering
\scriptsize
\begin{threeparttable}
\caption{Summary statistics for the benchmark H3 sample.}
\label{tab:h3-table3-summary-statistics}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.04}
\begin{tabular*}{0.99\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.31\textwidth}rrrrrrr@{}}
\toprule
Variable & \shortstack{Mean\\(full)} & \shortstack{StDev\\(full)} & \shortstack{Min\\(full)} & \shortstack{Max\\(full)} & \shortstack{Treated\\mean} & \shortstack{Matched\\mean} & \shortstack{Reference\\mean} \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular*}
\caption*{\footnotesize Note: Full-sample statistics pool the treated and matched-control benchmark sample. Stock-specific rows are computed from the cleaned regular-hours minute-bar sample. The VIX and macro rows use the cleaned common exogenous panels, while the earnings row pools the 13 firm-specific post-announcement path variables across the cleaned sparse earnings panel.}
\end{threeparttable}
\end{table}
"""
    )
    tex_path = _write_table(MAIN_TEXT_DIR / "h3_table3_summary_statistics.tex", tex_content)
    return csv_path, tex_path


def build_estimation_sample_table() -> Path:
    """Build the compact benchmark H3 sample table for the main text."""

    frame = _read_csv(PRESENTATION_TABLE_DIR / "h3_estimation_sample_summary.csv")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        short_shock = {
            "vix": "+/- dVIX",
            "macro": "Macro path",
            "earnings": "1\\% EPS path",
        }[str(row["family"]).lower()]
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(
                        FAMILY_LABELS.get(str(row["family"]).lower(), str(row["family"]))
                    ),
                    short_shock,
                    _format_int(row["benchmark_p_lags"]),
                    _format_int(row["n_draws"]),
                    _format_int(row["treated_pre_observations"]),
                    _format_int(row["treated_post_observations"]),
                    _format_int(row["control_pre_observations"]),
                    _format_int(row["control_post_observations"]),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, adjustbox, caption")
        + r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Benchmark H3 estimation sample.}
\label{tab:h3-estimation-sample}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\toprule
Family & Shock & $p$ & Draws & T pre & T post & C pre & C post \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\par\medskip
\footnotesize Note: The benchmark H3 design uses the retail-treated group and the matched control group, each with 70 stocks at construction (top and matched lower quintiles of the 375-firm retail-scored subset; the constant-membership universe is 487 firms, of which 112 have no Robintrack pre-window coverage). The benchmark lag choice is $p=2$ and the simulation layer uses 10,000 draws. The pre-window runs from 2019-06-10 to 2019-09-30 and the post-window runs from 2019-10-11 to 2020-02-19. The October 1 to October 10, 2019 exclusion window is excluded throughout.
\end{table}
"""
    )
    return _write_table(MAIN_TEXT_DIR / "h3_estimation_sample_summary.tex", content)


def build_robustness_dashboard_table() -> Path:
    """Build the main-text H3 summary table with the key robustness checks."""

    benchmark = _read_csv(PRESENTATION_TABLE_DIR / "h3_benchmark_reading_summary.csv")
    dashboard = _read_csv(PRESENTATION_TABLE_DIR / "h3_robustness_dashboard.csv")
    frame = benchmark.merge(dashboard, on="family", how="inner")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(FAMILY_LABELS[str(row["family"]).lower()]),
                    _latex_escape(str(row["key_window"])),
                    _latex_escape(_humanize_reading(str(row["benchmark_reading"]))),
                    _latex_escape(_humanize_reading(str(row["p3_reading"]))),
                    _latex_escape(_humanize_reading(str(row["reference_control_reading"]))),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, tabularx, array, threeparttable, caption, ragged2e")
        + r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.04}
\begin{threeparttable}
\caption{Benchmark H3 reading and targeted robustness checks.}
\label{tab:h3-robustness-dashboard}
\begin{tabularx}{\textwidth}{@{}ll>{\RaggedRight\arraybackslash}X>{\RaggedRight\arraybackslash}X>{\RaggedRight\arraybackslash}X@{}}
\toprule
Shock family & Key window & Benchmark reading & $p=3$ robustness & Least-retail control \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabularx}
\caption*{\footnotesize Note: The benchmark H3 design uses the matched control group, $p=2$, and 10,000 simulation draws. The $p=3$, $p=4$, and least-retail-control robustness checks use 5,000 draws for practicality. The benchmark matched-control design remains the preferred specification because the least-retail group is materially farther away from the treated stocks in the pre-window.}
\end{threeparttable}
\end{table}
"""
    )
    return _write_table(MAIN_TEXT_DIR / "h3_robustness_dashboard.tex", content)


def build_pretrend_sample_table() -> Path:
    """Build a compact appendix table describing the Step 1 pretrend sample."""

    frame = _read_csv(PRETREND_TABLE_DIR / "h3_pretrend_sample_summary.csv")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(str(row["group_label"])),
                    _format_int(row["n_assets"]),
                    _format_int(row["n_stock_days"]),
                    _format_float(row["mean_stock_days"], 1),
                    _latex_escape(str(row["first_date"])),
                    _latex_escape(str(row["last_date"])),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, threeparttable, caption")
        + r"""\begin{table}[htbp]
\centering
\small
\begin{threeparttable}
\caption{Pre-window sample used for H3 comparability diagnostics.}
\label{tab:h3-pretrend-sample}
\begin{tabular}{lrrrll}
\toprule
Group & Assets & Stock-days & Mean stock-days & First date & Last date \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Note: The pretrend sample covers the pre-window only. It is used to assess group comparability, trend alignment, and placebo breaks before the benchmark H3 estimation is interpreted.}
\end{threeparttable}
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_pretrend_sample_summary.tex", content)


def build_pretrend_comparability_table() -> Path:
    """Build a compact appendix table for the main pretrend balance metrics."""

    frame = _read_csv(PRETREND_TABLE_DIR / "h3_pretrend_comparability_summary.csv")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "        "
            + " & ".join(
                [
                    _latex_escape(OUTCOME_LABELS[str(row["outcome"])]),
                    _format_float(row["treated_mean"], 3),
                    _format_float(row["matched_control_mean"], 3),
                    _format_float(row["least_retail_reference_mean"], 3),
                    _format_float(row["treated_vs_matched_smd"], 3),
                    _format_float(row["treated_vs_reference_smd"], 3),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, adjustbox, caption")
        + r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Pre-window comparability of the H3 groups.}
\label{tab:h3-pretrend-comparability}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Outcome & Treated & Matched & Least-retail & SMD M & SMD L \\
\midrule
"""
        + "\n".join(body_rows)
        + r"""
\bottomrule
\end{tabular}
\par\medskip
\footnotesize Note: This table summarizes the pre-window group differences on the main H3 outcome-side variables. The matched control is uniformly closer to the treated group than the least-retail reference group, which is why the matched control remains the benchmark H3 control.
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_pretrend_comparability_summary.tex", content)


def build_key_triple_difference_table() -> Path:
    """Build the benchmark H3 key-summary appendix table."""

    frame = _read_csv(PRESENTATION_TABLE_DIR / "h3_key_triple_difference_summary.csv")

    shock_family_labels = {
        ("vix", "dVIX_pos_inv"): "VIX (+)",
        ("vix", "dVIX_neg_inv"): "VIX (-)",
        ("macro", "macro_event_path"): "Macro",
        ("earnings", "earnings_event_path"): "Earnings",
    }

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(
                        shock_family_labels[(str(row["family"]), str(row["shock_name"]))]
                    ),
                    _latex_escape(_horizon_label(row)),
                    _format_bps(row["dark_share_change_bps_point"]),
                    _latex_escape(
                        _build_ci(
                            row["dark_share_change_bps_lower95"],
                            row["dark_share_change_bps_upper95"],
                            1,
                        )
                    ),
                    _latex_escape(_bool_to_label(row["exclude_zero"])),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, longtable")
        + r"""\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.04}
\small
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedleft\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}@{}}
\caption{Benchmark H3 difference in difference summary.}
\label{tab:h3-key-triple-difference-summary} \\
\toprule
Shock family & Horizon & H3 dark share (bps) & 95\% CI & 95\% band excl.\ 0 \\
\midrule
\endfirsthead

\multicolumn{5}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
Shock family & Horizon & $\Delta$ dark share (bps) & 95\% CI & 95\% band excl.\ 0 \\
\midrule
\endhead

\midrule
\multicolumn{5}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\addlinespace[2pt]
\multicolumn{5}{@{}p{\linewidth}@{}}{\footnotesize Note: Tabular form of the H3 difference-in-difference impulse responses reported in Chapter~\ref{ch:h3}. The H3 statistic at horizon $h$ is the post-minus-pre change in the urgency-to-dark-share IRF for the retail-treated group minus the same post-minus-pre change for the matched-control group (basis points; see formal definition in Appendix~\ref{app:hypotheses}). The 95\% CI is the simulation band on that statistic; the final column reports whether the band excludes zero in either direction at that horizon. VIX appears twice for the separate positive ($+\sigma$) and negative ($-\sigma$) one-standard-deviation innovations. Horizons are in minutes for VIX and Macro and in 30-minute blocks of the post-announcement trading day for Earnings.} \\
\endlastfoot
"""
        + "\n".join(body_rows)
        + "\n"
        + r"""\end{longtable}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_key_triple_difference_summary.tex", content)


def build_group_change_table() -> Path:
    """Build the treated/control component appendix table."""

    frame = _read_csv(PRESENTATION_TABLE_DIR / "h3_group_change_key_summary.csv")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(FAMILY_LABELS[str(row["family"])]),
                    _latex_escape(GROUP_LABELS[str(row["group"])]),
                    _latex_escape(_horizon_label(row)),
                    _format_bps(row["dark_share_change_bps_point"]),
                    _latex_escape(
                        _build_ci(
                            row["dark_share_change_bps_lower95"],
                            row["dark_share_change_bps_upper95"],
                            1,
                        )
                    ),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs, longtable")
        + r"""\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.04}
\small
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.11\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedleft\arraybackslash}p{0.20\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}@{}}
\caption{Group-specific post-minus-pre changes behind the H3 object.}
\label{tab:h3-group-change-summary} \\
\toprule
Family & Group & Horizon & Post-pre dark share (bps) & 95\% CI \\
\midrule
\endfirsthead

\multicolumn{5}{l}{\textit{Table \thetable\ continued from previous page}} \\
\toprule
Family & Group & Horizon & Post-minus-pre dark share (bps) & 95\% CI \\
\midrule
\endhead

\midrule
\multicolumn{5}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
"""
        + "\n".join(body_rows)
        + "\n"
        + r"""\end{longtable}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_group_change_key_summary.tex", content)


def _filter_headline_robustness_rows(frame: pd.DataFrame, robustness_col: str) -> pd.DataFrame:
    """Keep the headline-shock rows that matter most for the appendix tables."""

    work = frame.copy()
    work = work[work.apply(lambda row: row["shock_name"] == PRIMARY_SHOCK[str(row["family"])], axis=1)]
    work = work[work["exclude_zero_benchmark"] | work[robustness_col]].copy()
    return work


def build_p3_comparison_table() -> Path:
    """Build the appendix table comparing benchmark H3 to the p=3 pass."""

    frame = _read_csv(ROBUSTNESS_P3_DIR / "h3_p3_vs_p2_comparison.csv")
    frame = _filter_headline_robustness_rows(frame, "exclude_zero_p3")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(FAMILY_LABELS[str(row["family"])]),
                    _latex_escape(_horizon_label(row)),
                    _format_bps(row["dark_share_change_bps_point_benchmark"]),
                    _format_bps(row["dark_share_change_bps_point_p3"]),
                    _latex_escape(_bool_to_label(row["exclude_zero_benchmark"])),
                    _latex_escape(_bool_to_label(row["exclude_zero_p3"])),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs")
        + r"""\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.05}
\normalsize
\caption{Headline H3 comparison: benchmark $p=2$ versus robustness $p=3$.}
\label{tab:h3-p3-vs-p2}
\begin{tabular*}{0.96\textwidth}{@{\extracolsep{\fill}}llrrll@{}}
\toprule
Family & Horizon & H3 $p=2$ & H3 $p=3$ & $p=2$ excl.\ 0 & $p=3$ excl.\ 0 \\
\midrule
"""
        + "\n".join(body_rows)
        + "\n"
        + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_p3_vs_p2_comparison.tex", content)


def build_reference_comparison_table() -> Path:
    """Build the appendix table comparing matched and least-retail controls."""

    frame = _read_csv(ROBUSTNESS_REFERENCE_DIR / "h3_reference_vs_matched_comparison.csv")
    frame = _filter_headline_robustness_rows(frame, "exclude_zero_reference")

    body_rows: list[str] = []
    for _, row in frame.iterrows():
        body_rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(FAMILY_LABELS[str(row["family"])]),
                    _latex_escape(_horizon_label(row)),
                    _format_bps(row["dark_share_change_bps_point_benchmark"]),
                    _format_bps(row["dark_share_change_bps_point_reference"]),
                    _latex_escape(_bool_to_label(row["exclude_zero_benchmark"])),
                    _latex_escape(_bool_to_label(row["exclude_zero_reference"])),
                ]
            )
            + r" \\"
        )

    content = (
        _header_comment("booktabs")
        + r"""\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.05}
\normalsize
\caption{Headline H3 comparison: matched control versus least-retail reference control.}
\label{tab:h3-reference-vs-matched}
\begin{tabular*}{0.98\textwidth}{@{\extracolsep{\fill}}llrrll@{}}
\toprule
Family & Horizon & Benchmark H3 & Ref. control H3 & Benchmark excl.\ 0 & Ref. excl.\ 0 \\
\midrule
"""
        + "\n".join(body_rows)
        + "\n"
        + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    )
    return _write_table(APPENDIX_DIR / "h3_reference_vs_matched_comparison.tex", content)


def build_all_tables() -> list[Path]:
    """Generate all H3 LaTeX tables."""

    ensure_output_dirs()
    paths = [
        *build_table1_average_venue_shares(),
        *build_table2_variable_descriptions(),
        *build_table3_summary_statistics(),
        build_estimation_sample_table(),
        build_robustness_dashboard_table(),
        build_pretrend_sample_table(),
        build_pretrend_comparability_table(),
        build_key_triple_difference_table(),
        build_group_change_table(),
        build_p3_comparison_table(),
        build_reference_comparison_table(),
    ]
    return paths


def main() -> None:
    """Generate the current H3 LaTeX tables."""

    paths = build_all_tables()
    print("Saved H3 LaTeX tables:")
    for path in paths:
        print(f"- {path}")


"""Configuration for the H3 estimation layer.

The H3 folder now has two implemented stages:

1. Group diagnostics in the pre period.
2. Benchmark H3 estimation using the treated and matched-control 
   groups.

This file keeps the sample choices and benchmark settings in one place 
so the diagnostic and estimation layers stay aligned.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H3_DIR = PROJECT_ROOT / "06_Estimation_H3"
RETAIL_DIR = PROJECT_ROOT / "02_RetailClassification"
GROUP_OUTPUT_DIR = RETAIL_DIR / "group_outputs"

OUTPUT_DIR = H3_DIR / "output" / "pretrend"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

ESTIMATION_OUTPUT_DIR = H3_DIR / "output" / "h3_estimation"
PRESENTATION_DIR = H3_DIR / "output" / "presentation"
PRESENTATION_TABLE_DIR = PRESENTATION_DIR / "tables"
PRESENTATION_FIGURE_DIR = PRESENTATION_DIR / "figures"
ROBUSTNESS_P3_DIR = H3_DIR / "output" / "robustness_p3"
ROBUSTNESS_P4_DIR = H3_DIR / "output" / "robustness_p4"
ROBUSTNESS_REFERENCE_DIR = H3_DIR / "output" / "robustness_reference"


# Group definitions
TREATED_PATH = GROUP_OUTPUT_DIR / "retail_treated_group.csv"
MATCHED_CONTROL_PATH = GROUP_OUTPUT_DIR / "retail_matched_control_group.csv"
REFERENCE_PATH = GROUP_OUTPUT_DIR / "retail_reference_group.csv"
SCORE_TABLE_PATH = GROUP_OUTPUT_DIR / "retail_score_asset_table.csv"

GROUP_ORDER = ("treated", "matched_control", "least_retail_reference")
MAIN_GROUPS = ("treated", "matched_control")
GROUP_LABELS = {
    "treated": "Retail-treated",
    "matched_control": "Matched control",
    "least_retail_reference": "Least-retail reference",
}

# Sample choices
PRE_PERIOD_START = "2019-06-10 00:00:00"
PRE_PERIOD_END = "2019-09-30 23:59:59"

POST_PERIOD_START = "2019-10-11 00:00:00"
POST_PERIOD_END = "2020-02-19 23:59:59"

# A simple placebo split inside the pre period. The exact midpoint is 
# not sacred. The point is to create a fake post window well before 
# October 2019.
PLACEBO_SPLIT_DATE = "2019-08-07 00:00:00"

# Diagnostic outcomes
OUTCOME_COLS = (
    "dark_share",
    "lit_share",
    "log_total_volume",
    "log_total_realized_variance",
)

OUTCOME_LABELS = {
    "dark_share": "Dark share",
    "lit_share": "Lit share",
    "log_total_volume": "Log total volume",
    "log_total_realized_variance": "Log realized variance",
}

# Benchmark H3 estimation choices
# H3 is a treatment-control extension of the H1/H2 benchmark. We 
# therefore keep the same Menkveld-style lag choice and the same 
# simulation settings.
BENCHMARK_P_LAGS = 2
DEFAULT_RIDGE = 1e-8
N_SIMULATION_DRAWS = 10000
SIMULATION_ALPHA = 0.05
BASE_SEED = 2017

FAMILY_NAMES = ("vix", "macro", "earnings")
BENCHMARK_GROUPS = ("treated", "matched_control")

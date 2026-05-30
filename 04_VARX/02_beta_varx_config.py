"""Configuration for the beta VARX Step 1 build.

In step 1 we:

1. Build a clean baseline panel VARX.
2. Treat the urgency variables as predetermined exogenous regressors.
3. Keep the implementation close to the Menkveld logic, adapted to
   the data we have in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PULL_DIR = PROJECT_ROOT / "01_Data_Pull"
RETAIL_DIR = PROJECT_ROOT / "02_RetailClassification"
VARX_DATA_DIR = PROJECT_ROOT / "03_VARX_Data"
VARX_DIR = PROJECT_ROOT / "04_VARX"

# Legacy alias
BETA_VARX_DIR = VARX_DIR

MINUTE_BAR_DIR = DATA_PULL_DIR / "data_clean" / "minute_bars"
SP500_PATH = DATA_PULL_DIR / "data_clean" / "sp500_tickers.csv"

VIX_PATH = VARX_DATA_DIR / "data_raw" / "VIX.txt"
MACRO_PANEL_PATH = VARX_DATA_DIR / "data_clean" / "macro_news_minute_panel.csv"
EARNINGS_PANEL_PATH = VARX_DATA_DIR / "data_clean" / "earnings_urgency_sparse_panel.csv"

# Sample choices
# The sample stops before the COVID-driven market regime.
SAMPLE_START = "2019-06-10 00:00:00"
SAMPLE_END = "2020-02-19 23:59:59"

# The window from October 1 to October 10, 2019 is treated as the
# exclusion window of the zero-commission war. We exclude it from
# the analysis sample so that the later pre/post comparisons are
# not blurred by the exclusion window itself.
ANALYSIS_EXCLUDE_WINDOWS = (
    ("2019-10-01 00:00:00", "2019-10-10 23:59:59"),
)

# Menkveld work with regular trading hours and exclude the exact open.
REGULAR_SESSION_START = "09:31"
REGULAR_SESSION_END = "16:00"

# Macro announcements require an earlier session because inflation 
# releases are anchored to 08:31 New York time in our processed 
# event panel.
MACRO_SESSION_START = "08:31"
MACRO_SESSION_END = "16:00"

# Variable definitions

# We model dark and lit activity separately and let dark share be 
# derived later from the simulated paths if needed.
Y_COLS = (
    "log_dark_volume_t",
    "log_lit_volume_t",
    "log_total_realized_variance_t",
)

# Common exogenous variables: same value for every stock at a given minute.
VIX_X_COLS = ("dVIX_pos_inv", "dVIX_neg_inv", "VIX_close")
MACRO_X_COLS = (
    "pre_news_1min",
    "post_news_0min",
    "post_news_1min",
    "post_news_2min",
    "post_news_3min",
    "post_news_4min",
)

# Macro decomposition variables. The benchmark pools all 22 macro
# events that survive the exclusion window (8 CPI + 8 PPI + 6 FOMC;
# the source list has 9 CPI and 9 PPI, but the Oct 10 CPI and Oct 8
# PPI releases fall inside the Oct 1-10 exclusion window and are
# dropped at sample-load time) into the combined macro family above.
MACRO_FOMC_X_COLS = (
    "pre_rate_1min",
    "post_rate_0min",
    "post_rate_1min",
    "post_rate_2min",
    "post_rate_3min",
    "post_rate_4min",
)
MACRO_INFLATION_X_COLS = (
    "pre_inflation_1min",
    "post_inflation_0min",
    "post_inflation_1min",
    "post_inflation_2min",
    "post_inflation_3min",
    "post_inflation_4min",
)

# Firm-specific exogenous variables: vary across stocks in the same minute.
EARNINGS_X_COLS = tuple(f"post_ea_{k}" for k in range(1, 14))

# Step 2 and Step 3 follow Menkveld's impulse-response logic. 
#
# - For VIX, the economic focus is on the innovation terms. Our VIX 
#   innovations are measured in index-point changes, so we use a small 
#   0.01-point innovation shock as the clean benchmark.
# - For macro, the event path is built from indicator variables, so 
#   the natural benchmark is an event path with ones in the relevant 
#   minutes.
# - For earnings, Menkveld discuss a 1% EPS surprise. Our earnings 
#   regressors are already scaled surprises, so a 1% shock maps 
#   directly to 0.01.
VIX_IRF_SHOCK_COLS = ("dVIX_pos_inv", "dVIX_neg_inv")
VIX_IRF_SHOCK_SIZE = 0.01
MACRO_EVENT_SHOCK_SIZE = 1.0
EARNINGS_EVENT_SHOCK_SIZE = 0.01

# Estimation defaults
# Menkveld choose two lags based on BIC. That is our starting point 
# here.
DEFAULT_P_LAGS = 2

# A small ridge penalty is useful for numerical stability in dense 
# least squares problems.
DEFAULT_RIDGE = 1e-8

# The minute bars contain zeros on some dimensions. These floors allow 
# us to work with log-transformed variables while avoiding undefined 
# values.
VOLUME_FLOOR = 1.0
REALIZED_VARIANCE_FLOOR = 1e-12

# Ticker normalization
# Massive uses dots for these share-class tickers.
TICKER_RENAMES = {
    "BF-B": "BF.B",
    "BRK-B": "BRK.B",
}


# Structured config objects
@dataclass(frozen=True)
class SessionConfig:
    """Simple trading-session definition."""

    start: str
    end: str


@dataclass(frozen=True)
class ModelConfig:
    """Baseline VARX specification.

    In this baseline we have:
    - pooled coefficient matrices,
    - stock fixed effects,
    - contemporaneous exogenous variables.
    """

    p_lags: int = DEFAULT_P_LAGS
    ridge: float = DEFAULT_RIDGE
    entity_fixed_effects: bool = True
    y_cols: tuple[str, ...] = Y_COLS


REGULAR_SESSION = SessionConfig(
    start=REGULAR_SESSION_START,
    end=REGULAR_SESSION_END,
)

MACRO_SESSION = SessionConfig(
    start=MACRO_SESSION_START,
    end=MACRO_SESSION_END,
)

BASELINE_MODEL = ModelConfig()

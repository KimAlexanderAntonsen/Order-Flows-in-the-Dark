"""Configuration for the H1/H2 estimation layer.

This folder sits on top of ``04_VARX``. The VARX folder is the engine.
The estimation folder uses that engine to estimate the actual pre/post 
regime objects needed for H1 and H2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BETA_VARX_DIR = PROJECT_ROOT / "04_VARX"
ESTIMATION_DIR = PROJECT_ROOT / "05_Estimation_H1_H2"
OUTPUT_DIR = ESTIMATION_DIR / "output" / "h1_h2"

# Regime windows
# The exclusion window is excluded upstream in the beta data layer.
# The regime split below therefore compares the clean pre-period to
# the clean post-period.
PRE_PERIOD_START = "2019-06-10 00:00:00"
PRE_PERIOD_END = "2019-09-30 23:59:59"

POST_PERIOD_START = "2019-10-11 00:00:00"
POST_PERIOD_END = "2020-02-19 23:59:59"

# Benchmark choices carried over from the beta build
BENCHMARK_P_LAGS = 2
DEFAULT_RIDGE = 1e-8

# Step 3 in the beta build already moved away from generic unit 
# shocks. The H1/H2 layer should inherit those calibrated benchmark 
# shocks.
N_SIMULATION_DRAWS = 10000
SIMULATION_ALPHA = 0.05
BASE_SEED = 2017


@dataclass(frozen=True)
class RegimeWindow:
    """Simple period definition for the H1/H2 regime split."""

    name: str
    start: str
    end: str


PRE_WINDOW = RegimeWindow(
    name="pre",
    start=PRE_PERIOD_START,
    end=PRE_PERIOD_END,
)

POST_WINDOW = RegimeWindow(
    name="post",
    start=POST_PERIOD_START,
    end=POST_PERIOD_END,
)


REGIME_WINDOWS = (PRE_WINDOW, POST_WINDOW)
FAMILY_NAMES = ("vix", "macro", "earnings")

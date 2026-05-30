"""Build the focused macro urgency minute panel.

See ``02_urgency_construction.md`` (section 2) for the methodology. The
curated CPI/PPI/FOMC release list is held inline in
``05_macro_urgency_helpers.py``; this script turns that list into a
market-wide minute-level dummy panel used downstream by the VARX.

Inputs:
- ``../01_Data_Pull/data_clean/minute_bars/{ticker}_1m_lit_dark.csv``

Outputs:
- ``data_clean/macro_news_events_clean.csv``
- ``data_clean/macro_news_minute_panel.csv``
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HELPERS_PATH = SCRIPT_DIR / "05_macro_urgency_helpers.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("macro_urgency_helpers", HELPERS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helpers module from {HELPERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the focused macro event panel for the VARX analysis.")
    parser.add_argument(
        "--template-ticker",
        default="AAPL",
        help="Ticker whose minute-bar file should be used to define the trading-minute template.",
    )
    args = parser.parse_args()

    helpers = _load_helpers()

    events_df, panel_df = helpers.build_and_save_outputs(template_ticker=args.template_ticker)

    summary = {
        "n_clean_events": int(len(events_df)),
        "n_inflation_events": int((events_df["event_group"] == "inflation").sum()),
        "n_rate_events": int((events_df["event_group"] == "rate").sum()),
        "panel_rows": int(len(panel_df)),
        "combined_flagged_minutes": int(panel_df[helpers.COMBINED_DUMMY_COLS].sum().sum()),
        "inflation_flagged_minutes": int(panel_df[helpers.INFLATION_DUMMY_COLS].sum().sum()),
        "rate_flagged_minutes": int(panel_df[helpers.RATE_DUMMY_COLS].sum().sum()),
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved clean events to: {helpers.CLEAN_EVENTS_PATH}")
    print(f"Saved macro panel to: {helpers.MACRO_PANEL_PATH}")


if __name__ == "__main__":
    main()

# 03_VARX_Data

This folder stores the upstream files that feed the VARX pipeline in
`04_VARX`. The estimation code itself lives in:

- `04_VARX`
- `05_Estimation_H1_H2`
- `06_Estimation_H3`

## File Structure

Overview markdown:

- `00_README.md` (this file)
- `02_urgency_construction.md`, methodology notes for the urgency panels

Diagnostics script:

- `01_ch3_data_diagnostics.py`

Urgency-construction pipeline (only needed to rebuild the upstream inputs):

- `03_fetch_nasdaq_earnings.py`, pulls the historical Nasdaq earnings calendar
- `04_construct_earnings_urgency.py`, builds the sparse earnings urgency panel
- `05_macro_urgency_helpers.py`, helpers + curated CPI/PPI/FOMC release list
- `06_construct_macro_news_urgency.py`, builds the macro urgency minute panel

Active upstream input files:

- `data_raw/VIX.txt` (from PiTrading)
- `data_raw/earnings_events.csv` (written by `03_fetch_nasdaq_earnings.py`)
- `data_clean/macro_news_minute_panel.csv`
- `data_clean/earnings_urgency_sparse_panel.csv`
- `data_clean/earnings_events_coverage.json`

## Recommended Order Of Use

There is nothing to execute in this folder during a normal rerun; the
data files in `data_clean/` are the inputs consumed downstream.

If the upstream raw data needs to be regenerated from scratch, run the
construction pipeline in order:

1. `03_fetch_nasdaq_earnings.py`
2. `04_construct_earnings_urgency.py`
3. `06_construct_macro_news_urgency.py`

Then move to `04_VARX` and run the VARX pipeline from there.

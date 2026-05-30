"""Build the constant S&P 500 sample.

The project contains a broad 2019 S&P 500 constituent snapshot. This 
script restricts it to the subset of firms that remain in the index 
throughout the full market-data window from 2019-06-10 to 2020-02-19 
and writes it to a csv file.

In set notation, the output universe is

    U_const = U_2019-06-10 \\ E

where E is the set of firms that leave the broad June 10, 2019 
universe during the sample window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean"

# The broad 2019 constituent snapshot already exists in the project. 
# It is the natural input for building the constant sample.
BROAD_2019_PATH = DATA_CLEAN_DIR / "sp500_tickers_2019.csv"

# Outputs written by this script.
CANONICAL_PATH = DATA_CLEAN_DIR / "sp500_tickers.csv"
EXCLUSIONS_PATH = DATA_CLEAN_DIR / "sp500_constant_sample_exclusions.csv"
AUDIT_PATH = DATA_CLEAN_DIR / "sp500_constant_sample_audit.csv"


@dataclass(frozen=True)
class SampleExit:
    """Ticker that leaves the broad sample during the data window."""

    ticker: str
    effective_date: str
    note: str


# These are the names present in the broad 2019 snapshot that cease to
# be valid S&P 500 members during the sample window. Excluding them 
# yields the constant-membership sample used downstream.
SAMPLE_WINDOW_EXITS = (
    SampleExit("AMCR", "2019-06-11", "Entered after the sample start via the Bemis/Amcor treatment; excluded from the constant start-of-sample universe."),
    SampleExit("RHT", "2019-07-15", "Removed after IBM acquired Red Hat."),
    SampleExit("APC", "2019-08-09", "Removed after Occidental acquired Anadarko."),
    SampleExit("FL", "2019-08-09", "Removed in market-cap driven S&P 500 change."),
    SampleExit("TSS", "2019-09-23", "Removed after Global Payments acquired TSYS."),
    SampleExit("JEF", "2019-09-26", "Removed after the Jefferies/SPB constituent change."),
    SampleExit("NKTR", "2019-10-03", "Removed in market-cap driven S&P 500 change."),
    SampleExit("CELG", "2019-11-21", "Removed after Bristol-Myers Squibb acquired Celgene."),
    SampleExit("VIAB", "2019-12-05", "Removed in the CBS/Viacom transaction."),
    SampleExit("STI", "2019-12-09", "Removed after BB&T acquired SunTrust to form Truist."),
    SampleExit("AMG", "2019-12-23", "Removed in market-cap driven S&P 500 change."),
    SampleExit("TRIP", "2019-12-23", "Removed in market-cap driven S&P 500 change."),
    SampleExit("MAC", "2019-12-23", "Removed in market-cap driven S&P 500 change."),
    SampleExit("WCG", "2020-01-28", "Removed after Centene acquired WellCare."),
    SampleExit("ALB", "2019-06-10", "Excluded from the constant start-of-sample universe; no minute-bar data available in the project pull."),
    SampleExit("LLL", "2019-07-01", "Removed after the L3 Technologies / Harris merger formed L3Harris (LHX)."),
    SampleExit("TMK", "2019-08-07", "Removed after Torchmark rebranded as Globe Life (GL)."),
    SampleExit("BRK.B", "2019-06-10", "Excluded from the constant start-of-sample universe; minute-bar pull has no post-period data."),
)


def build_constant_sample() -> None:
    """Create the constant-sample universe files."""

    if not BROAD_2019_PATH.exists():
        raise FileNotFoundError(
            f"Broad 2019 constituent file not found: {BROAD_2019_PATH}"
        )

    broad = pd.read_csv(BROAD_2019_PATH)
    if "ticker" not in broad.columns:
        raise ValueError(f"{BROAD_2019_PATH} must contain a 'ticker' column.")

    # Canonical convention used downstream: tickers stored with a dot for
    # class-B share notation (BRK.B, BF.B), matching the Massive minute-bar
    # filenames. The broad 2019 snapshot uses the Wikipedia-style dash form
    # for those names; we normalise here so sp500_tickers.csv has 
    # the dot form.
    broad["ticker"] = (
        broad["ticker"].astype(str).str.upper().str.replace("-", ".", regex=False)
    )

    excluded = pd.DataFrame([entry.__dict__ for entry in SAMPLE_WINDOW_EXITS])
    excluded["ticker"] = excluded["ticker"].str.upper().str.replace("-", ".", regex=False)

    broad_set = set(broad["ticker"])
    missing_exclusions = sorted(set(excluded["ticker"]) - broad_set)
    if missing_exclusions:
        raise ValueError(
            "Some exclusion tickers are not present in the broad 2019 file: "
            f"{missing_exclusions}"
        )

    constant = (
        broad.loc[~broad["ticker"].isin(excluded["ticker"]), ["ticker"]]
        .drop_duplicates()
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    # Sanity check: the constant sample should be broad - excluded.
    expected = len(set(broad["ticker"])) - len(set(excluded["ticker"]))
    if len(constant) != expected:
        raise ValueError(
            f"Constant sample size {len(constant)} does not match "
            f"broad ({len(set(broad['ticker']))}) - exclusions "
            f"({len(set(excluded['ticker']))}) = {expected}."
        )

    constant.to_csv(CANONICAL_PATH, index=False)
    excluded.to_csv(EXCLUSIONS_PATH, index=False)

    audit = pd.DataFrame(
        [
            {"metric": "broad_2019_snapshot_count", "value": int(len(broad))},
            {"metric": "excluded_during_sample_count", "value": int(len(excluded))},
            {"metric": "constant_sample_count", "value": int(len(constant))},
        ]
    )
    audit.to_csv(AUDIT_PATH, index=False)

    print(f"Updated canonical downstream file: {CANONICAL_PATH}")
    print(f"Saved exclusion table: {EXCLUSIONS_PATH}")
    print(f"Saved audit table: {AUDIT_PATH}")
    print(f"Broad 2019 count: {len(broad)}")
    print(f"Excluded during sample: {len(excluded)}")
    print(f"Constant sample count: {len(constant)}")


if __name__ == "__main__":
    build_constant_sample()

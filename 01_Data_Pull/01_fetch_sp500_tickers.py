"""Fetch and map the S&P 500 ticker universe used in the project.

This script builds the broad June 10, 2019 universe in two steps. 
First it downloads a current constituent list. It then maps that 
list back to the start-of-sample universe with the manual 
add/remove bridge.

In set notation, the constructed start-of-sample universe is

    U_2019-06-10 = (U_current \\ R) \\cup A

where R is the set of post-2019 entrants that must be removed and 
A is the set of June 2019 constituents that must be added back.

Outputs:
- data_clean/sp500_tickers_2019.csv  (broad 2019 snapshot)

Note:
The canonical downstream universe lives at data_clean/sp500_tickers.csv
and is written by 02_build_constant_sp500_sample.py.

WARNING: live Wikipedia fetch.
Running this script re-fetches the current S&P 500 constituent list from
Wikipedia as of the moment you run it. S&P 500 membership turns over
continually, so the list you pull today is almost certainly not the same
list that was used to build the thesis results (the thesis was written in
early 2026 against the then-current Wikipedia snapshot). The ADD_2019 and
REMOVE_POST_2019 bridges below are also calibrated against that earlier
snapshot. Re-running this script can therefore shift the broad 2019
snapshot in ways that propagate through every downstream stage. The
broad 2019 snapshot is treated as fixed historical data; the guard at the
bottom of the file prevents accidental refetches by requiring the user to
manually delete sp500_tickers_2019.csv before this script will run.
"""

from __future__ import annotations

import os
import re
import ssl
from pathlib import Path

import certifi
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "01_Data_Pull" / "data_clean"

TARGET_SP500_DATE = "2019-06-10"
SNAPSHOT_2019_PATH = DATA_CLEAN_DIR / "sp500_tickers_2019.csv"
CANONICAL_PATH = DATA_CLEAN_DIR / "sp500_tickers.csv"


# These manual lists convert the current constituent pull into the
# S&P 500 universe at the start of the minute-bar sample.
ADD_2019_TICKERS = [
    "AAL", "AAP", "ABC", "ABMD", "ADS", "AGN", "AIV", "ALK", "ALXN", 
    "AMG", "ANSS", "ANTM", "APC", "ARNC", "ATVI", "BBT", "BHGE", 
    "BLL", "BWA", "CBS", "CE", "CELG", "CERN", "CMA", "COG", "COTY", 
    "CPRI", "CTL", "CTXS", "CXO", "DFS", "DISCA", "DISCK", "DISH", 
    "DRE", "DXC", "EMN", "ETFC", "FB", "FBHS", "FISV", "FL", "FLIR", 
    "FLS", "FLT", "FMC", "FRC", "FTI", "GPS", "HBI", "HCP", "HES", 
    "HFC", "HOG", "HP", "HRB", "ILMN", "INFO", "IPG", "IPGP", "JEC", 
    "JEF", "JNPR", "JWN", "K", "KMX", "KSS", "KSU", "LB", "LEG", 
    "LKQ", "LLL", "LNC", "M", "MAC", "MHK", "MMC", "MRO", "MXIM", 
    "MYL", "NBL", "NKTR", "NLSN", "NOV", "NWL", "PBCT", "PKI", 
    "PRGO", "PVH", "PXD", "QRVO", "RE", "RHI", "RHT", "RTN", "SEE", 
    "SIVB", "SLG", "STI", "SYMC", "TFX", "TIF", "TMK", "TRIP", 
    "TSS", "TWTR", "UA", "UAA", "UNM", "UTX", "VAR", "VFC", "VIAB",
    "VNO", "WBA", "WCG", "WHR", "WLTW", "WRK", "WU", "XEC", "XLNX", 
    "XRAY", "XRX", "ZION"
]

REMOVE_POST_2019_TICKERS = [
    "ABNB", "ACGL", "ALB", "APP", "APO", "ARES", "AXON", "BALL", "BG",   
    "BKR", "BLDR", "BRO", "BX", "CARR", "CASY", "CDW", "CEG", "CIEN", 
    "COIN", "COR", "COHR", "CPAY", "CPT","CRL", "CRH", "CRWD", "CSGP", 
    "CTRA", "CVNA", "CZR", "DASH", "DAY", "DDOG", "DECK", "DELL", 
    "DOC", "DPZ", "DXCM", "EG", "ELV", "EME", "ENPH", "EPAM", 
    "EQT", "ERIE", "EXE", "FDS", "FI", "FICO", "FIX", "FSLR", 
    "GDDY", "GEHC", "GEN", "GEV", "GL", "GNRC", "HOOD", "HUBB", 
    "HWM", "IBKR", "IEX", "INVH", "J", "JBL", "KDP", "KKR", "KVUE",
    "LDOS", "LII", "LITE", "LULU", "LVS", "LYV", "META", "MKTX", "MOH", 
    "MPWR", "MRNA", "MRSH", "MTCH", "NDSN", "NOW", "NVR", "NXPI", 
    "ODFL", "ON", "OTIS", "PANW", "PAYC", "PCG", "PLTR", "PODD", 
    "POOL", "PSKY", "PTC", "Q", "RTX", "RVTY", "SATS", "SMCI", "SNDK", 
    "SOLV", "STE", "STLD", "SW", "TDY", "TECH", "TER", "TFC", 
    "TKO", "TMUS", "TPL", "TRGP", "TRMB", "TSLA", "TT", "TTD", 
    "TYL", "UBER", "VEEV", "VRT", "VICI", "VLTO", "VST", "VTRS", "WBD", 
    "WDAY", "WRB", "WSM", "WST", "WTW", "XYZ", "ZBRA"
]


def fetch_current_sp500_tickers() -> list[str]:
    """Download a broad current constituent set with layered fallbacks."""

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Use certifi for robust HTTPS in environments.
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )

    try:
        tables = pd.read_html(url)
        table = next(table for table in tables if "Symbol" in table.columns)
        tickers = table["Symbol"].astype(str).str.strip().tolist()
    except Exception:
        try:
            csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            tickers = pd.read_csv(csv_url)["Symbol"].astype(str).str.strip().tolist()
        except Exception:
            api = "https://en.wikipedia.org/w/api.php"
            response = requests.get(
                api,
                params={
                    "action": "parse",
                    "page": "List_of_S%26P_500_companies",
                    "prop": "text",
                    "format": "json",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            html = data.get("parse", {}).get("text", {}).get("*", "")
            table_match = re.search(
                r'(<table[^>]*class=\"wikitable[^\\\"]*\"[\\s\\S]*?</table>)', html
            )
            table_html = table_match.group(1) if table_match else html
            rows = re.findall(r"<tr>([\\s\\S]*?)</tr>", table_html)
            tickers = []
            for row in rows:
                match = re.search(r"<td[^>]*>.*?<a[^>]*>([^<]+)</a>", row)
                if not match:
                    continue
                symbol = match.group(1).strip()
                if re.match(r"^[A-Z0-9\\.\\-]+$", symbol):
                    tickers.append(symbol)

    normalized = []
    seen = set()
    for ticker in tickers:
        clean = str(ticker).strip().upper().replace(".", "-")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def build_start_sample_universe(current_tickers: list[str]) -> list[str]:
    """Map the current constituent pull into the June 10, 2019 universe."""

    effective_adds = sorted(set(ADD_2019_TICKERS) - set(current_tickers))
    effective_removes = sorted(set(REMOVE_POST_2019_TICKERS) & set(current_tickers))

    universe_2019 = sorted(
        (set(current_tickers) - set(effective_removes)) | set(effective_adds)
    )

    print(f"Target date: {TARGET_SP500_DATE}")
    print(f"Current pull count: {len(current_tickers)}")
    print(f"2019 mapped count: {len(universe_2019)}")
    print(f"Effective manual adds: {len(effective_adds)}")
    print(f"Effective manual removals: {len(effective_removes)}")

    return universe_2019


def write_universe_files(tickers_2019: list[str]) -> None:
    """Write the broad 2019 snapshot."""

    DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"ticker": tickers_2019})
    out.to_csv(SNAPSHOT_2019_PATH, index=False)

    print(f"Saved 2019 constituent file: {SNAPSHOT_2019_PATH}")
    print("Next step: run 02_build_constant_sp500_sample.py to build the constant-membership universe at sp500_tickers.csv.")


def main() -> None:
    """Build the broad June 10, 2019 universe used by the next script."""

    current_tickers = fetch_current_sp500_tickers()
    tickers_2019 = build_start_sample_universe(current_tickers)
    write_universe_files(tickers_2019)


if __name__ == "__main__":
    # Guard the Wikipedia fetch: the broad 2019 snapshot is fixed historical
    # data and shouldn't normally be refetched. Delete the file to force a
    # rebuild.
    if os.path.exists(SNAPSHOT_2019_PATH):
        print(
            f"Warning: {SNAPSHOT_2019_PATH} already exists. "
            "Delete the file to refetch from Wikipedia (which needs manual "
            "ADD/REMOVE bridge updating)."
        )
    else:
        main()

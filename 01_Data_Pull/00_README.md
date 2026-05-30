# 01_Data_Pull

This folder contains the upstream market-data construction for the project.
It builds the constant S&P 500 universe and the one-minute lit and dark
market panel that later feeds `02_RetailClassification`, `04_VARX`,
`05_Estimation_H1_H2`, and `06_Estimation_H3`.

## File Structure

Overview markdown:

- `00_README.md`

Scripts:

- `01_fetch_sp500_tickers.py`
- `02_build_constant_sp500_sample.py`
- `03_massive_fetch.py`

Core outputs:

- `data_clean/sp500_tickers.csv`
- `data_clean/sp500_constant_sample_exclusions.csv`
- `data_clean/sp500_constant_sample_audit.csv`
- `data_clean/minute_bars/*.csv`

## Construction Logic

The universe build has two stages. First, the project forms the historical start-of-sample index membership:

$$\mathcal{U}_{\text{2019-06-10}} = \left(\mathcal{U}_{current} \setminus \mathcal{R}\right) \cup \mathcal{A},$$

and then the constant-membership universe is

$$\mathcal{U}^{const} = \mathcal{U}_{\text{2019-06-10}} \setminus \mathcal{E},$$

where $\mathcal{E}$ is the set of firms that leave the broad June 2019 universe during the sample window. This yields the constant stock universe used downstream.

The trade aggregation step then converts raw trades into venue-specific minute bars. For each stock $i$ and minute $t$:

$$V^{dark}_{i,t} = \sum_{k \in \mathcal{D}_{i,t}} q_{i,k}, \qquad V^{lit}_{i,t} = \sum_{k \in \mathcal{L}_{i,t}} q_{i,k},$$

and venue-specific realized variance is

$$RV^{seg}_{i,t} = \sum_{k \in seg,t} (\Delta \log p_{i,k})^2.$$

## Recommended Order Of Execution

1. Read `00_README.md` (this file)
2. Run `01_fetch_sp500_tickers.py` if the broad constituent pull needs to be refreshed. Note: this pulls live from Wikipedia and needs manual adjustment. 
3. Run `02_build_constant_sp500_sample.py` to rebuild the canonical constant-membership universe
4. Run `03_massive_fetch.py` only if the minute bars need to be rebuilt. (Requires massive api-key, defined in .env, to fetch data)

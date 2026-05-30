# 02_RetailClassification

This folder constructs the retail-intensity classification used later in the
H3 analysis. It starts from merged Robintrack-holder and market-volume inputs,
estimates a pre-period retail score, and then defines the treated,
matched-control, and least-retail reference groups used downstream.

**Sample.** The retail classification operates on a subset of the constant-membership universe. The firm drop is almost
entirely explained by Robintrack archive coverage, firms without a
`*_rh_with_massive_volume.csv` file in `data_clean/` are silently skipped
by the implementation script. Four additional tickers (`LEN`, `STZ`,
`TAP`, `MKC`) are removed for known data irregularities in the holder
series (`EXCLUDE_TICKERS` in the implementation script). A minimum of
`MIN_OBS = 100` valid pre-period observations is required for a firm to
receive a score; in practice this threshold is non-binding. 

The H3 group selection is then restricted to the scored firms that have
at least one active earnings-event row in both the pre and the post
window (the earnings-eligibility filter described in
`load_earnings_eligible_tickers`), so that the same treated and
matched-control groups can be reused across the VIX, macro, and
earnings urgency families. From this firm pool, the top **70**
firms by retail score form the treated group,
70 matches drawn from the remaining earnings-eligible
firms form the main control group, and the bottom **70** firms form the least-retail reference group. The fixed group size of 70 is set by `H3_GROUP_SIZE` in the implementation
script.

## File Structure

Overview markdown:

- `00_README.md` (this file)

Scripts:

- `01_retail_classification_implementation.py`

Core inputs used by the implementation script:

- `data_clean/*_rh_with_massive_volume.csv`
- `../01_Data_Pull/data_clean/minute_bars/*.csv`
- `../01_Data_Pull/data_clean/sp500_tickers.csv`

Core outputs used later in the project:

- `group_outputs/retail_score_asset_table.csv`
- `group_outputs/retail_score_with_matching_features.csv`
- `group_outputs/retail_treated_group.csv`
- `group_outputs/retail_matched_control_group.csv`
- `group_outputs/retail_reference_group.csv`
- `group_outputs/retail_group_balance_summary.csv`
- `group_outputs/retail_group_balance_smd.csv`

## Classification Logic

For stock $i$ and interval $t$, the script defines

$$y_{i,t} = \log\left(1 + |\Delta H_{i,t}|\right), \qquad x_{i,t} = \log\left(1 + V_{i,t}\right),$$

where $\Delta H_{i,t}$ is the Robintrack holder change and $V_{i,t}$ is the local market volume aligned to the Robintrack interval (`massive_volume` in `data_clean/*_rh_with_massive_volume.csv`). A pooled LOWESS benchmark $\hat m(x)$ is estimated on a random sub-sample of $N=10^5$ pre-period observations (smoothing fraction $f=0.20$, one robustifying iteration, seed 0) and then applied to every interval by linear interpolation. The retail residual is

$$u_{i,t} = y_{i,t} - \hat m(x_{i,t}).$$

The stock-level retail score is the mean winsorized residual

$$RS_i = \frac{1}{T_i} \sum_{t=1}^{T_i} \tilde u_{i,t}, \qquad \tilde u_{i,t} = \mathrm{clip}\!\left(u_{i,t},\, q_{0.01},\, q_{0.99}\right),$$

where $q_{0.01}$ and $q_{0.99}$ are the 1st and 99th percentiles of the pooled residual distribution.

The matched control group is then formed by one-to-one assignment without replacement, minimizing Euclidean distance in standardized pre-period market characteristics:

$$d(i,j) = \sqrt{\sum_k \left(z_{ik} - z_{jk}\right)^2}, \qquad z_{ik} = \frac{x_{ik} - \bar{x}_k}{s_k},$$

with $\bar{x}_k$ and $s_k$ computed on the pooled treated-plus-control-candidate sample. The matched set $\pi^{\ast}$ solves

$$\pi^{\ast} = \arg\min_{\pi} \sum_{i \in \text{treated}} d\!\left(i,\, \pi(i)\right),$$

over bijections $\pi$ from treated firms to the earnings-eligible non-treated pool.

## Recommended Order Of Execution

1. Read `00_README.md` (this file)
2. Run `01_retail_classification_implementation.py` to rebuild the score table and the three group definitions
3. Use the saved `group_outputs` files in the later H3 estimation folder

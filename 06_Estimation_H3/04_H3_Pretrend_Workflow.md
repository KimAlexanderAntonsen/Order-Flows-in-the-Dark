# H3 Group Diagnostics Block (Files 01–04)

This file is the code-companion for the first block of `06_Estimation_H3`. It describes what files `01` to `04` compute and which outputs they write.

## 1. Purpose Of The Block

Before the H3 treatment-control extension is estimated in files `05`–`07`, the project first builds a small pre-period diagnostic battery that compares the treated group to two candidate control groups (a one-to-one nearest-neighbour distance-matched control on four pre-period features, and a least-retail reference). 

## 2. Files In The Block

- `01_h3_config.py`, sample window, group file paths, placebo split date, output locations.
- `01_Step_1_Pretrend_And_Comparability.md`, descriptive companion that explains the diagnostic logic at a higher level.
- `02_h3_pretrend_diagnostics.py`, helper layer that builds the diagnostic panel and computes the four diagnostic tables.
- `03_run_h3_pretrend_diagnostics.py`, runner that calls the helper and saves all CSVs and figures.
- `04_h3_group_cells_and_scatter.py`, separate runner that builds the retail-score histogram, the retail-score-vs-volume scatter, the retail-score-vs-market-cap scatter, the four-cell group-by-regime dark-share table, and the five-row Welch balance table for the chapter.
- This file, code-companion for everything in the block.

## 3. Sample And Groups

The diagnostic block uses only the pre window (`PRE_PERIOD_START` -> `PRE_PERIOD_END` in `01_h3_config.py`, i.e. 2019-06-10 to 2019-09-30). The exclusion window 2019-10-01 to 2019-10-10 is excluded upstream in `04_VARX/04_beta_varx_data.py`.

Three groups are read from CSV in `02_h3_pretrend_diagnostics.py:load_group_definitions`:

- `treated`, from `TREATED_PATH`
- `matched_control`, from `MATCHED_CONTROL_PATH`
- `least_retail_reference`, from `REFERENCE_PATH`

## 4. Stock-Day Panel Construction

`_collapse_stock_to_days` (in `02_h3_pretrend_diagnostics.py`) reads one stock's minute-bar file, filters to the regular trading session, restricts to the pre window, and aggregates to daily totals. The derived columns it produces per stock-day are:

- `dark_volume`, `lit_volume`, `total_volume` (sums of minute totals)
- `total_realized_variance` (sum of dark + lit minute realized variance)
- `dark_share`, `lit_share` (ratios of the volume aggregates; the divide is masked on zero-volume days to avoid NaNs)
- `log_total_volume`, `log_total_realized_variance` (via `safe_log` with the same floors used by the VARX layer)
- `minute_obs` (a coverage diagnostic, how many minute observations fed each daily row)
- `week_start` (Monday-anchored, used by the weekly aggregation later)

The four "outcome" columns used by all downstream diagnostics are `OUTCOME_COLS` from `01_h3_config.py`: `dark_share`, `lit_share`, `log_total_volume`, `log_total_realized_variance`.

`build_pretrend_stock_day_panel` concatenates these per-stock daily frames across all three groups into one long panel keyed by `(group, asset, date)`.

## 5. Four Diagnostic Tables

The helper module exposes four `build_*` functions that all consume the same stock-day panel:

**`build_sample_summary`**, group sizes and coverage. Reports `n_assets`, `n_stock_days`, `mean_stock_days`, `first_date`, `last_date` per group. Tells the reader whether the three groups have comparable coverage in the pre window before any inference is run.

**`build_comparability_summary`**, pre-period level balance on stock means. Collapses each stock to its pre-period mean of the four outcome columns, then for each outcome reports:
- group means (treated, matched, reference)
- raw treated-minus-matched and treated-minus-reference differences
- standardized mean differences (SMD) using the pooled-variance denominator $\sqrt{(Var_{a}+Var_{b})/2}$

The SMD is a unitless balance metric; the function returns it without any "good vs bad" threshold.

**`build_trend_test_summary`**, linear pre-trend interaction tests. For each control group (`matched_control`, `least_retail_reference`), it stacks the treated and control stocks, computes `day_index = (date − date.min()).days`, and fits

```text
outcome ~ day_index + treated_flag:day_index + C(asset)
```

via `statsmodels` OLS with cluster-robust standard errors on `asset`. The reported coefficient is on the `treated_flag:day_index` interaction. Rows include `coef`, `std_error`, `t_stat`, `p_value`, `n_obs`, `n_assets`.

**`build_placebo_did_summary`**, placebo DiD around a fake event date. For each control group, the helper constructs `pseudo_post = (date ≥ PLACEBO_SPLIT_DATE)` (where `PLACEBO_SPLIT_DATE = 2019-08-07 00:00:00` in `01_h3_config.py`) and fits

```text
outcome ~ pseudo_post + treated_flag:pseudo_post + C(asset)
```

with the same cluster-robust SE. The reported coefficient is on `treated_flag:pseudo_post`. The same date is reused by the H1/H2 pre-stability runner (`05_Estimation_H1_H2/18_run_h1_h2_pre_stability.py`) and the H3 pre-stability runner (`06_Estimation_H3/17_run_h3_pre_stability.py`) so the placebo geometry is shared across the project.

## 6. Group And Gap Series

Two complementary plotting objects are also built:

- `build_group_series(frequency=...)`, collapses the stock-day panel into per-group averages at daily or weekly frequency. Weekly averaging is done at the stock level first (`groupby(group, asset, week_start).mean()`) before averaging across stocks, so weekly plots are not dominated by a few high-volume names.
- `build_gap_series`, pairs each control group with the treated group on a common time axis and reports `outcome_gap = treated − control` per outcome, for both `matched_control` and `least_retail_reference` comparisons and both frequencies.

`plot_group_series` and `plot_gap_series` (helpers in the same module) draw 2×2 outcome panels and save PNGs to `FIGURE_DIR`.

## 7. File 04: Group-Cells And Scatter Outputs

`04_h3_group_cells_and_scatter.py` is a separate runner from the pretrend-diagnostics pipeline. It uses the same group files and pre-period concept but produces objects for the chapter rather than for the H3 estimator:

- `retail_score_distribution.pdf`, histogram of the pre-period retail-intensity score across all scored firms.
- `retail_score_vs_volume.pdf`, scatter of retail score vs log mean-minute trading volume, with treated and matched-control highlighted.
- `retail_score_vs_mktcap.pdf`, scatter of retail score vs average market capitalisation (log-scaled axis), with treated and matched-control highlighted.
- `group_cells.tex`, LaTeX snippet for the four-cell (group x regime) mean-dark-share table with Welch t-tests on each treated-vs-matched gap and on the post-minus-pre DiD.
- `group_balance.tex`, five-row Welch balance table (retail score plus four matching features).
- `group_cells_table.csv`, `group_cells_levels_table.csv`, the row-level numbers behind the two LaTeX snippets.

The figure and LaTeX outputs target `07_Thesis/latex/`; the supporting CSVs go to `06_Estimation_H3/output/`.

## 8. Saved Outputs From The Runner

`03_run_h3_pretrend_diagnostics.py` writes the following to `TABLE_DIR` (i.e. `06_Estimation_H3/output/pretrend/tables/`):

- `h3_pretrend_stock_day_panel.csv`
- `h3_pretrend_daily_group_series.csv`, `h3_pretrend_weekly_group_series.csv`
- `h3_pretrend_daily_gap_series.csv`, `h3_pretrend_weekly_gap_series.csv`
- `h3_pretrend_sample_summary.csv`
- `h3_pretrend_comparability_summary.csv`
- `h3_pretrend_trend_tests.csv`
- `h3_pretrend_placebo_did_summary.csv`
- `h3_retail_score_asset_table_snapshot.csv`

and to `FIGURE_DIR` (i.e. `06_Estimation_H3/output/pretrend/figures/`):

- `h3_pretrend_daily_group_series.png`, `h3_pretrend_weekly_group_series.png`
- the matching gap-series PNGs

These are the artifacts the chapter and the LaTeX-export layer read; no inference downstream re-derives them from the minute panel.

## 9. Recommended Order

`03_run_h3_pretrend_diagnostics.py` is the canonical entry point for this block. `04_h3_group_cells_and_scatter.py` can be run before or after, it does not depend on any artifact written by `03`.

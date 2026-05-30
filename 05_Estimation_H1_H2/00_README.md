# 05_Estimation_H1_H2

This folder sits on top of `04_VARX` and uses the VARX engine defined there to estimate the `H1/H2` pre/post regime objects and to turn them into presentation-ready tables and figures. 

## Dependency On 04_VARX

H1/H2 reuses the VARX engine in `04_VARX` directly: it imports the panel construction, estimator, IRF, and inference modules from there rather than reading any saved CSV. The runtime dependency is therefore on the **input data layer**, not on the VARX output files. Concretely, the pipeline needs:

1. The minute-bar files in `01_Data_Pull/data_clean/minute_bars/`,
2. The processed urgency panels in `03_VARX_Data/data_clean/` (`macro_news_minute_panel.csv`, `earnings_urgency_sparse_panel.csv`) and the VIX feed in `03_VARX_Data/data_raw/VIX.txt`,
3. The constant-membership universe in `01_Data_Pull/data_clean/sp500_tickers.csv`.

Running the standalone Step 1, Step 2, and Step 3 scripts in `04_VARX` (`07_run_step1_baseline.py`, `11_run_step2_irf.py`, `14_run_step3_inference.py`) is not required for the H1/H2 estimation to execute. Those scripts are useful for auditing the baseline (constant-universe sample audit, single-period coefficient tables) and for the mechanics-check diagnostics (recursion-vs-companion equivalence, horizon-0 contemporaneous check), but the H1/H2 runner re-fits the model from scratch on the pre and post regime windows.

## Methodology

The implementation stays close to Menkveld et al. (2017) on the dimensions that matter for the dynamic analysis: a panel VARX with stock fixed effects, lag length `p = 2`, urgency variables entering as predetermined exogenous regressors, simulation-based IRFs, and simulation-based confidence bands. The adaptation is that our data only allow a lit-versus-dark execution split rather than Menkveld's richer dark-venue categories, so the present estimation focuses on reduced venue-share objects built from dark and lit activity.

**Regime windows.** The benchmark is estimated separately in two regimes:

- pre period: `2019-06-10` to `2019-09-30`
- post period: `2019-10-11` to `2020-02-19`

The exclusion window `2019-10-01` to `2019-10-10` is excluded as the introduction window of the zero-commission war.

**Benchmark equation.** In each regime, the panel VARX of Section 4.1 of the thesis is

$$
y_{j,t} = \alpha_j + \Phi_1 \, y_{j,t-1} + \Phi_2 \, y_{j,t-2} + \Gamma \, z_{j,t} + \varepsilon_{j,t},
$$

where $j$ indexes stocks, $t$ indexes minutes, $\alpha_j$ is the stock fixed effect, $\Phi_1, \Phi_2$ are the autoregressive matrices, $z_{j,t}$ stacks the exogenous urgency variables for the family being estimated, and $\Gamma$ collects the coefficients on the exogenous block. The implementation splits $z_{j,t}$ into a market-wide piece $c_t$ (joined on timestamp only) and a firm-specific piece $f_{j,t}$ (joined on stock and timestamp), with the coefficient block $\Gamma = [\, B \mid G \,]$ matching the split.

**Main H1/H2 comparison object.** For each horizon $h$,

$$
\Delta \mathrm{IRF}_h = \mathrm{IRF}_h^{\text{post}} - \mathrm{IRF}_h^{\text{pre}}.
$$

**Venue-share interpretation.** The model is estimated on the reduced dark-versus-lit system, but the figures and tables report the derived dark share (in percent), recovered ex post from the simulated volume paths as in the thesis chapter 3 variable table:

$$
\mathrm{DarkShare}_{j,t} = 100 \times \frac{V^{\mathrm{dark}}_{j,t}}{V^{\mathrm{dark}}_{j,t} + V^{\mathrm{lit}}_{j,t}}, \qquad \mathrm{LitShare}_{j,t} = 100 - \mathrm{DarkShare}_{j,t}.
$$

**Urgency inputs.** `VIX` uses the intraday PiTrading file with the calibrated positive/negative innovations and the level control. Macro uses `macro_news_minute_panel.csv` (built from `24` source events; the Oct 10 CPI and Oct 8 PPI releases fall inside the Oct 1-10 exclusion window and are dropped at sample-load time, leaving `22` analysis events: `16` inflation + `6` FOMC), shocked over minutes `-1` to `+4`. Earnings uses `earnings_urgency_sparse_panel.csv` expanded into the `13` half-hour event blocks.

**Lag choice.** The benchmark keeps `p = 2`. `13_run_h1_h2_lag_selection.py` runs the BIC/AIC sweep over `p ∈ {1, 2, 3, 4}` for the appendix, and `09_run_h1_h2_p3_robustness.py` reruns the full split-sample workflow at `p = 3`.

**Two-way Petersen clustered SE.** The benchmark covariance clusters both by stock and by minute timestamp. Per-family diagnostics (minimum eigenvalue of the two-way combiner and PSD-repair flag) are written alongside each coefficient table and stacked into `h1h2_cluster_diagnostics_summary.csv` for the appendix.

**VIX shock calibration.** VIX-family IRFs are reported at a 1-σ shock to the positive-innovation series $\Delta\mathrm{VIX}_t^{*+}$ (column `dVIX_pos_inv` in the code, $\sigma$ estimated on the full analysis sample). The $\sigma$ is computed once and cached in `output/h1_h2/vix_dvix_pos_sigma.txt`.

**Macro decomposition diagnostic.** The benchmark macro family pools all 22 analysis events (24 source events minus the Oct 10 CPI and Oct 8 PPI releases that fall inside the Oct 1-10 exclusion window) into a single shock. Because 16 of them fire pre-market at 08:30 ET and only the 6 FOMC events at 14:00 ET fall in the regular session, `16_run_h1_h2_macro_decomposition.py` re-fits the VARX on two disjoint event subsets:

- `macro_fomc`: the 6 FOMC rate-decision events (14:00 ET, regular session)
- `macro_inflation`: the 16 CPI/PPI events (08:30 ET, pre-market)

Both reuse the benchmark VARX engine and pre/post windows; only the exogenous block changes. Outputs are written to `output/h1_h2/` with the `macro_fomc_` and `macro_inflation_` prefixes. The FOMC subset is identified off 3 events per regime. 

## File Map

Documentation files:

- `00_README.md` (this file)
- `04_H1_H2_Presentation.md`
- `07_H1_H2_Implementation.md`

Core estimation:

- `01_estimation_config.py`, regime windows and benchmark settings
- `02_estimation_h1_h2.py`, helper layer that reuses the VARX engine for pre/post estimation
- `03_run_h1_h2_estimation.py`, main runner for the benchmark `H1/H2` estimation

Presentation:

- `05_h1_h2_presentation.py`, helper layer for Menkveld-style figures and tables
- `06_run_h1_h2_presentation.py`, main runner for the presentation outputs

Robustness:

- `08_h1_h2_robustness.py`, helper for the final `p = 3` robustness pass
- `09_run_h1_h2_p3_robustness.py`, main runner for the split-sample `p = 3` robustness check

LaTeX export:

- `10_h1_h2_latex_tables.py`, helper for LaTeX table snippets
- `11_run_h1_h2_latex_tables.py`, main runner for the LaTeX export layer
- `12_compile_h1_h2_latex_tables.py`, compiles the saved LaTeX snippets into local PDF previews with `tectonic`

Lag selection:

- `13_run_h1_h2_lag_selection.py`, BIC/AIC sweep for `p ∈ {1, 2, 3, 4}` across the three families and both regimes; writes the appendix-ready CSV

Menkveld-style descriptive extras:

- `14_h1_h2_menkveld_extras.py`, helper for the daily dark-share figure, pre/post descriptive stats, and stacked cluster diagnostics
- `15_run_h1_h2_menkveld_extras.py`, main runner for those three objects

Macro decomposition:

- `16_run_h1_h2_macro_decomposition.py`, FOMC-only and CPI/PPI-only macro refits described above

## Recommended Order Of Execution

### 1. Run the H1/H2 estimation layer

- 03_run_h1_h2_estimation.py

This loads the constant stock universe, reuses the cleaned VARX pipeline, splits into pre/post regimes, estimates the benchmark VARX in each regime, builds pre, post, and `post_minus_pre` objects for `VIX`, `macro`, and `earnings`, and saves to `output/h1_h2/`.

### 2. Run the presentation layer

- 06_run_h1_h2_presentation.py

This reads the saved estimation outputs and writes Menkveld-inspired summary tables, full-sample figures, and the compact `H1/H2` hypothesis-support table to `output/presentation/`.

### 3. p=3 robustness pass

- 09_run_h1_h2_p3_robustness.py

This reruns the split-sample workflow at `p = 3`, writes outputs to `output/robustness_p3/`, and builds a direct `p = 3` vs `p = 2` comparison.

### 3b. BIC lag-selection sweep

- 13_run_h1_h2_lag_selection.py

This fits the panel VARX for `p ∈ {1, 2, 3, 4}` in both regimes and across the three families; reports BIC, AIC, log-likelihood, spectral radius, and stability flags; writes `output/h1_h2/h1h2_lag_selection_bic.csv`.

### 3c. Menkveld-style descriptive extras

- 15_run_h1_h2_menkveld_extras.py

This builds the universe-wide daily dark-share series with pre/transition/post shading (`h1h2_aggregate_dark_share_daily.pdf`), the pre/post descriptive moments (`h1h2_descriptive_stats_pre_post.csv`), and the stacked cluster-diagnostics summary (`h1h2_cluster_diagnostics_summary.csv`).

### 3d. macro decomposition diagnostic

- 16_run_h1_h2_macro_decomposition.py

This writes the FOMC-only and CPI/PPI-only macro refits to `output/h1_h2/` with `macro_fomc_` and `macro_inflation_` prefixes.

### 4. export final LaTeX tables

- 11_run_h1_h2_latex_tables.py

This reads the final CSVs from the presentation layer and writes compact main-text and detailed appendix LaTeX snippets to `output/presentation/latex/`.

### 5. compile LaTeX snippets to PDFs

- 12_compile_h1_h2_latex_tables.py

This wraps each `.tex` snippet in a minimal standalone document, compiles with `tectonic`, and saves rendered previews to `output/presentation/latex/rendered/pdf/`.

## Output Structure

- `output/h1_h2/`, regime-specific coefficient tables, pre/post and `post_minus_pre` confidence-band files, draw diagnostics, run summaries, per-family cluster diagnostics (`h1h2_*_cluster_diagnostics.csv`), BIC/AIC lag sweep (`h1h2_lag_selection_bic.csv`), and the cached 1-σ(dVIX⁺) shock size (`vix_dvix_pos_sigma.txt`). Also holds the `macro_fomc_*` and `macro_inflation_*` decomposition outputs.
- `output/presentation/tables/`, compact sample, IRF, and hypothesis-support tables; Menkveld-style VARX tables; pre/post descriptive stats; stacked cluster-diagnostics summary.
- `output/presentation/figures/`, full-sample figures for `VIX`, `macro`, and `earnings`; the motivational universe-wide daily dark-share figure (`h1h2_aggregate_dark_share_daily.pdf` / `.png`).
- `output/presentation/latex/main_text/`, compact main-text table snippets.
- `output/presentation/latex/appendix/`, detailed IRF and VARX appendix snippets.
- `output/presentation/latex/rendered/pdf/`, compiled PDF previews.
- `output/robustness_p3/`, split-sample `p = 3` bands, draw diagnostics, run summary, key-horizon summary, and the direct `p = 3` vs `p = 2` comparison.

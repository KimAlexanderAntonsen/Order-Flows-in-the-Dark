# H1/H2 Implementation

This file is the code-companion for `05_Estimation_H1_H2`. It describes what each runner does, what objects the helper modules build, and what each saved output contains. 

## 1. What This Layer Adds On Top Of `04_VARX`

`04_VARX` defines the panel VARX, the IRF simulator, and the Monte Carlo inference logic. The H1/H2 layer reuses all of that and adds three things:

1. **A regime split.** The same cleaned sample is fit twice, once on the pre window and once on the post window.
2. **A comparison object.** The post IRF minus the pre IRF is constructed at every horizon, with confidence bands built from the same Monte Carlo draws that produced the regime-specific bands.
3. **Diagnostic and robustness runners** layered on the same fit logic: a macro decomposition (FOMC-only vs CPI/PPI-only), a within-pre stability check, a trend-controlled refit, the BIC/AIC lag sweep, and full reruns at $p = 3$ and $p = 4$.

Regime windows and benchmark settings live in `01_estimation_config.py`. The thin helper layer that calls the VARX engine is `02_estimation_h1_h2.py`.

## 2. Regime Split

The pre and post windows are defined by hard timestamp boundaries in `01_estimation_config.py`:

```text
PRE_PERIOD_START  = 2019-06-10 00:00:00
PRE_PERIOD_END    = 2019-09-30 23:59:59
POST_PERIOD_START = 2019-10-11 00:00:00
POST_PERIOD_END   = 2020-02-19 23:59:59
```

The exclusion window of the zero-commission war (`2019-10-01` to `2019-10-10`) is already removed in the shared data-loading layer `04_VARX/04_beta_varx_data.py`. The H1/H2 split therefore inherits a clean per-side sample and never re-applies the exclusion locally.

Each urgency family is fit independently in each regime, so a single run produces six fits (3 families × 2 regimes) plus their associated Monte Carlo bands.

## 3. Reusing The Benchmark VARX Per Family And Per Regime

`02_estimation_h1_h2.py` defines a `FamilySpec` for each urgency family that bundles:

- the iterator factory that yields stock-minute panels (`iter_vix_panel_pieces`, `iter_macro_panel_pieces`, `iter_earnings_panel_pieces`),
- the common exogenous block (`VIX_X_COLS`, `MACRO_X_COLS`, or empty for earnings),
- the firm-specific exogenous block (empty for VIX and macro, `EARNINGS_X_COLS` for earnings),
- the shock-path builder used at the IRF stage.

When `run_family_h1_h2` is called for one family, it:

1. Constructs a regime-windowed iterator factory by filtering each stock-minute piece on its `timestamp` column.
2. Calls `BaselinePanelVARX(p_lags=BENCHMARK_P_LAGS).fit_from_iterator(...)` to produce the coefficient table, the residual cross-products, and the within-transformed bread $(X'X)^{-1}$.
3. Calls `apply_two_way_clustered_covariance` (from `06_beta_varx_model.py`) with a freshly-constructed iterator so the meat matrices are accumulated on the same demeaned regressors used during the fit. The Cameron-Gelbach-Miller PSD repair is applied if the combiner produces a negative eigenvalue.
4. Calls the IRF simulator with the family's shock-path builder, draws `N_SIMULATION_DRAWS` Monte Carlo perturbations of the coefficient vector using the two-way-clustered covariance, and accepts only draws whose companion-form spectral radius is below unity.
5. Assembles the dark-share and lit-share paths from the simulated log dark and lit responses (same overflow-safe ratio form used by `04_VARX/10_beta_varx_irf.py`).

Step 4 is parallelized over draws via `joblib`. Threading is the default backend (`VARX_MC_BACKEND=threading`) because the numerically-prefixed module imports do not survive process pickling cleanly. The number of workers is controlled by `VARX_MC_N_JOBS` (default: all cores).

## 4. The Post-Minus-Pre Object

After both regime IRFs are built, the helper pairs the regime-specific Monte Carlo draws by index and computes

$$
\Delta \mathrm{IRF}_h = \mathrm{IRF}_h^{\text{post}} - \mathrm{IRF}_h^{\text{pre}}
$$

per draw, then quantiles across draws to produce the `post_minus_pre` bands. Because the pre and post fits use disjoint samples and independent random seeds (`BASE_SEED + 100*shock_index` for pre, `BASE_SEED + 1000 + 100*shock_index` for post), the two coefficient draws are independent under the cluster covariance, so the difference's bands are simulated jointly from independent regime draws.

The dark-share path is constructed inside each regime first, using that regime's own baseline dark share $\bar{S}_{\text{pre}} = \bar D_{\text{pre}} / (\bar D_{\text{pre}} + \bar L_{\text{pre}})$ for the pre fit and $\bar{S}_{\text{post}} = \bar D_{\text{post}} / (\bar D_{\text{post}} + \bar L_{\text{post}})$ for the post fit (the two baselines differ because the regime volume means differ). What is reported in `dark_share_change_bps_post_minus_pre` is therefore

$$
\big(\mathrm{DarkShare}_h^{\text{post}} - \bar{S}_{\text{post}}\big) - \big(\mathrm{DarkShare}_h^{\text{pre}} - \bar{S}_{\text{pre}}\big),
$$

the difference between two IRF-induced deviations, each measured relative to its own regime's baseline. The regime-level shift in average dark share has already been absorbed by the per-regime baseline subtraction, so the column reads directly as the regime-on-regime change in the IRF response itself, in basis points.

## 5. Saved Outputs Per Family

For each family, `03_run_h1_h2_estimation.py` writes the following to `output/h1_h2/`:

- `h1h2_<family>_pre_coefficients.csv`, `h1h2_<family>_post_coefficients.csv`, fitted coefficient tables, one regressor per row, one endogenous variable per column.
- `h1h2_<family>_<shock>_pre_bands.csv`, `h1h2_<family>_<shock>_post_bands.csv`, IRF point estimates and confidence bands per horizon. VIX has two shock files (`dVIX_pos_inv`, `dVIX_neg_inv`); macro and earnings have one each (`macro_event_path`, `earnings_event_path`).
- `h1h2_<family>_<shock>_post_minus_pre_bands.csv`, the regime difference and its bands at every horizon.
- `h1h2_<family>_<shock>_<regime>_draw_diagnostics.csv`, per-draw spectral radius and acceptance flag, used to audit the Monte Carlo loop.
- `h1h2_<family>_cluster_diagnostics.csv`, the two-way-cluster diagnostics for that family, including the minimum eigenvalue of the unrepaired combiner and a PSD-repair flag.

Across families, the runner also writes `h1h2_run_summary.csv` (one row per family + shock with sample bounds, observation counts, and entity counts). The cross-family stacking into `h1h2_cluster_diagnostics_summary.csv` is done later by `15_run_h1_h2_menkveld_extras.py` and is written to `output/presentation/tables/`, not to `output/h1_h2/` (see §10).

## 6. VIX Shock Calibration

VIX-family IRFs are reported at a one-standard-deviation shock to the positive-innovation series $\Delta\mathrm{VIX}_t^{*+}$ (column `dVIX_pos_inv` in the code). The standard deviation is estimated once on the full analysis sample (both regimes pooled), cached in `output/h1_h2/vix_dvix_pos_sigma.txt`, and reused for both regime fits. The negative-innovation IRF uses the same shock magnitude, so the two sides are reported on a common scale. Both regimes are shocked with the same value, so any difference in the reported IRF comes from the coefficient block.

## 7. Macro Decomposition

`16_run_h1_h2_macro_decomposition.py` re-fits the H1/H2 workflow with the macro exogenous block split into two disjoint subsets:

- `macro_fomc`: the 6 FOMC rate decisions (14:00 ET, regular session).
- `macro_inflation`: the 16 CPI/PPI releases (08:30 ET, pre-market).

Both subsets reuse the same VARX engine, the same iterator factories, and the same pre/post windows. Only the exogenous block changes, so the contrast between the two decomposition IRFs comes from the events themselves rather than from sampling or specification differences.

Outputs use the prefixes `macro_fomc_` and `macro_inflation_` and otherwise mirror the file names listed in §5. A summary CSV `h1h2_macro_decomposition_summary.csv` is also written.

## 8. Lag Selection

`13_run_h1_h2_lag_selection.py` fits the panel VARX at $p \in \{1, 2, 3, 4\}$ for every family and every regime, twelve cells per lag length, forty-eight fits in total. For each fit it reports `nobs`, `n_entities`, `k_regressors`, the multivariate log-likelihood, AIC, BIC, the companion-form spectral radius, and a stability flag. The output `h1h2_lag_selection_bic.csv` flags the BIC-minimizing and AIC-minimizing lag per (family, regime) cell.

## 9. Robustness Runners

Two full reruns of the H1/H2 workflow exist at non-benchmark lag lengths:

- `09_run_h1_h2_p3_robustness.py` reruns the entire baseline workflow at $p = 3$. Outputs go to `output/robustness_p3/` with `h1h2_p3_` filename prefixes.
- `22_run_h1_h2_p4_robustness.py` runs the same workflow at $p = 4$. Outputs go to `output/robustness_p4/` with `h1h2_p4_` prefixes.

Both are full reruns: they re-fit the VARX, redo the Monte Carlo, and write the same per-family band, draw-diagnostic, and cluster-diagnostic files. A direct comparison table is produced alongside each pass, lining up the $p = 2$ benchmark IRFs against the higher-lag version at the same horizons.

Two narrower robustness passes also run on top of the benchmark fits without changing the lag length:

- `18_run_h1_h2_pre_stability.py` splits the pre window at `2019-08-07` (the same date used by the H3 placebo) into `pre_a` and `pre_b` sub-windows, re-fits each family separately on each half, and computes a within-pre drift IRF `IRF(pre_b) − IRF(pre_a)` with simulation-based bands. This drift IRF is the placebo analogue of the H2 difference. Outputs go to `output/pre_stability/`.

- `20_run_h1_h2_trend_robustness.py` injects a regime-specific linear `day_index` covariate into the exogenous block and re-runs `run_family_h1_h2` via the `family_spec_override` hook. The IRF is then constructed with the trend partialled out. Outputs go to `output/robustness_trend/`.

Each of these isolates one alternative explanation: pre-existing within-window drift, or slow regime trend that the discrete window split would otherwise absorb into the regime dummy. Neither changes the structural form of the VARX.

## 10. Menkveld-Style Descriptive Extras

`15_run_h1_h2_menkveld_extras.py` writes three descriptive objects that are not IRFs but feed the chapter's framing tables and figures:

- `h1h2_aggregate_dark_share_daily.pdf` / `.png`, the universe-wide daily dark share with pre / transition / post shading. Built from the same endogenous minute panel that feeds the VARX so the daily series and the estimation sample line up exactly.
- `h1h2_descriptive_stats_pre_post.csv`, per-regime average dark share, lit share, log-volume moments, and realized-variance moments.
- `h1h2_cluster_diagnostics_summary.csv`, the per-family cluster diagnostics stacked into a single appendix table.

## 11. Presentation And LaTeX Layer

`06_run_h1_h2_presentation.py` consumes the saved estimation outputs and produces:

- `output/presentation/tables/`, compact main-text tables (per-family IRF summary, H1/H2 verdict table, sample summary) plus Menkveld-style VARX coefficient tables.
- `output/presentation/figures/`, full-sample IRF figures for each family with pre, post, and post-minus-pre panels.

`11_run_h1_h2_latex_tables.py` then writes LaTeX snippets to `output/presentation/latex/main_text/` and `output/presentation/latex/appendix/`. `12_compile_h1_h2_latex_tables.py` wraps each snippet in a standalone document and compiles it with `tectonic` to `output/presentation/latex/rendered/pdf/` for visual proofing.

## 12. Recommended Run Order

The runners are numbered so a left-to-right execution produces every dependency in order. The recommended sequence in `00_README.md` is the canonical one: estimation -> presentation -> robustness -> LaTeX export. The pre-stability, trend, and lag-selection runners can run independently after the estimation step; the macro decomposition runner is independent of all of them.

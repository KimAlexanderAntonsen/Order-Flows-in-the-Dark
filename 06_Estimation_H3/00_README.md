# 06_Estimation_H3

This folder is the `H3` work layer. `H3` asks whether the post-October 2019 change in urgency-to-venue responses is concentrated in retail-exposed stocks relative to comparable less-retail-exposed stocks. It is the cross-sectional retail extension of the market-wide `H1/H2` results: it asks whether the regime shift is concentrated in the retail-treated group rather than in a credible control group.

This folder builds on:

- `04_VARX`, which contains the benchmark `VARX` engine,
- `05_Estimation_H1_H2`, which contains the market-wide pre/post estimation layer.

## Methodology

**Regime split** (inherited from earlier stages):

- pre period: `2019-06-10` to `2019-09-30`
- post period: `2019-10-11` to `2020-02-19`
- The exclusion window `2019-10-01` to `2019-10-10` is excluded throughout.

**Benchmark H3 estimand.** For urgency family $u$ and horizon $h$:

$$
\mathrm{H3}_{u,h} = \big[\mathrm{IRF}^{T,\text{post}}_{u,h} - \mathrm{IRF}^{T,\text{pre}}_{u,h}\big] - \big[\mathrm{IRF}^{C,\text{post}}_{u,h} - \mathrm{IRF}^{C,\text{pre}}_{u,h}\big],
$$

where $T$ denotes the retail-treated group and $C$ the matched-control group. Equivalently, $\mathrm{H3}_{u,h} = \Delta\mathrm{IRF}^{T}_{u,h} - \Delta\mathrm{IRF}^{C}_{u,h}$ with $\Delta\mathrm{IRF}^{g}_{u,h} = \mathrm{IRF}^{g,\text{post}}_{u,h} - \mathrm{IRF}^{g,\text{pre}}_{u,h}$, so the H3 object is the H1/H2 post-minus-pre IRF computed inside group $g$, then differenced across groups. A negative $\mathrm{H3}$ value on the dark-share response means the post-October dark-share response shifted more negatively in the treated group than in the matched control; under the two-venue split, that is equivalent to a more positive shift in the lit-share response in the treated group.

**Robustness battery.** Mirroring the H1/H2 battery in thesis chapter 4 §6, H3 runs four robustness checks plus a final LaTeX appendix builder. The four checks divide into a lag block (specification) and an identification block (parallel trends and slow drift):

1. **Longer-lag refits** (files 10-11 for $p = 3$; file 20 for $p = 4$): re-runs the full benchmark workflow at $p = 3$ and $p = 4$. BIC prefers $p = 4$ in every regime-by-family cell of the H1/H2 lag selection, so the $p = 4$ pass is needed to show the benchmark conclusions do not hinge on the Menkveld-anchored choice of $p = 2$.
2. **Trend-controlled benchmark** (files 18-19): reruns the benchmark with a group-and-regime-specific linear `day_index` added as an exogenous covariate so slow-moving drift in dark share is absorbed before the impulse response is built.
3. **Within-pre IRF stability placebo** (files 16-17): splits the pre period in half at `2019-08-07`, refits the VARX on each pre half for both groups, and checks whether the treated-vs-control IRF drift is close to zero at the key horizons. This is a placebo on the actual object the benchmark estimates (IRFs, not levels), so it is the most direct identification check available.
4. **Alternative control** (also files 10-11): re-runs the benchmark replacing the matched control with the least-retail reference group.

After all four passes run, file 21 assembles the H3 robustness point-estimate matrix LaTeX table (for thesis appendix `app:h3-robustness-matrix`), reporting every benchmark rejection alongside its point estimate under each of the four robustness checks.

The Step 1 pretrend diagnostics rejected parallel trends on `dark_share` between treated and matched-control groups at the daily level (linear pre-trend slope $p = 0.037$, placebo DiD $p = 0.008$). The benchmark flags this in text but does not partial it out; the trend-controlled and within-pre passes above are what stress-test the resulting reading.

**Known limitation**: Independence of treated and control coefficient draws. The benchmark and all four robustness passes build confidence bands by drawing $\beta_T^{(d)}$ and $\beta_C^{(d)}$ independently from $N(\hat\beta_T, \hat V_T)$ and $N(\hat\beta_C, \hat V_C)$ and forming the difference in difference IRF per draw. This assumes the two estimators are independent, a conservative assumption: in reality the treated and matched-control panels overlap in time, so $\mathrm{Cov}(\hat\beta_T, \hat\beta_C)$ is positive under any shared market-level shock, and correcting for it would generally narrow the bands. The correct fix is either (a) a stacked-panel joint estimator of $(\beta_T, \beta_C)$ with cluster-robust covariance on time, or (b) an explicit estimate of $\mathrm{Cov}(\hat\beta_T, \hat\beta_C)$ from shared time clusters plugged into the joint draw. A shared-seed shortcut is not a valid substitute because it forces the two draws into perfect lockstep under the common standard Normal, which narrows bands by assumption rather than by estimation. This is a known conservative bias; acting on it requires a medium refactor and is left for future work.

## File Map

The folder contains six stages: (1) pre-period group diagnostics, (2) benchmark estimation, (3) presentation, (4) targeted robustness and integrated interpretation, (5) identification-focused robustness, (6) LaTeX export, PDF compilation, and the robustness-matrix appendix table.

Entry-point and stage markdown files:

- `00_README.md` (this file), entry-point markdown; explains what H3 is doing, how the stages fit together, and what each file does
- `01_Step_1_Pretrend_And_Comparability.md`, detailed note for Step 1; explains why group comparability and placebo checks come before estimation
- `04_H3_Pretrend_Workflow.md`, review for files `01`-`04`; what the pretrend and comparability diagnostics imply for the benchmark design
- `07_H3_Estimation_Review.md`, review for files `05`-`07`; summarizes the benchmark H3 findings before presentation and robustness
- `12_H3_Interpretation.md`, interpretation for files `08`-`12`; how to read the benchmark results, what survives robustness, and how the findings relate to Menkveld and the H3 literature note

Shared config:

- `01_h3_config.py`, group files, sample windows, benchmark settings, and output folders in one place

Step 1: pre-period group diagnostics:

- `02_h3_pretrend_diagnostics.py`, helper module; pre-period stock-day panel, summary tables, trend tests, placebo DiD checks, and group/gap plots
- `03_run_h3_pretrend_diagnostics.py`, runner for Step 1
- `04_h3_group_cells_and_scatter.py`, generates the retail-score histogram, the retail-score vs trading-volume and vs market-cap scatters, and the level-analogue group-cells balance table used in thesis chapter 5. Writes figures and LaTeX snippets directly into `07_Thesis/latex/figures/` and `07_Thesis/latex/tables/`, plus row-level CSVs under `output/`

Step 2: benchmark H3 estimation:

- `05_h3_estimation.py`, helper module; reuses the finished H1/H2 and VARX layers and constructs the difference in difference IRFs
- `06_run_h3_estimation.py`, runner; saves group-specific pre/post objects and the final treated-minus-control post-minus-pre bands

Step 3: presentation:

- `08_h3_presentation.py`, figure and table helper; same Menkveld-style visual language as `H1/H2`
- `09_run_h3_presentation.py`, runner for the benchmark presentation outputs

Step 4: targeted robustness ($p = 3$, $p = 4$, alternative control):

- `10_h3_robustness.py`, runs the $p = 3$ pass and the alternative-control pass using the least-retail reference group; saves benchmark-versus-robustness comparison tables. Uses `500` simulation draws for practicality; the benchmark layer still uses `10,000`
- `11_run_h3_robustness.py`, runner for the $p = 3$ and alternative-control block
- `20_run_h3_p4_robustness.py`, runner for the $p = 4$ pass; BIC prefers $p = 4$ in every regime-by-family cell of the H1/H2 lag selection, so this pass is what completes the H3 lag-robustness battery (matching the H1/H2 chapter 4 §6 specification). Uses `500` draws and writes to `output/robustness_p4/` with `h3_p4_` prefixes

Step 5: identification-focused robustness (parallel trends and slow drift):

- `16_h3_pre_stability.py`, within-pre IRF stability placebo: splits the pre period at `2019-08-07`, refits the VARX on each pre half for both groups, builds the per-group drift IRF and the placebo difference in difference `drift_treated - drift_control`
- `17_run_h3_pre_stability.py`, runner; saves drift and placebo bands per family/shock plus a compact run summary and key-horizon summary. Uses `500` draws
- `18_h3_trend_robustness.py`, trend-controlled robustness pass: injects a group-and-regime-specific linear `day_index` covariate into the VARX exogenous block so slow-moving drift in dark share is absorbed before the impulse response is constructed. Reuses the benchmark plumbing through a `family_spec_override` hook on `run_family_h3`; shock paths auto-fill `day_index` with zero, which is exactly the trend-control assumption
- `19_run_h3_trend_robustness.py`, runner; saves trend-adjusted group-and-difference in difference bands plus a benchmark-versus-trend comparison table. Uses `500` draws

Step 6: LaTeX export, PDF compilation, and the robustness-matrix appendix table:

- `13_h3_latex_tables.py`, turns the key H3 CSV outputs into LaTeX tables for the main text and appendix; also builds the adapted Menkveld-style H3 Tables 1-3 (average venue shares, variable descriptions, summary statistics)
- `14_run_h3_latex_tables.py`, runner for the LaTeX export layer
- `15_compile_h3_latex_tables.py`, compiles the saved snippets into standalone PDFs with `tectonic`; the widest appendix tables are compiled in landscape format
- `21_h3_robustness_matrix.py`, builds the H3 robustness point-estimate matrix LaTeX table referenced by thesis appendix `app:h3-robustness-matrix`: for every benchmark H3 rejection, it reports the point estimate at the same horizon under each of the four robustness checks ($p = 3$, $p = 4$, trend, within-pre placebo). Must run after Steps 2, 4, and 5 so all inputs exist

## Recommended Workflow

1. read `00_README.md` (this file)
2. read `01_Step_1_Pretrend_And_Comparability.md`
3. run `03_run_h3_pretrend_diagnostics.py`
4. run `04_h3_group_cells_and_scatter.py` (writes chapter 5 figures and the group-cells table into `07_Thesis/`)
5. review `04_H3_Pretrend_Workflow.md`
6. run `06_run_h3_estimation.py`
7. review `07_H3_Estimation_Review.md`
8. run `09_run_h3_presentation.py`
9. run `11_run_h3_robustness.py` ($p = 3$ and alternative-control passes)
10. run `20_run_h3_p4_robustness.py` ($p = 4$ pass)
11. run `17_run_h3_pre_stability.py` (within-pre IRF placebo)
12. run `19_run_h3_trend_robustness.py` (trend-controlled pass)
13. read `12_H3_Interpretation.md`
14. run `21_h3_robustness_matrix.py` (assembles the appendix robustness matrix after all four robustness passes are saved)
15. run `14_run_h3_latex_tables.py`
16. run `15_compile_h3_latex_tables.py`

## Output Structure

### Pretrend diagnostics

- `output/pretrend/tables/`, `output/pretrend/figures/`
- `h3_pretrend_comparability_summary.csv`
- `h3_pretrend_trend_tests.csv`
- `h3_pretrend_placebo_did_summary.csv`
- `h3_pretrend_daily_group_series.png`
- `h3_pretrend_weekly_group_series.png`

### Benchmark H3 estimation

- `output/h3_estimation/`
- `h3_run_summary.csv`
- `h3_key_triple_difference_summary.csv`
- `h3_*_treated_minus_control_post_minus_pre_bands.csv`

### Benchmark presentation

- `output/presentation/tables/`, `output/presentation/figures/`, `output/presentation/latex/`
- `h3_table1_average_venue_shares.csv`
- `h3_table2_variable_descriptions.csv`
- `h3_table3_summary_statistics.csv`
- `h3_estimation_sample_summary.csv`
- `h3_benchmark_reading_summary.csv`
- `h3_key_triple_difference_summary.csv`
- `h3_earnings_triple_difference.pdf`, `h3_macro_triple_difference.pdf`, `h3_vix_triple_difference.pdf`
- `latex/main_text/*.tex`, `latex/appendix/*.tex`, `latex/rendered/pdf/*.pdf`

### Robustness

- `output/robustness_p3/`, `output/robustness_p4/`, `output/robustness_reference/`
- `h3_p3_run_summary.csv`, `h3_p3_key_triple_difference_summary.csv`, `h3_p3_vs_p2_comparison.csv`
- `h3_p4_run_summary.csv`, `h3_p4_key_triple_difference_summary.csv`, `h3_p4_vs_p2_comparison.csv`
- `h3_reference_run_summary.csv`, `h3_reference_key_triple_difference_summary.csv`, `h3_reference_vs_matched_comparison.csv`
- `output/presentation/tables/h3_robustness_dashboard.csv`
- `output/presentation/latex/appendix/h3_robustness_matrix.tex` (assembled by `21_h3_robustness_matrix.py` from the four robustness passes; feeds thesis appendix `app:h3-robustness-matrix`)

### Within-pre IRF stability (placebo on IRF drift)

- `output/pre_stability/`
- `h3_pre_stability_run_summary.csv`, `h3_pre_stability_key_summary.csv`
- `h3_pre_stability_<family>_<shock>_treated_drift_bands.csv`
- `h3_pre_stability_<family>_<shock>_control_drift_bands.csv`
- `h3_pre_stability_<family>_<shock>_placebo_bands.csv`

### Trend-controlled H3 robustness

- `output/robustness_trend/`
- `h3_trend_run_summary.csv`, `h3_trend_key_triple_difference_summary.csv`, `h3_trend_vs_benchmark_comparison.csv`
- `h3_trend_<family>_treated_minus_control_post_minus_pre_bands.csv`

# H3 Presentation, Robustness, and LaTeX Layers (Files 08-20)

## 1. Presentation (Files 08-09)

- `08_h3_presentation.py` reads the benchmark outputs from `output/h3_estimation/` and turns them into the figure and table set under `output/presentation/`. It builds the three difference in difference IRF figures (`h3_earnings_triple_difference.pdf`, `h3_macro_triple_difference.pdf`, `h3_vix_triple_difference.pdf`), the key-horizon and estimation-sample summary CSVs, and the adapted Menkveld-style H3 Tables 1-3 (average venue shares, variable descriptions, summary statistics). The plot style mirrors the H1/H2 presentation layer.
- `09_run_h3_presentation.py` is the runner; it ensures the presentation output folders exist, calls the helpers in `08_h3_presentation.py` in order, and prints a short save log.

## 2. Targeted Robustness (Files 10-11)

- `10_h3_robustness.py`, runs three targeted robustness passes on top of the benchmark and saves their outputs next to the benchmark for direct comparison:
- **`p = 3`**, refits the H3 difference in difference design with one extra lag. Outputs to `output/robustness_p3/`, with a `h3_p3_vs_p2_comparison.csv` written alongside.
- **`p = 4`**, refits with the BIC-preferred lag from the H1/H2 lag-selection grid (`05_Estimation_H1_H2/output/h1_h2/h1h2_lag_selection_bic.csv`). Outputs to `output/robustness_p4/`, with a `h3_p4_vs_p2_comparison.csv`.
- **Alternative control**, keeps `p = 2` but replaces the matched control with the least-retail reference group. Outputs to `output/robustness_reference/`, with a `h3_reference_vs_matched_comparison.csv`.

All three passes reuse `run_family_h3` from `05_h3_estimation.py`, so the underlying VARX estimator and band-construction logic are identical to the benchmark. Confidence bands use `5,000` simulation draws (the benchmark uses `10,000`). The module also writes a combined dashboard CSV at `output/presentation/tables/h3_robustness_dashboard.csv` that lines up the benchmark reading next to each robustness pass for the key horizons.

- `11_run_h3_robustness.py` is the runner for the `p=3` and least-retail-control passes; it calls those two helpers in `10_h3_robustness.py` and writes their run-summary, key-horizon, and comparison CSVs. The `p=4` pass has its own runner (file `20`, see Section 4).

## 3. LaTeX Export and PDF Compilation (Files 13-15)

- `13_h3_latex_tables.py` converts the saved H3 CSV outputs into `.tex` snippets. It writes a main-text bundle and an appendix bundle into `output/presentation/latex/main_text/` and `output/presentation/latex/appendix/`. The export covers the H3 key-horizon summary, the estimation-sample summary, the three robustness comparison tables, and the adapted H3 Tables 1-3.
- `14_run_h3_latex_tables.py` is the runner; it imports `13_h3_latex_tables.py` and writes the snippets to disk.
- `15_compile_h3_latex_tables.py` compiles the saved snippets to standalone PDFs with `tectonic`. It writes per-table wrapper `.tex` files into `output/presentation/latex/rendered/src/` and the compiled PDFs into `output/presentation/latex/rendered/pdf/`. A small allow-list (`LANDSCAPE_TABLES`) compiles the widest appendix tables in landscape; a margins map (`TABLE_MARGINS`) tightens margins on the summary-statistics table.

## 4. Identification-Focused Robustness (Files 16-20)

The targeted block in Section 2 stresses the lag choice and the control-group choice. The block below stresses identification: it adds an IRF-level placebo on the parallel-trends assumption, a trend-controlled rerun of the benchmark, and the BIC-preferred lag runner. All three passes use `5,000` simulation draws.

- `16_h3_pre_stability.py` implements the within-pre IRF stability placebo. For each urgency family it splits the pre period at `2019-08-07` (the same date used in the level-based placebo DiD in file `02`), refits the family-specific VARX separately on the first and second pre halves for both the treated and matched-control groups, and constructs the per-group drift IRF `drift_group = IRF(pre_B) - IRF(pre_A)` and the placebo difference in difference `placebo_drift = drift_treated - drift_control`. The point is to test parallel trends on the actual object the benchmark estimates (the IRF), not on dark-share levels.
- `17_run_h3_pre_stability.py` is the runner; it calls the helpers in `16_h3_pre_stability.py` and writes the per-family drift and placebo bands into `output/pre_stability/`, together with `h3_pre_stability_run_summary.csv` and a compact `h3_pre_stability_key_summary.csv` at the H3 key horizons.
- `18_h3_trend_robustness.py` reruns the benchmark H3 specification with a group-and-regime-specific linear `day_index` (calendar days since the pre-period start) added as an exogenous covariate to the VARX. This absorbs slow-moving drift in dark share before the impulse response is constructed. It reuses the benchmark plumbing via a `family_spec_override` hook on `run_family_h3`; shock paths auto-fill `day_index` with zero through the IRF engine's existing `_prepare_shock_path` logic, which is exactly the trend-control assumption (a unit shock does not perturb the day counter).
- `19_run_h3_trend_robustness.py` is the runner; it calls the helpers in `18_h3_trend_robustness.py` and writes the trend-adjusted group-and-difference in difference bands into `output/robustness_trend/`, together with `h3_trend_run_summary.csv`, `h3_trend_key_triple_difference_summary.csv`, and a `h3_trend_vs_benchmark_comparison.csv`.
- `20_run_h3_p4_robustness.py` is the standalone runner for the `p=4` pass described in Section 2. It calls `run_p4_robustness` and `save_p4_outputs` from `10_h3_robustness.py`, so the underlying VARX fit and band construction are identical to the other lag-robustness passes. The lag choice itself is motivated by the H1/H2 lag-selection table: BIC picks `p=4` in every regime-by-family cell of the grid (`05_Estimation_H1_H2/output/h1_h2/h1h2_lag_selection_bic.csv`).

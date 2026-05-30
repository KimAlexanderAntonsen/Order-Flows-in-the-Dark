# 04_VARX

This folder is the methodology backbone of the project. It contains the benchmark panel VARX, the impulse-response layer, and the simulation-based confidence-band layer that `05_Estimation_H1_H2` and `06_Estimation_H3` later reuse.

The upstream VIX, macro, and earnings input files live in `03_VARX_Data`.

## What This Folder Does

The engine runs in three sequential steps:

1. Step 1 fits the benchmark VARX on the cleaned constant firm universe and saves the baseline outputs in `output/`.
2. Step 2 turns that fitted VARX into Menkveld-style impulse responses for `VIX`, `macro`, and `earnings`, saving the IRF outputs in `output/step2/`.
3. Step 3 adds simulation-based `95%` confidence bands and draw diagnostics, saving the inference outputs in `output/step3/`.

## File Structure

Overview markdown:

- `00_README.md` (this file)

Step 1 files (`01` to `08`):

- `01_Step_1_Implementation_of_VARX.md` (descriptive file)
- `02_beta_varx_config.py`
- `03_beta_varx_utils.py`
- `04_beta_varx_data.py`
- `05_beta_varx_panel.py`
- `06_beta_varx_model.py`
- `07_run_step1_baseline.py`
- `08_run_step1_diagnostics.py`

Step 2 files (`09` to `11`):

- `09_Step_2_Impulse_Response_Implementation.md` (descriptive file)
- `10_beta_varx_irf.py`
- `11_run_step2_irf.py`

Step 3 files (`12` to `14`):

- `12_Step_3_Confidence_Bands.md` (descriptive file)
- `13_beta_varx_inference.py`
- `14_run_step3_inference.py`

## Recommended Order Of Execution

1. read `00_README.md` (this file)
2. run `07_run_step1_baseline.py`
3. optionally run `08_run_step1_diagnostics.py` for lag-selection and stability checks
4. run `11_run_step2_irf.py`
5. run `14_run_step3_inference.py`

## Inference: two-way Petersen-clustered covariance

The engine computes a two-way Petersen (2009) / Cameron-Gelbach-Miller (2011) cluster-robust parameter covariance that clusters both by stock and by minute timestamp. The combiner formula is `V_TWC = V_entity + V_time - V_HC0`, and a minimum-eigenvalue PSD repair is applied on the rare cells where the combiner produces a slightly negative eigenvalue. The cluster diagnostics for each estimated cell are saved alongside the coefficient outputs so the reader can audit whether the repair was ever needed (it is flagged in the per-family `*_cluster_diagnostics.csv` files written by `05_Estimation_H1_H2` and `06_Estimation_H3`).

This is the covariance that feeds the Monte Carlo confidence bands in Step 3, so the reported IRF bands are two-way-clustered intervals.

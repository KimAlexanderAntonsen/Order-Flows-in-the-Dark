# Step 3 Confidence Bands

Step 3 adds inference to the fitted VARX and IRF layers. The goal is to move from a point IRF to an IRF with 95% confidence bands, so that later hypothesis statements can be based on uncertainty as well as on point estimates.

## 1. Relation to Menkveld

Menkveld construct 95% confidence bounds for their impulse responses by simulation. The same basic idea is used here. Once the baseline VARX has been estimated, the parameter vector is treated as approximately multivariate normal with mean equal to the point estimate and covariance equal to the estimated parameter covariance matrix.

Step 3 adds an uncertainty layer around the model already implemented in Step 1 and Step 2.

The main similarities to Menkveld are therefore:

- the benchmark lag choice remains $p=2$,
- the impulse responses are simulated from the fitted VARX,
- the confidence bands are built from repeated parameter draws.

The Step 3 shock sizes are inherited verbatim from Step 2: a 0.01-point VIX innovation, the natural macro event dummies, and a 1% EPS-surprise earnings path. This matches Step 2's mechanics-validation calibration. The headline H1/H2 IRFs in the thesis use a different VIX shock (one standard deviation of `dVIX_pos_inv` on the analysis sample, $\approx 0.048$); the H1/H2 inference pass rebuilds the bands at that calibration. Bands at the Step 3 calibration are still useful internally because the linear-in-shock IRF mechanics let downstream code rescale.

The main differences are inherited from our data and earlier modeling choices:

- our endogenous system is a reduced adaptation based on log dark volume, log lit volume, and log realized variance rather than Menkveld's richer venue-share system,
- our confidence-band draws use the covariance matrix stored by the Step 1 beta estimator rather than Menkveld's exact double-clustered covariance estimator,
- the dark-share response is derived after simulation from the dark and lit volume paths rather than being estimated directly as a primitive market-share variable.

## 2. Analysis Sample and Timing

Step 3 uses the same cleaned analysis sample as the earlier beta steps. This means that the sample runs from June 10, 2019 to February 19, 2020, but excludes the window from October 1 to October 10, 2019.

That exclusion is important because the window is treated as the introduction window of the zero-commission war. In the beta build, it is excluded directly in the shared data-loading layer, so the fitted coefficients, the point IRFs, and the confidence bands are all based on the same sample definition.

## 3. The Simulation Idea

Let the fitted Step 1 coefficient matrix be denoted by $\hat\beta$ (same symbol as in the Step 1 estimator equation). Step 3 treats the vectorized parameter estimate as approximately distributed as

$$
\mathrm{vec}(\hat\beta) \sim N\left(\mathrm{vec}(\beta),\ \widehat{V}_{\hat\beta}\right).
$$

The confidence-band simulation then works as follows:

1. Draw a parameter vector from the multivariate normal distribution centered on the point estimate.
2. Reshape that draw back into a coefficient matrix (column-major, to match the Kronecker structure of the covariance).
3. Recompute the impulse response using the Step 2 propagation code.
4. Repeat $B = 10{,}000$ times.
5. Take the 2.5th and 97.5th percentiles across the simulated responses, independently at each horizon and for each reported series.

This gives 95% pointwise confidence bands around the point IRF. The run is parallelized across draws via `joblib` (configurable through the `VARX_MC_N_JOBS` and `VARX_MC_BACKEND` environment variables; default is all cores, threading backend).

## 4. What Covariance Matrix We Use

The baseline VARX is estimated as a multivariate regression with a common regressor matrix across equations. In that setting, the beta build uses the fitted parameter covariance matrix stored in Step 1 as the covariance object for the simulation draws.

$$
\mathrm{Var}\left(\mathrm{vec}(\hat\beta)\right) = \widehat{\Sigma}_u \otimes (X'X)^{-1},
$$

where $\widehat{\Sigma}_u$ is the estimated residual covariance matrix and $X'X$ is the regressor cross-product matrix from the fitted system. Before sampling, the covariance is symmetrized and, if its minimum eigenvalue is negative, lifted by that amount plus a tiny jitter ($10^{-12}$) so the matrix is positive semi-definite. This avoids `multivariate_normal` failing on numerical asymmetries without materially changing the draws. The simulation layer is then close to Menkveld's logic: the IRF bands are driven by repeated parameter draws from the fitted VARX rather than by ad hoc resampling.

This is also the most important methodological difference from Menkveld in Step 3. In the paper, the simulation draws are based on the estimated double-clustered covariance matrix of the VARX coefficients. In the beta build, we instead use the multivariate OLS-style covariance matrix implied by the fitted Step 1 system. The Step 1 model file does implement a two-way Petersen-style cluster covariance (`apply_two_way_clustered_covariance` in [06_beta_varx_model.py](06_beta_varx_model.py)), but the Step 3 runner does not apply it, so the bands reported here are based on the classical multivariate OLS covariance. The cluster-robust covariance is what the downstream H1/H2 and H3 estimation passes use.

## 5. Stability Restriction

Not every simulated parameter draw is economically usable. Some draws can imply an unstable autoregressive system. Since unstable draws would make the impulse responses explode mechanically, Step 3 checks the companion-matrix spectral radius for every draw and rejects draws with eigenvalues outside the unit circle.

In the final Step 3 run, all requested draws were accepted for all four shock designs. So the covariance-based simulations did not generate an instability problem in practice.

## 6. What Is Simulated

The economic objects are unchanged from Step 2:

- `VIX`: one-off 0.01-point innovation shocks in `dVIX_pos_inv` and `dVIX_neg_inv` over a 60-minute horizon, with `VIX_close` kept in the fitted model as a control. Two band tables, one per shock, with seeds `BASE_SEED` and `BASE_SEED + 1`.
- `Macro`: the event path from `pre_news_1min` through `post_news_4min` over a 60-minute horizon. Seed `BASE_SEED + 100`.
- `Earnings`: the event path implied by the 13 half-hour `post_ea_k` blocks, scaled to a 1% EPS surprise, over a 450-minute horizon (390-minute shock path plus 60-minute decay). Seed `BASE_SEED + 200`.

`BASE_SEED = 2017` and `N_SIMULATION_DRAWS = 10000`, both set at the top of `14_run_step3_inference.py`. Each family also re-fits the VARX at $p = 2$ on its own panel before sampling, so the bands are based on the same fitted system as the corresponding Step 2 point IRF.

The model still produces responses in log dark volume, log lit volume, and log realized variance. As in Step 2, a dark-share path is then derived from the simulated dark and lit volume paths using the volume-weighted baseline shares $\bar D, \bar L$, so the reported dark-share bands are expressed as a *level relative to the baseline dark share* (`dark_share_level`, `dark_share_change`, `dark_share_change_bps`). This is the same steady-state reference Menkveld use; the difference from Menkveld is in the system being modelled, they put venue shares directly into the endogenous block, we put log dark and log lit volume in and recover the share post-hoc.

## 7. Output

The Step 3 run writes to `04_VARX/output/step3/`:

- `step3_vix_irf_bands.csv`, `step3_macro_irf_bands.csv`, `step3_earnings_irf_bands.csv`, one band table per family.
- `step3_draw_diagnostics.csv`, per-draw spectral radius and acceptance flag for every accepted draw across all four shock designs.
- `step3_inference_run_summary.csv`, one row per spec/shock with draws requested, draws accepted, attempts, unstable rejections, acceptance rate, and the max and mean spectral radius across accepted draws.

The band tables carry, for each reported series:

- the point IRF (`<col>_point`),
- the lower 95% bound (`<col>_lower95`),
- the upper 95% bound (`<col>_upper95`),

where `<col>` ranges over the three endogenous variables (`log_dark_volume_t`, `log_lit_volume_t`, `log_total_realized_variance_t`), the derived dark-share series (`dark_share_level`, `dark_share_change`, `dark_share_change_bps`), and the cumulative versions of those five series (`cum_log_dark_volume_t`, ..., `cum_dark_share_change_bps`), matching the Step 2 post-processing.

Step 3 is the inference layer that the later empirical folders consume. Once these bands exist, the next step is to use them in the hypothesis analysis.

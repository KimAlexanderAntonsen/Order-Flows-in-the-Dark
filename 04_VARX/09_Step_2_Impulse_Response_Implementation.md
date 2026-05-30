# Step 2 Impulse Response Implementation

Step 2 takes the matrices fitted in Step 1 as fixed inputs and propagates a chosen exogenous shock path through them. Estimation belongs to Step 1; Step 2 is pure propagation, plus a derived dark-share path and two internal validation checks.

## 1. Lag Choice

Step 2 re-fits the Step 1 baseline at $p = 2$ for each urgency family, matching the Menkveld benchmark. The motivation is to have a first IRF implementation that mirrors Menkveld as closely as possible before later robustness work compares alternative lag lengths. The matrices $\Phi_1$, $\Phi_2$, $B$, $G$ from Step 1 are then frozen for the rest of the run; the IRF module stacks the two exogenous-coefficient blocks into a single matrix $\Psi = [\, B \mid G \,]$ aligned with $z_{j,t} = (c_t, f_{j,t})'$, and never re-estimates.

## 2. Two IRF Representations

Each shock path is simulated twice using two mathematically equivalent forms. Let $z_h$ denote the exogenous shock vector at horizon $h$ (in the same column order as the fitted exogenous block).

The direct recursion is

$$
y_h = \Phi_1 \, y_{h-1} + \Phi_2 \, y_{h-2} + \Psi \, z_h,
$$

starting from a zero pre-shock state. Companion form is

$$
s_h = F \, s_{h-1} + R \, z_h,
\qquad
s_h = \begin{bmatrix} y_h \\ y_{h-1} \end{bmatrix},
$$

where $F$ is built from $\Phi_1, \Phi_2$ and $R$ loads the shock into the top $y$ block of the state. Both paths are simulated minute by minute and then compared. The pre-shock state is zero because the Step 1 model is estimated after within-stock demeaning, so "before the shock" is each stock's own mean and there is no level for the IRF to pin down. Equivalently, the simulated $y_h$ is read throughout as a deviation from the stock-mean steady state.

## 3. Shock Design per Urgency Family

The three families differ because their upstream data already encode different event structures.

**VIX.** The fitted model keeps `VIX_close` as a level control, but the IRFs shock only the two innovation regressors `dVIX_pos_inv` and `dVIX_neg_inv`, each with a 0.01-point impulse at horizon 0 and zero elsewhere. Two separate IRFs are produced, one per shock. Horizon end: 60 minutes.

The 0.01-point size here is a small diagnostic impulse chosen for the mechanics checks in Section 6 (the recursion-vs-companion equivalence and the horizon-0 check are exact in the shock size, so the calibration only has to be numerically convenient). The reported H1/H2 IRFs in the thesis use a different calibration: a one-standard-deviation innovation in `dVIX_pos_inv` measured on the analysis sample. That $\sigma$ is computed and cached by `05_Estimation_H1_H2/02_estimation_h1_h2.py::get_vix_pos_innovation_sigma()` (full-sample value $\approx 0.048$; per-regime $\sigma$'s of about $0.070$ pre and $0.057$ post are reported in Chapter 3). Step 2 is therefore the implementation-validation pass, not the calibration that backs the headline numbers.

**Macro.** The upstream macro panel already expands each scheduled release into a six-minute event window, so the shock path uses the natural dummy structure (each active dummy is set to 1):

| Horizon | Active dummy      |
|--------:|-------------------|
| $-1$    | `pre_news_1min`   |
| $0$     | `post_news_0min`  |
| $1$     | `post_news_1min`  |
| $2$     | `post_news_2min`  |
| $3$     | `post_news_3min`  |
| $4$     | `post_news_4min`  |

Horizon end: 60 minutes (so the decay after the last event minute is also saved).

**Earnings.** The upstream earnings panel defines 13 half-hour blocks `post_ea_1, ..., post_ea_13`. Step 2 expands this into the minute grid used by the VARX, so each `post_ea_k` is held at the shock value for 30 consecutive minutes, then turns off as `post_ea_{k+1}` turns on. The shock path therefore spans 390 minutes (block 1 starts at horizon 0, block 13 ends at horizon 389). The shock size matches the Menkveld 1% EPS-surprise calibration. Horizon end is 450 minutes, so the saved earnings IRF covers the full 390-minute event path plus a 60-minute post-event decay window.

## 4. Dark-Share Path Derivation

Dark share is not in $y_t$, so it is derived after the simulation. Because Step 1 is estimated after within-stock demeaning, the IRF columns `log_dark_volume_t` and `log_lit_volume_t` are read as deviations from the stock-mean steady state, i.e. $\Delta \log D_h$ and $\Delta \log L_h$. A naive level reconstruction $D_h = \bar D \exp(\Delta \log D_h)$ overflows when a simulated log path becomes very large, so Step 2 instead works with the algebraically equivalent ratio form

$$
\text{DarkShare}_h
= \frac{1}{1 + (\bar L / \bar D)\exp(\Delta \log L_h - \Delta \log D_h)},
$$

and clips the exponent argument to $[-700, 700]$ to keep `exp()` finite without distorting the qualitative shape. From this we report the dark-share level, the change relative to the sample-mean baseline, and the change in basis points. The baseline volume means $\bar D, \bar L$ are computed by streaming the same panel iterators used during estimation, so the baseline reflects the actual estimation sample rather than a wider universe.

## 5. Cumulative Columns

Each IRF also reports cumulative versions of `log_dark_volume_t`, `log_lit_volume_t`, `log_total_realized_variance_t`, `dark_share_change`, and `dark_share_change_bps`. The minute-level columns are kept alongside, so a reader can switch between point-in-time and area-under-curve interpretation without re-running the simulation.

## 6. Validation Built into the Run

Every IRF goes through two automatic checks:

1. **Recursion vs companion form.** For each endogenous column, the largest absolute gap between the two implementations must be at most $10^{-10}$. The run raises if any spec fails.
2. **VIX horizon-0 check.** For the one-off VIX shocks, the response at horizon 0 must equal the corresponding column of $\Psi$ times the shock size, because with a zero pre-shock state and a single contemporaneous impulse the recursion collapses to exactly that.

The macro and earnings paths are not given a horizon-0 check because their event paths span multiple horizons; the recursion-vs-companion check is the appropriate test for them.

## 7. What the Step 2 Run Produced

The reproducible Step 2 run (`11_run_step2_irf.py`) writes the following to `04_VARX/output/step2/`:

- `step2_vix_coefficients_p2.csv`, `step2_macro_coefficients_p2.csv`, `step2_earnings_coefficients_p2.csv`, fitted coefficient tables at $p = 2$,
- `step2_vix_irf.csv`, `step2_macro_irf.csv`, `step2_earnings_irf.csv`, the IRFs themselves, with dark-share and cumulative columns appended,
- `step2_irf_validation.csv`, recursion-vs-companion gaps per spec,
- `step2_vix_h0_checks.csv`, VIX horizon-0 contemporaneous checks,
- `step2_irf_run_summary.csv`, one-row-per-spec summary with lag, shock design, saved rows, max abs diff, and pass flags.
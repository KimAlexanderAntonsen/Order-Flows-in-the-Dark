# Step 1 Implementation of VARX

This notebook documents the implemented baseline VARX layer. Step 1 establishes the benchmark panel VARX with predetermined urgency variables and the cleaned 2019 to 2020 analysis sample that is reused throughout the later H1, H2, and H3 work. At this stage the focus is the fitted system itself.

## 1. Analysis Sample Exclusion

All beta analysis steps exclude the window from October 1 to October 10, 2019. The reason is that this is the introduction window of the zero-commission war, so it is treated as an exclusion window.

This exclusion is implemented in the shared data-loading layer. So Step 1, Step 2, and Step 3 all use the same cleaned analysis sample.

## 2. The Baseline VARX We Implemented

For stock $j$ and minute $t$, the baseline model is the panel VARX of Section 4.1 of the thesis:

$$
y_{j,t} = \alpha_j + \Phi_1 \, y_{j,t-1} + \cdots + \Phi_p \, y_{j,t-p} + \Gamma \, z_{j,t} + \varepsilon_{j,t}.
$$

The meaning of each part:

- $y_{j,t}$ is the endogenous market vector for stock $j$ at minute $t$.
- $\alpha_j$ is a stock fixed effect.
- $\Phi_1, \dots, \Phi_p$ are the autoregressive coefficient matrices.
- $z_{j,t}$ stacks the exogenous urgency variables for the specification being estimated.
- $\Gamma$ collects the coefficients on the exogenous block.
- $\varepsilon_{j,t}$ is the remaining unexplained shock.

The implementation splits $z_{j,t}$ into a common and a firm-specific piece, because some urgency variables are market-wide and some are not:

$$
z_{j,t} =
\begin{bmatrix} c_t \\ f_{j,t} \end{bmatrix},
\qquad
\Gamma \, z_{j,t} = B \, c_t + G \, f_{j,t},
$$

where $c_t$ collects the variables that are the same for every stock at minute $t$ and $f_{j,t}$ collects the variables that vary across stocks. This split is purely about the merge logic ($c_t$ joins on timestamp only, $f_{j,t}$ joins on stock and timestamp); the model is the same as the thesis equation above.

This is the right starting point for our project because it keeps the Menkveld-style VARX structure, while also allowing us to distinguish between common urgency shocks and firm-specific earnings shocks from the beginning.

## 3. What We Put in $y_{j,t}$ and $z_{j,t}$

The benchmark endogenous vector matches Panel A of Table 3.1 (variables used in the VARX) in the thesis:

$$
y_{j,t} =
\begin{bmatrix}
\log V^{\mathrm{dark}}_{j,t} \\
\log V^{\mathrm{lit}}_{j,t} \\
\log \mathrm{RV}_{j,t}
\end{bmatrix}.
$$

So the baseline system models dark volume, lit volume, and per-minute realised variance jointly. Dark share itself is not in $y_{j,t}$; it is recovered ex post from the simulated IRF paths as $V^{\mathrm{dark}}/(V^{\mathrm{dark}} + V^{\mathrm{lit}})$.

The exogenous block depends on the specification. For the VIX run, all three variables are market-wide:

$$
c_t =
\begin{bmatrix}
\mathrm{VIX}_t \\
\Delta \mathrm{VIX}_t^{*+} \\
\Delta \mathrm{VIX}_t^{*-}
\end{bmatrix},
\qquad f_{j,t} = \varnothing.
$$

For the macroeconomic run, the six event-time dummies are also market-wide:

$$
c_t =
\begin{bmatrix}
\mathrm{PreNews1min}_t \\
\mathrm{PostNews0min}_t \\
\mathrm{PostNews1min}_t \\
\vdots \\
\mathrm{PostNews4min}_t
\end{bmatrix},
\qquad f_{j,t} = \varnothing.
$$

For the earnings run, the thirteen post-announcement half-hour-block regressors are firm-specific:

$$
c_t = \varnothing,
\qquad
f_{j,t} =
\begin{bmatrix}
\mathrm{PostEA}^{1}_{j,t} \\
\vdots \\
\mathrm{PostEA}^{13}_{j,t}
\end{bmatrix}.
$$

This split matters in practice:

- `VIX` and `macro` are merged by timestamp only, because they are the same for every stock at a given minute.
- `earnings` are merged by stock and timestamp, because only the announcing stock should receive that shock path.

## 4. How the Data Become a Stock-Minute Panel

Step 1 uses the constant-membership universe; see `01_Data_Pull/data_clean/sp500_constant_sample_exclusions.csv`. The raw market side comes from the minute-bar files in `01_Data_Pull/data_clean/minute_bars/`.

For each stock, the panel builder does four things:

1. It loads the stock's minute-bar file.
2. It converts the raw timestamps into naive New York minute-end timestamps.
3. It applies the relevant session filter.
4. It constructs the endogenous variables and then merges the appropriate urgency variables.

The session choice depends on the specification:

- `VIX`: regular session, `09:31` to `16:00`
- `Macro`: extended morning session, `08:31` to `16:00`
- `Earnings`: regular session, `09:31` to `16:00`

After that, each observation is a single stock-minute row. So the baseline panel is a true panel with one row for stock $j$ at minute $t$.

## 5. Timestamp Alignment Assumptions

One important part of the implementation is that the three urgency families do not arrive in exactly the same raw timestamp format. Step 1 therefore makes a deliberate alignment choice before estimation: all specifications are merged on a common naive New York minute-end timestamp.

For the market data, this means the raw Massive timestamps are converted from UTC and shifted into the minute-end convention used elsewhere in the project. A trade that belongs to the minute ending at 09:31 is therefore stored under the `09:31` timestamp.

For the macro variables we use the already-processed minute panel in `macro_news_minute_panel.csv`. This means the macro dummies are interpreted exactly as they were constructed upstream: a scheduled release is expanded into a short sequence of minute-level indicators and merged by the processed local timestamp.

For the earnings variables, the issue is slightly different. The earnings panel contains more than one timestamp column, and Step 1 deliberately uses `timestamp_legacy_varx`, this is the timestamp convention that already matches the minute-end alignment. 

In practice, Step 1 assumes that the processed macro and earnings files already encode the correct event-time mapping. This keeps the model consistent with the upstream event construction.

## 6. How the Lag Structure Enters the Model

The main beta path uses $p=2$. This matches the Menkveld benchmark choice and is therefore the lag structure carried forward into Step 2 and Step 3.

This means the regressor block for minute $t$ contains

$$
x_{j,t} =
\begin{bmatrix}
y_{j,t-1} \\
y_{j,t-2} \\
z_{j,t}
\end{bmatrix}.
$$

In the code, this is implemented by creating the lagged variables stock by stock. For example, the model matrix includes terms such as

- `Y_L1.log_dark_volume_t`
- `Y_L1.log_lit_volume_t`
- `Y_L2.log_dark_volume_t`
- `Y_L2.log_total_realized_variance_t`

and then appends the urgency variables of $z_{j,t}$ that belong to the specification being estimated.

This is why the first usable timestamp in the final sample is later than the raw market open. We lose the earliest rows because the model needs enough history to form lags, and in the `VIX` case we also need enough observations to construct the innovation series.

## 7. How We Handle Stock Fixed Effects

The baseline estimator uses a within transformation instead of creating a large dummy matrix for every stock. In other words, for each stock $j$ we subtract the stock-specific mean from both the dependent variables and the regressors.

For any variable $w_{j,t}$, this is

$$
\tilde w_{j,t} = w_{j,t} - \bar w_j.
$$

Applying this to the VARX gives a demeaned system of the form

$$
\tilde y_{j,t} = \Phi_1 \, \tilde y_{j,t-1} + \cdots + \Phi_p \, \tilde y_{j,t-p} + \Gamma \, \tilde z_{j,t} + \tilde \varepsilon_{j,t}.
$$

This is useful because it removes the stock fixed effects without filling the design matrix with hundreds of dummy variables. The economic interpretation stays the same: we still allow each stock to have its own average level, but the dynamic coefficients are estimated from within-stock variation over time.

## 8. How the Estimator Is Implemented in Code

After the panel is built, the baseline estimator solves a pooled least-squares problem. In matrix form, the coefficient estimate is

$$
\hat\beta = (X'X + \lambda I)^{-1} X'Y,
$$

where $\lambda$ is a very small ridge term used only for numerical stability.

The model loops through the stocks one by one, builds the contribution from each stock, and accumulates the cross-products $X'X$, $X'Y$, and $Y'Y$. 

The Step 1 result also stores the fitted residual covariance matrix and the corresponding parameter covariance approximation. That matters because Step 3 later uses those stored covariance objects to simulate confidence bands around the impulse responses.

## 9. Step 1 output

The final Step 1 audit and full-sample baseline run are saved in `04_VARX/output/`.

The exact first usable timestamp differs across the three specifications because the lag construction and the urgency-variable construction trim the sample differently. These are saved in the Step 1 summary.
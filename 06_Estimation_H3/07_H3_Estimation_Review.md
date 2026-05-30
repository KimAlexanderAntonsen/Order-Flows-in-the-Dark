# H3 Benchmark Estimation Review (Files 05-07)

This markdown reviews the second implemented block in `06_Estimation_H3`, namely files `05` to `07`. This block takes the benchmark treated and matched-control groups as given and estimates the retail-treatment extension of the earlier `VARX` framework.

## 1. Files 05-07

- `05_h3_estimation.py` reuses the finished beta and H1/H2 layers and estimates the benchmark H3 objects for the treated and matched-control groups.
- `06_run_h3_estimation.py` runs that benchmark and saves the group-specific pre/post outputs as well as the final treated-minus-control post-minus-pre bands.
- This markdown explains what those outputs mean for the research question.

## 2. Benchmark H3 Object

The benchmark H3 estimand is the difference-in-differences in impulse responses:

$$
H3_u(h) = [IRF^{post}_{T,u}(h) - IRF^{pre}_{T,u}(h)] - [IRF^{post}_{C,u}(h) - IRF^{pre}_{C,u}(h)].
$$

Here `T` is the treated group and `C` is the matched control group. A negative H3 dark-share response means that the post-October dark-share response became more negative in the treated group than in the matched control group. In our two-venue setting, that is equivalent to a more positive lit-share response in the treated group.

## 3. Relation To H1, H2, and The Research Question

`H1/H2` already told us what happened in the market as a whole. `H3` asks whether that same regime shift is concentrated in the retail-exposed part of the market.

So the H3 benchmark should be read as a retail-channel extension of the earlier market-wide results. This is why the benchmark keeps the same Menkveld-style lag choice `p = 2`, the same exogenous urgency design, and the same simulation-based confidence-band logic.

## 4. Sample Used In The Benchmark H3 Run

The benchmark H3 run uses:

- treated group: `70` stocks
- matched control group: `70` stocks
- pre period: `2019-06-10` to `2019-09-30`
- post period: `2019-10-11` to `2020-02-19`
- benchmark lag choice: `p = 2`
- simulation draws: `10,000`

The run summary is saved in `h3_run_summary.csv`, and the compact treatment-effect table is saved in `h3_key_triple_difference_summary.csv`.

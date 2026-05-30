# Step 1: Pre-Trend and Comparability Diagnostics

This file is the detailed companion to `00_README.md`. The `00` markdown explains the overall logic of `H3`. This markdown zooms in on the first implemented task: checking whether the benchmark treated and matched-control groups are credible enough for a later difference-in-differences style extension of the `VARX` analysis.

The goal is to decide whether the benchmark treatment-control design is strong enough to deserve a formal `H3` estimation step.

## 1. Why Step 1 Comes Before Estimation

`H3` asks whether the post-October 2019 change in urgency responses is stronger in retail-exposed stocks than in comparable less-retail-exposed stocks. 

Before estimating any treatment-control `VARX`, we therefore need to check three things:

1. whether the matched control is materially closer to the treated group than the extreme least-retail reference group,
2. whether the treated and matched-control groups move broadly in parallel in the pre period,
3. whether a fake break inside the pre period produces large placebo treatment effects.

## 2. Relation To The Research Question and Literature

The research question is not simply whether venue responses changed after October 2019. `H1/H2` already address that market-wide question. `H3` asks whether the regime shift is most visible in the retail-exposed part of the market.

If the treated and control groups are poorly matched, then a later `H3` result could reflect pre-existing differences rather than a retail-specific regime shift.

## 3. Sample and Group Structure Used In Step 1

Step 1 uses only the pre period:

- `2019-06-10` to `2019-09-30`

The exclusion window from `2019-10-01` to `2019-10-10` remains excluded, exactly as in the rest of the project. The matched control group is the main control and the least-retail reference group is a secondary comparison that helps show why the matched control is the more credible benchmark.

## 4. What Step 1 Measures

The diagnostics focus on the outcomes that matter most for the eventual `H3` interpretation:

- dark share,
- lit share,
- log total volume,
- log total realized variance.

These outcomes are built from the same minute-bar data and timestamp conventions used in the earlier `VARX` layers. For interpretation, the most important two are dark share and lit share, since the later `H3` question is ultimately about whether retail-exposed stocks shift more strongly away from dark trading and toward lit trading after the October 2019 break.

## 5. Step 1

The implemented Step 1 workflow is deliberately simple.

1. `01_h3_config.py` defines the benchmark groups, sample window, placebo split date, and output folders.
2. `02_h3_pretrend_diagnostics.py` loads the treated, matched-control, and least-retail group definitions and then builds a pre-period stock-day panel from the minute-bar data.
3. The same module computes:

   - sample summaries,
   - comparability tables,
   - group-average time series,
   - treated-minus-control gap series,
   - linear pre-trend tests,
   - placebo DiD checks using a fake break inside the pre period.

4. `03_run_h3_pretrend_diagnostics.py` executes that workflow and saves the tables and figures.
5. `04_h3_group_cells_and_scatter.py` builds the retail-score histogram, the retail-score-vs-volume and retail-score-vs-market-cap scatters, and the four-cell group-by-regime dark-share table (with the `group_cells.tex` and `group_balance.tex` LaTeX snippets) used in thesis chapter 5.
6. `04_H3_Pretrend_Workflow.md` is the code-companion for the block: it lists what each file in `01`-`04` computes and which outputs it writes. Substantive interpretation of whether the matched control is credible enough is deferred to thesis chapter 5.

## 6. A Strong Step 1 Outcome Looks Like

A strong Step 1 outcome does not require the groups to be identical. It requires the benchmark control to be credible.

That means:

- the matched control should be meaningfully closer to treated than the least-retail reference group,
- the treated-minus-matched gaps should be relatively stable in the pre period,
- linear pre-trend differences should be small,
- placebo treatment effects inside the pre period should be weak or absent.

If those conditions hold, then the later `H3` estimates can be interpreted as reasonably credible quasi-experimental evidence.

## 7. How Step 1 Connects To Step 2

Given that Step 1 is acceptable, Step 2 estimates the benchmark `H3` object as a difference-in-differences in impulse responses. For a given urgency family `u` and horizon `h`, the intended benchmark object is

$$
\mathrm{IRF}^{\mathrm{H3}}_{u}(h) = \big[\mathrm{IRF}^{\mathrm{post}}_{T,u}(h) - \mathrm{IRF}^{\mathrm{pre}}_{T,u}(h)\big] - \big[\mathrm{IRF}^{\mathrm{post}}_{C,u}(h) - \mathrm{IRF}^{\mathrm{pre}}_{C,u}(h)\big].
$$

Here, `T` denotes the treated group and `C` denotes the matched control group. Step 1 therefore exists to justify whether this comparison is meaningful before we estimate it.

In practical terms, Step 1 is the identification checkpoint for the later `H3` `VARX` extension.

# H1 and H2 Presentation Layer

This notebook documents the final presentation layer built on top of the benchmark `H1/H2` estimation outputs. The aim is to convert the saved pre/post IRFs and the `post_minus_pre` objects into tables and figures that are close to Menkveld et al. (2017).

## 1. Relation to Menkveld

Menkveld present three kinds of objects that are especially relevant for the replication effort:

- summary tables describing the sample and variables,
- VARX coefficient tables,
- and impulse-response figures with confidence information.

Our presentation layer follows that same structure. The main difference is that our reduced system produces dark-share and lit-share objects rather than Menkveld's richer set of venue categories. The design goal is therefore visual and empirical comparability.

## 2. What the Presentation Layer Uses

The presentation layer reads the saved outputs from `05_Estimation_H1_H2/output/h1_h2`. In particular, it uses:

- the regime-specific IRF band files,
- the regime-specific coefficient tables,
- and the benchmark run summary.

The helper file `05_h1_h2_presentation.py` then builds a dedicated presentation folder:

- `05_Estimation_H1_H2/output/presentation/tables`
- `05_Estimation_H1_H2/output/presentation/figures`

After those CSV tables have been created, the helper file `10_h1_h2_latex_tables.py` turns the selected outputs into `.tex` snippets in:

- `05_Estimation_H1_H2/output/presentation/latex/main_text`
- `05_Estimation_H1_H2/output/presentation/latex/appendix`
- `05_Estimation_H1_H2/output/presentation/latex/rendered/pdf`

This keeps the presentation outputs separate from the underlying estimation files and makes it easy to lift the final tables directly into a document and inspect rendered PDF previews locally.

## 3. Tables

The presentation layer creates four table families.

First, it creates a compact estimation-sample table from the benchmark run summary. This is the nearest analogue to a Menkveld-style sample table in the implemented setup.

Second, it creates a compact IRF summary table that stores the main pre, post, and `post_minus_pre` response objects by horizon.

Third, it creates a compact `H1/H2` hypothesis-support table. This table is designed as a reading aid. It summarizes, for each urgency family, whether the relevant key horizons have the predicted sign and whether the `95%` simulation bands exclude zero in that direction. It should be read as a transparent summary classification, not as a formal joint test of the entire impulse-response path.

Fourth, it builds Menkveld-style `VARX` tables from the saved inference output for each family and each regime. These tables are still reduced relative to Menkveld because the underlying system is reduced, but they are organized so that the lag blocks and the exogenous block are easy to read.

A final LaTeX export layer sits on top of those CSV tables. The compact sample table and the compact hypothesis-support table are saved as main-text snippets, while the detailed IRF table and the family-specific `VARX` tables are saved as appendix-style snippets.

## 4. Figures

The figure layer creates two figure types for each urgency family.

The first type plots the pre and post regime levels side by side. These are the closest analogue to Menkveld's level figures. Because the reduced system works through dark share and the lit-share complement, these figures use two rows rather than Menkveld's three-venue layout.

The second type plots the `post_minus_pre` response with confidence shading. This is the natural analogue to Menkveld's difference figures once the question is framed as a pre/post regime comparison.

At the presentation stage, the share objects are

$$
DarkShare_t = 100 \times \frac{D_t}{D_t + L_t},
$$

and

$$
LitShare_t = 100 - DarkShare_t.
$$

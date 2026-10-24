# v17 — Environmetrics submission

## Manuscript
- Retargeted to Environmetrics: 12pt double-spaced layout, 39 pp against the journal's 40-page
  limit, abstract 191 words, five keywords none of which appear in the title, running title 62
  characters, author-year references with a DOI on every entry.
- Empirical results compressed into one subsection; the detail it carried is unchanged in the
  supplement, which was checked value by value before the compression.
- Conclusions rewritten: the previous version still carried the pre-temporal-unit headline.

## Simulation
- `spatial_component` now exposes the two levers the single-specification objection turns on:
  the covariance family (`exponential`, `matern32`) and the range (`--spatial-range-km`). The
  exponential branch at 70 km reproduces the primary run bit-for-bit, verified on a 33-row smoke
  comparison across 40 shared columns.
- The 70 km range and the exponential family are now documented in the manuscript. They were
  previously set in code and in the run config only, which left the data-generating mechanism
  incompletely specified.

## Figure scripts
- `make_fig_audit.py` and `make_fig5.py` honoured no output-directory environment variable and
  wrote to a path fixed at import time, unlike their siblings. Both now read `CAMSCAST_FIGDIR`.
- All figures regenerated at 800 dpi.

## Covariance sensitivity (added)
- `agg_cov_sensitivity.py` compares the primary specification against three alternatives at the
  primary radius: exponential at 35 km, exponential at 140 km, and Matern 3/2 at 70 km. Each is a
  full 32-scenario run with 100 replicates. The analysis is post hoc and is labelled as such in
  both the manuscript and the supplement.
- Every conclusion holds in all four specifications. The failing corner is invariant across them
  by construction: spatial dependence is switched off in that cell, so the spatial component is
  identically zero and the covariance specification cannot act on it.

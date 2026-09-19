# serra_camscast

Reproduction package for

> **A data-adaptive audit of spatial transfer in environmental residual-correction models**

Everything here is produced by the code in `code/` and `simulation/`, from the data in
`data_processed/`, with no manual steps. `smoke_test.py` reproduces the integrity checks in under a
minute.

---

## 1. The question

A residual-correction model is fitted at monitored stations and applied at a station it has never
seen. Withholding that station is necessary to call the result spatial transfer. It is not
sufficient: information can still reach the target through nearby stations, through source
observations at the same timestamps, through a shared coarse model-grid cell, through coordinates
used as features, and through the quality-control rule that decides which observations are evaluated
at all.

This repository answers the question twice.

**Empirically**, on a 19-station PM10 network with CAMS EAC4 as the mechanistic field. Three
parameters inside the audit are normally set by convention rather than measured, and each one, moved
across the range a reasonable analyst would defend, flips the verdict on its own.

| Convention | Value A | Value B | Sign-test *q* at R = 0 | Cost of the stricter value |
|---|---|---|---|---|
| Temporal guard band | 3 days | 10 days | 0.008 → 0.251 | 11.6% of training rows |
| Block boundaries | calendar-equal | equal-count (quantile) | fold imbalance 124× → 1.01× | none |
| Flat-line threshold | K = 3 | K = 6 | 0.011 → 0.251 | removes 9.9% vs 1.2% of rows |

Under the measured value of all three — guard 10 d, quantile blocks, K = 6 — no cold-start transfer
gain is established at any radius.

**Under known ground truth**, in a 32-scenario stochastic experiment where the presence of a
transferable signal, of spatial, temporal and shared-grid dependence, and the network geometry are
fixed by design. This separates "the strict audit removed an artefact" from "the strict audit
suppressed a real but weak signal", which no analysis of a real network can do. See
[`simulation/README.md`](simulation/README.md).

The headline of that experiment is a trade-off, not a victory: the full audit holds false-positive
transfer verdicts near 2–3% on average across 25/50/100 km buffers — though one dependence corner
reaches 11–13%, for a reason the manuscript works through — and in exchange retains only about
35–37% detection of a genuine transferable signal for coordinate-free models and 22–28% for
coordinate-bearing ones. Both halves of that trade-off survive substitution of a gradient-boosted
tree for the transparent simulation corrector; the unaudited station holdout does not.

### Why each stricter value is the measured one, not the conservative one

**Guard band.** The residual autocorrelation function (`code/resid_acf.py`, `results/resid_acf.json`)
is +0.87 at 3 h, falls below 1/e by 24 h, then plateaus near +0.20 out to 7 days and reaches +0.08
only at 10 days. A 3-day guard leaves the plateau inside the training window. Widening to 10 days
costs 11.6% of the training rows (median 11,068 → 9,789 per fold), so the change in verdict is
leakage removal, not power loss.

**Block boundaries.** Four months of 2023 hold no observations, so calendar-equal blocks leave three
of the six folds nearly empty at every station — at station 369959 they hold 218 / 9 / 4 / 7 / 335 /
489 points. Across the 19 stations the smallest calendar fold holds 4 points and the largest 517, a
median largest-to-smallest ratio of 124. Equal-count blocks hold 170–190 points, a within-station
ratio of 1.01.

**Flat-line threshold.** 15.5% of observations repeat the previous value, and the record is
integer-valued with a median absolute step of 3 μg/m³, so short ties are expected from reporting
resolution alone. Runs of 2–10 steps sit below the network median (74.5 μg/m³) and are depleted in
values above 100 μg/m³, which is the integer-reporting signature; the two longest runs (51 and 22
steps) show no such depletion and are treated as faults. K = 6 removes 1.2% of rows. K = 3 removes
9.9%, and the removed points have median 60 μg/m³ against a network median of 74.5 with only 9.8%
above 100 against 28.5% — so K = 3 strips clean air, raises the retained mean CAMS error from 46.6 to
48.4 μg/m³, and manufactures the significance it then reports.

### What survives

The comparison that does not depend on the three conventions is the **paired** one: the same station,
the same folds, coordinate-free features versus coordinate-bearing features. Under the primary
specification (guard 10 d, quantile blocks, K = 6, buffer + time + cell controls):

| Buffer radius | coordinate-free better at | sign *p* | Wilcoxon *p* |
|---|---|---|---|
| 0 km | 11 / 19 | 0.648 | 0.738 |
| 25 km | 15 / 19 | 0.019 | 0.113 |
| 50 km | 14 / 19 | 0.064 | 0.096 |
| 100 km | 13 / 19 | 0.167 | 0.036 |

This is a paired difference test, not two independent one-sample verdicts compared against each other
— the error Gelman & Stern (2006) describe, which an earlier version of this analysis committed. The
four-radius family does not clear multiplicity correction, so it is reported as a direction, not a
finding.

---

## 2. Layout

```
code/                     29 analysis scripts for the empirical audit, no hardcoded paths
data_retrieval/           request parameters for rebuilding data_processed/ from the
                          source archives (specification only; no retrieval client ships)
simulation/               the ground-truth stochastic experiment and its Monte Carlo outputs
data_processed/           aligned three-hourly records (24 series), plus the two
                          quality-controlled copies the sensitivity needs (K=6, K=3)
results/                  29 JSON artefacts, per-point predictions, audit outputs
manuscript/               Springer Nature LaTeX sources, figures, figure scripts,
                          and the superseded EMS version under archive_ems/
requirements.txt          version ranges that will not disturb your environment
requirements-frozen.txt   the exact versions the reported analysis ran on
smoke_test.py             loads the data and re-checks the integrity invariants
LICENSE
```

## 3. Running the empirical audit

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py          # ~1 min; prints "OK"
```

`requirements.txt` gives version ranges wide enough not to disturb an environment you already have;
the smoke test has been run under pandas 2.3 and 3.0. To reproduce the published numbers on the exact
versions the analysis ran on, use `pip install -r requirements-frozen.txt` instead.

The scripts resolve their own paths relative to the repository root, so they can be run from
anywhere:

```bash
cd code

python qc_flatline.py         # the flat-line evidence and the QC'd records
python resid_acf.py           # residual autocorrelation -> guard band
python moran.py               # residual spatial autocorrelation -> buffer ladder

# the reported audit: guard 10 d, quantile blocks, K = 6
CAMSCAST_DATA=../data_processed/aligned_qc6 \
CAMSCAST_OUT=../results/audit_v7_qc6.json \
CAMSCAST_GUARD=10D CAMSCAST_BLOCKS=quantile python audit_v7.py

# the two convention variants
CAMSCAST_DATA=../data_processed/aligned_qc6 \
CAMSCAST_OUT=../results/audit_v7_qc6_guard3.json \
CAMSCAST_GUARD=3D CAMSCAST_BLOCKS=quantile python audit_v7.py

CAMSCAST_DATA=../data_processed/aligned_qc3 \
CAMSCAST_OUT=../results/audit_v7_qc3.json \
CAMSCAST_GUARD=10D CAMSCAST_BLOCKS=quantile python audit_v7.py

python number_audit.py        # every numeric literal in the LaTeX vs the artefacts
```

Full run time is roughly 6 hours on one core; `audit_v7.py` caches per-cell results into its output
JSON and resumes, so it can be interrupted.

## 4. Running the stochastic experiment

See [`simulation/README.md`](simulation/README.md). In short:

```bash
cd simulation
python serra_simulation_v2_tests.py
for R in 25 50 100; do
  OMP_NUM_THREADS=1 python serra_simulation_v2.py \
    --n-rep 100 --jobs 8 --radius-km $R --outdir results/out_R$R
done
```

## 5. Determinism

`HistGradientBoostingRegressor(random_state=0, early_stopping=False)` for the empirical audit; ridge
regression on a fixed RBF basis for the simulation. The bootstrap and permutation generators are
seeded (`default_rng(0)`; the simulation's master seed is `20260910`). Re-running any script
overwrites its artefact with byte-identical content.

## 6. Reported numbers

Every numeric literal in the manuscript and supplement is extracted with its sentence and matched
against the artefacts by `code/number_audit.py`. About 95% match an artefact value automatically; the
remaining two dozen are station identifiers, publication years, page numbers, sentinel codes and
fixed hyperparameters, listed with their context in `results/number_audit_unmatched.csv`.

Three further checks ship with the package because they are how the reported numbers were verified:
`code/xref_audit.py` rebuilds the float numbering from source and checks every hard-coded "Table S"
and "Figure" reference against the float it lands on; `code/doi_verify.py` resolves every
bibliography entry at CrossRef, falling back to DataCite for arXiv registrations; and
`code/ai_remover.py` flags machine-writing constructions in the prose.

## 7. Data provenance

CAMS **global** reanalysis (EAC4), three-hourly surface PM2.5 and PM10 on a 0.75° (~80 km) grid,
matched by nearest grid cell to hourly observations from public monitoring stations and aggregated to
three-hourly resolution. The PM10 audit uses 19 stations and 20,818 matched samples; the benchmark
arm adds five PM2.5 sites. `data_processed/` holds the aligned records; the raw CAMS downloads are
not redistributed here. `data_retrieval/RETRIEVAL.md` records the exact request parameters —
dataset, variables, grid, time grid, unit conversion and station-matching rule — for CAMS EAC4, the
CAMS operational forecast and MERRA-2, so the aligned records can be rebuilt from the providers.

## 8. Citation

Cite the manuscript. The archived version of this code carries its own tag in the repository history.

This repository supersedes `camscast-audit`, which held the version of the analysis submitted to
*Environmental Modelling & Software*. The empirical audit is unchanged; what is new here is the
ground-truth stochastic experiment and the Springer Nature manuscript built on it.

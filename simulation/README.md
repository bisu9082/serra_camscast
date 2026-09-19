# Ground-truth stochastic experiment

The empirical audit in this repository can show that a transfer verdict changes when a validation
parameter changes. It cannot show whether the strict verdict is correct, because the truth about the
observed network is unknown. This directory holds the experiment that supplies that truth.

## What it does

Residuals are generated with a known structure

    r_i(t) = beta * F_i(t) + S_i + T(t) + G_c(i)(t) + eps_i(t)
    y_i(t) = m_c(i)(t) + r_i(t)

where `F` is a transferable function of the mechanistic field and season, `S` a spatially correlated
station effect, `T` a temporally autocorrelated common component, `G` a shared-grid-cell component
and `eps` independent noise. Stations sharing a CAMS cell receive an identical predictor series, as
they do in the observed network.

Five binary factors give 32 scenarios: transferable signal (`beta` 0 or 1), spatial dependence,
temporal dependence, shared-grid dependence, and dense or sparse network geometry. The dense geometry
uses the 19 observed station coordinates and their 9 CAMS cells; the sparse geometry doubles
source-target distances about the centroid while holding cell membership fixed, so it varies geometry
without confounding the shared-grid factor.

Four validation designs are applied to every realization:

| design | what is withheld |
|---|---|
| `station` | the target station |
| `space` | + every source within R km |
| `space_time` | + the target's test block and a data-adaptive temporal guard |
| `full` | + every source sharing the target's CAMS cell |

The guard is re-estimated in each simulated source network as the first lag at which the median
absolute source-station residual autocorrelation is at most 0.10, capped at 14 days. It never sees
the target outcome.

A positive transfer verdict requires both a positive median station-level
`Delta = log(MAE_raw / MAE_transfer)` and a one-sided exact sign-test `p < 0.05`. Under `beta = 0`
the verdict rate is a false-positive rate; under `beta = 1` it is detection power.

## Running it

```bash
pip install numpy pandas scipy scikit-learn

python serra_simulation_v2_tests.py          # structural invariants, a few seconds

# the reported experiment: RBF-basis ridge corrector, 100 replicates, three radii
for R in 25 50 100; do
  OMP_NUM_THREADS=1 python serra_simulation_v2.py \
    --n-rep 100 --jobs 8 --radius-km $R --outdir results/out_R$R
done

# the learner sensitivity check: the empirical audit's gradient-boosted tree.
# Full ladder at the primary radius:
OMP_NUM_THREADS=1 python serra_simulation_v2.py \
  --n-rep 25 --jobs 8 --radius-km 50 --learner hgb --outdir results/out_R50_hgb

# full-audit rung only at the two sensitivity radii (--designs cuts the work to a quarter;
# the rows it does compute are bit-identical to the same rows of an unrestricted run):
for R in 25 100; do
  OMP_NUM_THREADS=1 python serra_simulation_v2.py \
    --n-rep 25 --jobs 8 --radius-km $R --learner hgb --designs full \
    --outdir results/out_R${R}_hgb
done

# aggregate the three radii and test the learners against each other directly:
python agg_learner.py            # writes learner_check_3radii.csv, learner_direct_tests.csv
```

`OMP_NUM_THREADS=1` matters. Without it the BLAS threads oversubscribe the process pool and the run
slows by roughly an order of magnitude. On two cores one ridge radius takes about 14 minutes; the
gradient-boosted check took 3 h 24 min for 25 replicates, because the tree refits once per target,
design, specification and replicate.

`--learner ridge` is the default and reproduces the reported numbers exactly; the default seed is
`20260910` and both estimators are deterministic.

## What is here

    serra_simulation_v2.py          the experiment
    serra_simulation_v2_tests.py    structural invariants (station count, cell count, geometry
                                    scaling, guard rule)
    serra_simulation_v2_config.json the pre-declared configuration
    agg_learner.py                  aggregates the ridge and tree runs across the three radii and
                                    tests each learner pair directly (Fisher exact + BH), rather
                                    than reading whether their separate intervals overlap
    aoa_knndm_sim.py                nearest-neighbour distance structure and unweighted
                                    dissimilarity index of the simulated ladder, the two
                                    diagnostics Supplementary Tables S27 and S28 report for the
                                    observed network.  No Monte Carlo needed; runs in a minute.
    results/
      simulation_results_R{25,50,100}.csv.gz    one row per scenario x replicate x spec x design
                                                (25,600 rows per radius)
      simulation_results_R50_hgb.csv.gz         the learner check, full ladder (6,400 rows)
      simulation_results_R{25,100}_hgb.csv.gz   the learner check, full-audit rung only
      simulation_summary_R*.csv                 the per-scenario aggregation
      simulation_config_R*.json                 the configuration each run actually used
      ladder_all_radii_reproduced.csv           ladder x radius summary quoted in the manuscript
      learner_check_R50.csv                     ridge vs tree at R = 50, all four rungs
      learner_check_3radii.csv                  ridge vs tree, full audit, all three radii
      learner_direct_tests.csv                  the twelve paired learner comparisons: difference,
                                                95% CI, Fisher exact p, Benjamini-Hochberg q
      aoa_knndm_sim.csv                         distance structure and AOA of the simulated ladder

## Reported rates

Aggregated over the 16 null and 16 signal scenarios; ridge cells are 1,600 realizations, the
gradient-boosting cells 400.

| R | spec | β=0 station | β=0 full | β=1 station | β=1 full |
|---|---|---|---|---|---|
| 25 | coordinate-free | 20.00% | 2.31% | 77.94% | 36.81% |
| 25 | coordinate-bearing | 29.50% | 2.94% | 79.56% | 27.81% |
| 50 | coordinate-free | 20.00% | 2.69% | 77.94% | 36.69% |
| 50 | coordinate-bearing | 29.50% | 2.44% | 79.56% | 25.69% |
| 100 | coordinate-free | 20.00% | 2.62% | 77.94% | 35.31% |
| 100 | coordinate-bearing | 29.50% | 2.06% | 79.56% | 22.50% |

Two qualifications ship with those numbers, because both are in the manuscript.

**The 2-3% is a marginal average and is not uniform.** In the corner with temporal dependence alone —
no spatial and no shared-grid dependence — the full audit still returns a positive verdict in 11-13%
of realizations at every radius. The cause is the inference unit rather than the guard: a purely
common temporal component moves all 19 stations together, so the station-level sign test is reading
dependent outcomes as independent ones. Supplementary Table S40 reports every corner.

**The full audit is learner-invariant; the unaudited rungs are not.** The learners are compared by
testing the difference between them, not by asking whether their separate intervals overlap — the
distinction the manuscript itself insists on. That matters here: the overlap reading calls all twelve
full-audit comparisons identical, and the direct test does not.

Specificity is learner-invariant at all three radii: under beta = 0 the full audit returns 1.00-1.75%
under the tree against 2.06-2.94% under ridge, and none of the six pairs differs (Fisher p from 0.061
to 0.440, every q > 0.35). Detection matches in eleven of twelve cells. The exception is
coordinate-bearing detection at R = 100 km — 28.75% under the tree against 22.50% under ridge,
+6.25 pp [+1.37, +11.14], p = 0.010 — which does not survive Benjamini-Hochberg across the family
(q = 0.12). Its direction is that the tree retains *more* detection, so it qualifies a secondary
claim rather than the headline: the ridge-based decline of coordinate-bearing detection as the buffer
widens (27.81% at 25 km to 22.50% at 100 km, p = 0.001) is not reproduced by the tree (30.75% to
28.75%, p = 0.59) on a quarter of the sample. The manuscript therefore reports that decline as a
property of the transparent corrector.

The unaudited rungs are a different matter. At R = 50 a plain station holdout makes the tree declare
transfer in 52.0% and 56.3% of null realizations against 20.0% and 29.5% for ridge: +32.0 pp
[+26.7, +37.3] and +26.8 pp [+21.4, +32.1], both p < 1e-22. The buffer alone leaves the gap open
(28.3% and 24.0% against 16.6% and 13.6%, p < 1e-6). Only after the temporal guard do the two
learners stop differing (p = 0.29 and p = 0.075). Supplementary Tables S41 and S43 have the full
comparison.

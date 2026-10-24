"""Does a coarser inference unit repair the anti-conservative corner?

The station-level verdict applies a one-sided exact sign test across the nineteen held-out
targets, which treats them as responding independently. The stochastic experiment shows
that assumption failing in one corner of the factorial: when the residual carries a
temporal component and neither spatial nor shared-grid dependence is present, the full
audit still declares transfer in about one realization in eight.

The network's own structure suggests a coarser unit. Stations sharing a mechanistic grid
cell carry an identical covariate series, so the nine occupied cells, not the nineteen
stations, look like the units that move independently. The empirical analysis reads its
verdict at that coarser unit. Whether that is a repair or only a reflex cannot be settled
on observed data, where the answer is unknown; it has to be put against a known answer.

This script does that. serra_simulation_v2.py now forms both verdicts from the same
per-target deltas of the same realization -- the station-level sign test and a cell-level
sign test over the within-cell median deltas -- so the two are paired and can be compared
directly rather than by inspecting two separate rates. The comparison here is McNemar's
exact test on the discordant pairs, which is the test that belongs with paired binary
outcomes.

Reads  <outdir>/R{25,50,100}/simulation_results.csv
Writes results/cell_unit_comparison.csv  and prints the three tables the manuscript uses.

Usage:  python agg_cell_unit.py
        CAMSCAST_CELLRUN=/path/to/runs python agg_cell_unit.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "simulation" else HERE
RUNDIR = os.environ.get("CAMSCAST_CELLRUN", os.path.join(HERE, "results", "cell_unit"))
# The run also carries a control that holds the number of units fixed: the same targets are
# partitioned at random into as many groups as there are occupied cells, and the identical
# group-level sign test is applied. It separates what the coarsening buys from what the
# geography buys, which a comparison against the station rule alone cannot do.
OUT = os.environ.get("CAMSCAST_OUT",
                     os.path.join(ROOT, "results", "cell_unit_comparison.csv"))
RADII = (25, 50, 100)
PATHWAYS = [("off", "off", "off", "none"), ("on", "off", "off", "spatial only"),
            ("off", "on", "off", "temporal only"), ("off", "off", "on", "grid only"),
            ("on", "on", "off", "spatial+temporal"), ("on", "off", "on", "spatial+grid"),
            ("off", "on", "on", "temporal+grid"), ("on", "on", "on", "all three")]


def load(R):
    p = os.path.join(RUNDIR, f"R{R}", "simulation_results.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing run: {p}")
    d = pd.read_csv(p)
    return d[d.design == "full"]


def mcnemar(a, b):
    """Exact McNemar on paired binary outcomes a (station) and b (cell).

    The random-grouping rates printed alongside are the mean over N_RND partitions per
    realization, which is the stable estimate of what an arbitrary coarsening of that size does;
    the paired test uses one unbiased draw, since McNemar needs a binary outcome per pair.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    p = float(stats.binomtest(b10, n, 0.5).pvalue) if n else float("nan")
    diff = 100 * (b.mean() - a.mean())
    se = 100 * np.sqrt(n) / len(a) if len(a) else float("nan")
    return diff, diff - 1.96 * se, diff + 1.96 * se, b01, b10, p


rows = []
print("=== full audit, station-level verdict against cell-level verdict ===\n")
for R in RADII:
    d = load(R)
    for beta, lab in (("absent", "false positive"), ("present", "detection")):
        for spec in ("coordfree", "full"):
            x = d[(d.beta_state == beta) & (d.spec == spec)]
            diff, lo, hi, b01, b10, p = mcnemar(x.positive_verdict, x.cell_positive_verdict)
            rows.append(dict(radius_km=R, outcome=lab, spec=spec, n=len(x),
                             station_pct=round(100 * x.positive_verdict.mean(), 2),
                             cell_pct=round(100 * x.cell_positive_verdict.mean(), 2),
                             diff_pp=round(diff, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                             discordant_station_only=b01, discordant_cell_only=b10,
                             mcnemar_p=p))
            print(f"  R={R:>3} km  {lab:14s} {spec:10s} station {100*x.positive_verdict.mean():6.2f}%"
                  f"  cell {100*x.cell_positive_verdict.mean():6.2f}%"
                  f"  diff {diff:+6.2f} pp [{lo:+.2f},{hi:+.2f}]  McNemar p={p:.2e}")
    print()

print("=== the corner that fails: temporal dependence acting alone ===\n")
for R in RADII:
    d = load(R)
    c = d[(d.beta_state == "absent") & (d.temporal == "on")
          & (d.spatial == "off") & (d.grid == "off")]
    diff, lo, hi, b01, b10, p = mcnemar(c.positive_verdict, c.cell_positive_verdict)
    rows.append(dict(radius_km=R, outcome="false positive, temporal-only corner", spec="both",
                     n=len(c), station_pct=round(100 * c.positive_verdict.mean(), 2),
                     cell_pct=round(100 * c.cell_positive_verdict.mean(), 2),
                     diff_pp=round(diff, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                     discordant_station_only=b01, discordant_cell_only=b10, mcnemar_p=p))
    print(f"  R={R:>3} km  station {100*c.positive_verdict.mean():6.2f}%"
          f"  cell {100*c.cell_positive_verdict.mean():6.2f}%  diff {diff:+6.2f} pp  n={len(c)}")

print("\n=== control: geographic cells against a random partition of the same size ===\n")
for R in RADII:
    d = load(R)
    for beta, lab in (("absent", "false positive"), ("present", "detection")):
        x = d[d.beta_state == beta]
        diff, lo, hi, b01, b10, p = mcnemar(x.rnd_positive_verdict, x.cell_positive_verdict)
        rows.append(dict(radius_km=R, outcome=lab + ", cells vs random groups", spec="both",
                         n=len(x), station_pct=round(100 * x.rnd_frac.mean(), 2),
                         cell_pct=round(100 * x.cell_positive_verdict.mean(), 2),
                         diff_pp=round(diff, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         discordant_station_only=b01, discordant_cell_only=b10, mcnemar_p=p))
        print(f"  R={R:>3} km  {lab:14s} random groups {100*x.rnd_frac.mean():6.2f}%"
              f"  geographic cells {100*x.cell_positive_verdict.mean():6.2f}%"
              f"  diff {diff:+6.2f} pp  McNemar p={p:.3g}")
    c = d[(d.beta_state == "absent") & (d.temporal == "on")
          & (d.spatial == "off") & (d.grid == "off")]
    print(f"           temporal-only corner: station {100*c.positive_verdict.mean():.2f}%"
          f"  cells {100*c.cell_positive_verdict.mean():.2f}%"
          f"  random {100*c.rnd_frac.mean():.2f}%")

print("\n=== by dependence pathway ===")
print("    relative reduction in the false-positive rate, pooled over radii\n")
for sp, te, gr, name in PATHWAYS:
    st = cl = n = 0
    for R in RADII:
        d = load(R)
        z = d[(d.beta_state == "absent") & (d.spatial == sp)
              & (d.temporal == te) & (d.grid == gr)]
        st += z.positive_verdict.sum()
        cl += z.cell_positive_verdict.sum()
        n += len(z)
    a, b = 100 * st / n, 100 * cl / n
    rel = (100 * (1 - b / a)) if a > 0 else float("nan")
    rows.append(dict(radius_km="pooled", outcome="false positive by pathway", spec=name,
                     n=n, station_pct=round(a, 2), cell_pct=round(b, 2),
                     diff_pp=round(b - a, 2), ci_lo=None, ci_hi=None,
                     discordant_station_only=None, discordant_cell_only=None,
                     mcnemar_p=None))
    print(f"  {name:18s} station {a:6.2f}%  cell {b:6.2f}%  relative reduction "
          f"{rel:5.1f}%   n={n}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}")

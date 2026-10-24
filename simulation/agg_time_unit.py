"""Does a unit defined over time repair the corner that no spatial coarsening could?

The companion analysis (agg_cell_unit.py) asked what a coarser *spatial* inference unit buys and
answered, against a control holding the number of units fixed, that most of the gain is arithmetic.
It left one corner unrepaired: when the residual carries a component common to the whole network in
time, with neither spatial proximity nor shared grid support present, the station-level sign test
declares transfer in about one null realization in eight, and reading the verdict at the
mechanistic cells does not lower that rate beyond what an arbitrary grouping of the same size
lowers it.

That corner is created by dependence in time, so the unit that could absorb it is a unit defined
over time. This script reads the verdict four more ways on the same realizations:

  episode rule        the evaluation window cut into contiguous episodes, as many as there are
                      occupied cells, so the unit count matches the spatial rules exactly
  random time groups  the same timesteps scattered at random into the same number of groups,
                      which destroys contiguity while holding the count fixed

The quantity that settles it is not the corner rate on its own but the ratio of the corner rate to
the same rule's own pooled rate. A rule whose corner rate sits at its pooled rate is calibrated
there; a rule whose corner rate is several times its pooled rate is anti-conservative there. A unit
that merely declares transfer less often everywhere lowers both numbers and leaves the ratio alone.

Reads  <rundir>/R{25,50,100}/simulation_results.csv
Writes results/time_unit_comparison.csv and prints the tables the manuscript uses.

Usage:  python agg_time_unit.py
        CAMSCAST_EPRUN=/path/to/runs python agg_time_unit.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "simulation" else HERE
RUNDIR = os.environ.get("CAMSCAST_EPRUN", os.path.join(HERE, "results", "time_unit"))
OUT = os.environ.get("CAMSCAST_OUT",
                     os.path.join(ROOT, "results", "time_unit_comparison.csv"))
RADII = (25, 50, 100)

# (column carrying the per-realization verdict, label, whether it is a mean over repeats)
RULES = [
    ("positive_verdict", "station (19 units)", False),
    ("cell_positive_verdict", "grid cells (9)", False),
    ("rnd_frac", "random spatial groups (9)", True),
    ("ep_positive_verdict", "contiguous episodes (9)", False),
    ("eprnd_frac", "random time groups (9)", True),
]


def load(R):
    p = os.path.join(RUNDIR, f"R{R}", "simulation_results.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing run: {p}")
    d = pd.read_csv(p)
    return d[d.design == "full"]


def mcnemar(a, b):
    """Exact McNemar on paired binary verdicts, with the difference in percentage points."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    p = float(stats.binomtest(b10, n, 0.5).pvalue) if n else float("nan")
    return 100 * (b.mean() - a.mean()), b01, b10, p


def corner(d):
    return d[(d.beta_state == "absent") & (d.temporal == "on")
             & (d.spatial == "off") & (d.grid == "off")]


rows = []

print("=== verdict rates by inference unit, full audit ===\n")
print(f"  {'':28s} " + "".join(f"{'R=' + str(R):>12s}" for R in RADII))
for beta, lab in (("absent", "false positive"), ("present", "detection")):
    print(f"\n  {lab}")
    for col, name, _ in RULES:
        vals = []
        for R in RADII:
            x = load(R)
            x = x[x.beta_state == beta]
            vals.append(100 * x[col].mean())
        print(f"    {name:26s} " + "".join(f"{v:11.2f}%" for v in vals))

print("\n=== the corner that spatial coarsening could not repair ===")
print("    temporal dependence acting alone, beta = 0\n")
print(f"  {'':28s} " + "".join(f"{'R=' + str(R):>12s}" for R in RADII) + "   corner/pooled")
for col, name, _ in RULES:
    vals, ratios = [], []
    for R in RADII:
        d = load(R)
        c = corner(d)
        pooled = d[d.beta_state == "absent"][col].mean()
        v = 100 * c[col].mean()
        vals.append(v)
        ratios.append(v / (100 * pooled) if pooled > 0 else float("nan"))
    print(f"  {name:26s} " + "".join(f"{v:11.2f}%" for v in vals)
          + f"   {np.mean(ratios):7.2f}x")
    for R, v, r in zip(RADII, vals, ratios):
        rows.append(dict(radius_km=R, quantity="corner false-positive rate",
                         rule=name, value_pct=round(v, 2),
                         corner_over_pooled=round(r, 3)))

print("\n  A ratio near 1 means the rule is no more likely to declare transfer in this corner")
print("  than it is overall. A ratio of 5 to 7 is the anti-conservative failure the audit reports.")

print("\n=== paired comparisons against the station rule, corner only ===\n")
for R in RADII:
    c = corner(load(R))
    for col, name, mean_over_repeats in RULES[1:]:
        if mean_over_repeats:
            continue
        diff, b01, b10, p = mcnemar(c.positive_verdict, c[col])
        print(f"  R={R:>3} km  {name:26s} {diff:+7.2f} pp   "
              f"discordant {b01:3d}/{b10:3d}   McNemar p={p:.3g}")
        rows.append(dict(radius_km=R, quantity="corner, paired vs station",
                         rule=name, value_pct=round(diff, 2), mcnemar_p=p))
    print()

print("=== does contiguity matter, or only the dimension? ===")
print("    contiguous episodes against a random partition of the same size\n")
for R in RADII:
    d = load(R)
    c = corner(d)
    for lab, sub in (("false positive", d[d.beta_state == "absent"]),
                     ("detection", d[d.beta_state == "present"]),
                     ("corner", c)):
        ep = 100 * sub.ep_positive_verdict.mean()
        rd = 100 * sub.eprnd_frac.mean()
        print(f"  R={R:>3} km  {lab:14s} contiguous {ep:6.2f}%   random {rd:6.2f}%   "
              f"difference {ep - rd:+6.2f} pp")
        rows.append(dict(radius_km=R, quantity=f"contiguity, {lab}",
                         rule="episodes minus random time groups",
                         value_pct=round(ep - rd, 2)))
    print()

ep_len = [float(load(R).episode_len_steps.median()) for R in RADII]
print(f"episode length: {np.mean(ep_len):.2f} steps "
      f"({int(load(50).n_episodes.median())} episodes across a 20-step window)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}")

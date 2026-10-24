"""Do the reported operating characteristics survive a different spatial covariance?

The primary experiment draws the spatially correlated station component from a zero-mean Gaussian
field with exponential covariance at a 70 km range. That range sits inside the 25-100 km band over
which the observed residual field is spatially structured, but it is assumed rather than estimated,
and a single covariance family does not establish that the verdict rates, or the ranking of
inference units, hold for a field of different roughness or correlation length.

This script compares the primary specification against three alternatives run at the primary
buffer, each a full 32-scenario factorial with 100 replicates:

  rho35      exponential, range halved to 35 km
  rho140     exponential, range doubled to 140 km
  matern32   Matern nu = 3/2 at 70 km, once mean-square differentiable rather than rough

The quantity that matters is not whether the rates move -- a different field is a different world
and they should move -- but whether the conclusions drawn from them survive: that the full audit
holds false-positive verdicts near a few per cent, that one corner remains anti-conservative, that
spatial coarsening does not repair that corner, and that a temporal unit does.

Reads  <primary>/simulation_results.csv and <covdir>/{rho35,rho140,matern32}/simulation_results.csv
Writes results/cov_sensitivity.csv and prints the table the manuscript and supplement use.

Usage:  CAMSCAST_PRIMARY=/path/to/R50 CAMSCAST_COVDIR=/path/to/runs python agg_cov_sensitivity.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "simulation" else HERE
PRIMARY = os.environ.get("CAMSCAST_PRIMARY",
                         os.path.join(HERE, "results", "time_unit", "R50"))
COVDIR = os.environ.get("CAMSCAST_COVDIR", os.path.join(HERE, "results", "cov_sensitivity"))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(ROOT, "results", "cov_sensitivity.csv"))

SPECS = [("primary", PRIMARY, "exponential, 70 km"),
         ("rho35", os.path.join(COVDIR, "rho35"), "exponential, 35 km"),
         ("rho140", os.path.join(COVDIR, "rho140"), "exponential, 140 km"),
         ("matern32", os.path.join(COVDIR, "matern32"), "Matern 3/2, 70 km")]

RULES = [("positive_verdict", "station (19)", False),
         ("cell_positive_verdict", "grid cells (9)", False),
         ("rnd_frac", "random spatial (9)", True),
         ("ep_positive_verdict", "episodes (9)", False),
         ("eprnd_frac", "random time (9)", True)]


def load(path):
    p = os.path.join(path, "simulation_results.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing run: {p}")
    return pd.read_csv(p).query("design == 'full'")


def corner(d):
    return d[(d.beta_state == "absent") & (d.temporal == "on")
             & (d.spatial == "off") & (d.grid == "off")]


def mcse(p, n):
    return 100 * np.sqrt(p * (1 - p) / n) if n else float("nan")


rows = []
frames = {}
for tag, path, label in SPECS:
    frames[tag] = load(path)

print("=== full audit at R = 50 km, by spatial covariance specification ===\n")
hdr = f"  {'':26s}" + "".join(f"{lab:>22s}" for _, _, lab in SPECS)
print(hdr)

for beta, blab in (("absent", "false positive"), ("present", "detection")):
    print(f"\n  {blab}")
    for col, rlab, _ in RULES:
        cells = []
        for tag, _, _ in SPECS:
            d = frames[tag]
            x = d[d.beta_state == beta]
            v = 100 * x[col].mean()
            cells.append(f"{v:21.2f}%")
            rows.append(dict(spec=tag, quantity=blab, rule=rlab,
                             value_pct=round(v, 2), n=len(x),
                             mcse_pp=round(mcse(x[col].mean(), len(x)), 3)))
        print(f"    {rlab:24s}" + "".join(cells))

print("\n=== the corner: temporal dependence acting alone ===")
print("    rate, and that rule's corner rate over its own pooled false-positive rate\n")
print(hdr)
for col, rlab, _ in RULES:
    cells = []
    for tag, _, _ in SPECS:
        d = frames[tag]
        c = corner(d)
        pooled = d[d.beta_state == "absent"][col].mean()
        v = 100 * c[col].mean()
        r = (c[col].mean() / pooled) if pooled > 0 else float("nan")
        cells.append(f"{v:14.2f}% {r:5.2f}x")
        rows.append(dict(spec=tag, quantity="corner false positive", rule=rlab,
                         value_pct=round(v, 2), corner_over_pooled=round(float(r), 3),
                         n=len(c), mcse_pp=round(mcse(c[col].mean(), len(c)), 3)))
    print(f"  {rlab:24s}" + "".join(cells))

print("\n=== do the conclusions survive? ===\n")
checks = []
for tag, _, label in SPECS:
    d = frames[tag]
    nul = d[d.beta_state == "absent"]
    c = corner(d)
    st_pool = nul.positive_verdict.mean()
    st_corner = c.positive_verdict.mean()
    cl_corner = c.cell_positive_verdict.mean()
    rnd_corner = c.rnd_frac.mean()
    ep_corner = c.ep_positive_verdict.mean()
    ep_pool = nul.ep_positive_verdict.mean()
    out = dict(
        spec=label,
        audit_specific=st_pool < 0.05,
        corner_anticonservative=(st_corner / st_pool) > 2 if st_pool > 0 else False,
        cells_do_not_repair=cl_corner >= rnd_corner - 0.01,
        episodes_repair=(ep_corner / ep_pool) <= 1.0 if ep_pool > 0 else False,
    )
    checks.append(out)
    print(f"  {label:24s} "
          f"full audit < 5% : {str(out['audit_specific']):5s}   "
          f"corner > 2x pooled : {str(out['corner_anticonservative']):5s}   "
          f"cells no better than random : {str(out['cells_do_not_repair']):5s}   "
          f"episodes calibrated : {str(out['episodes_repair']):5s}")
    rows.append(dict(spec=tag, quantity="conclusion check", rule="-",
                     **{k: v for k, v in out.items() if k != "spec"}))

allsurvive = all(all([c["audit_specific"], c["corner_anticonservative"],
                      c["cells_do_not_repair"], c["episodes_repair"]]) for c in checks)
print(f"\n  every conclusion holds in every specification: {allsurvive}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}")

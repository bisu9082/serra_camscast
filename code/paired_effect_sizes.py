"""Paired effect sizes for the coordinate contrast under the full three-fold control.

The manuscript compares the coordinate-free and coordinate-bearing specifications as a
paired difference across the nineteen held-out stations, rather than by comparing two
separate one-sample verdicts (Gelman and Stern 2006). A paired test should be reported
with a paired effect size, and the supplement already does this for the benchmark arm
(Cohen's d_z, Supplementary Section S13). This script supplies the same two quantities for
the paper's own contrast, so that the reporting standard is the one standard throughout:

  Cohen's d_z          mean of the paired differences over their standard deviation, with a
                       bootstrap percentile interval over stations resampled with replacement
  rank-biserial r      the matched-pairs effect size that belongs with the Wilcoxon
                       signed-rank statistic actually reported

Both are computed from the per-station deltas already deposited in
results/audit_v7_qc6.json under the full spatial, temporal and grid-cell control. Nothing
here re-fits a model; it reads the same per-station values the verdicts were formed from.

Reads  results/audit_v7_qc6.json
Writes results/paired_effect_sizes.json
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "code" else HERE
RESULTS = os.environ.get("CAMSCAST_RESULTS", os.path.join(ROOT, "results"))
SRC = os.path.join(RESULTS, "audit_v7_qc6.json")
OUT = os.path.join(RESULTS, "paired_effect_sizes.json")

RADII = (0, 25, 50, 100)
N_BOOT = 10000
SEED = 0

if not os.path.exists(SRC):
    raise SystemExit(f"missing {SRC}")
v7 = json.load(open(SRC))

rows = []
for R in RADII:
    a = v7[f"buffer_time_cell_coordfree_R{R}"]["per_station"]
    b = v7[f"buffer_time_cell_full_R{R}"]["per_station"]
    ids = sorted(set(a) & set(b))
    d = np.array([a[i]["delta"] - b[i]["delta"] for i in ids], float)
    n = len(d)
    d_z = float(d.mean() / d.std(ddof=1))
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d, (N_BOOT, n), replace=True)
    lo, hi = np.percentile(boot.mean(1) / boot.std(axis=1, ddof=1), [2.5, 97.5])
    w = stats.wilcoxon(d)
    r_rb = float(1 - (2 * w.statistic) / (n * (n + 1) / 2))
    rows.append(dict(radius_km=R, n=n,
                     median_paired_diff=round(float(np.median(d)), 4),
                     d_z=round(d_z, 3), d_z_ci=[round(float(lo), 3), round(float(hi), 3)],
                     rank_biserial_r=round(r_rb, 3),
                     n_favouring_coordfree=int((d > 0).sum()),
                     wilcoxon_p=round(float(w.pvalue), 5)))

json.dump(dict(contrast="coordinate-free minus coordinate-bearing",
               control="full spatial, temporal and grid-cell",
               bootstrap_replicates=N_BOOT, seed=SEED, by_radius=rows),
          open(OUT, "w"), indent=1)

print("paired effect sizes, coordinate-free minus coordinate-bearing, full control")
for r in rows:
    print(f"  R={r['radius_km']:>3} km  n={r['n']}  median diff {r['median_paired_diff']:+.4f}  "
          f"d_z={r['d_z']:+.3f} [{r['d_z_ci'][0]:+.3f},{r['d_z_ci'][1]:+.3f}]  "
          f"rank-biserial r={r['rank_biserial_r']:+.3f}  "
          f"{r['n_favouring_coordfree']}/{r['n']} favour coordinate-free")
print(f"\nwrote {OUT}")

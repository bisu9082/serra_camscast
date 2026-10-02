"""Paired comparison between the four learners of the empirical audit.

Supplementary Section S13 reports two separate things about the empirical four-learner
check, and they have to be kept apart:

  (a) does each learner, on its own, reproduce a positive transfer verdict?
  (b) do the learners differ from one another?

Answering (b) by inspecting whether each learner's own interval excludes zero is the error
this manuscript objects to elsewhere (Gelman and Stern 2006), and an earlier version of the
supplement committed it. The four learners are fitted on the same 19 stations, so the
difference between two of them is directly testable, paired by station. That is what this
script does: a paired Wilcoxon signed-rank test of the per-station gains against the
reported depth-6 model, with a bootstrap interval on the median paired difference and
Benjamini-Hochberg across the three comparisons within each specification.

Reads results/audit_v5.json, writes results/empirical_learner_paired.csv.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "code" else HERE
RESULTS = os.environ.get("CAMSCAST_RESULTS", os.path.join(ROOT, "results"))
OUT = os.path.join(RESULTS, "empirical_learner_paired.csv")

REFERENCE = "gbm_d6"
COMPARATORS = ("gbm_d2", "rf", "ridge")
SPECS = ("full", "coordfree")       # coordinate-bearing, coordinate-free
RADIUS = 100
N_BOOT = 4000
SEED = 0


def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = min(1.0, prev)
    return q


path = os.path.join(RESULTS, "audit_v5.json")
if not os.path.exists(path):
    sys.exit(f"missing {path}")
learners = json.load(open(path))["learners"]

rows = []
for spec in SPECS:
    ref = learners[f"{REFERENCE}_{spec}_R{RADIUS}"]["per_station"]
    ids = sorted(ref)
    a = np.array([ref[i] for i in ids], float)
    block, pvals = [], []
    for comp in COMPARATORS:
        key = f"{comp}_{spec}_R{RADIUS}"
        if key not in learners:
            print(f"[missing] {key}", file=sys.stderr)
            continue
        b = np.array([learners[key]["per_station"][i] for i in ids], float)
        diff = a - b
        p = float(stats.wilcoxon(diff).pvalue)
        rng = np.random.default_rng(SEED)
        boot = np.median(rng.choice(diff, (N_BOOT, len(diff)), replace=True), axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pvals.append(p)
        block.append(dict(spec=spec, radius_km=RADIUS, reference=REFERENCE, comparator=comp,
                          median_paired_diff_pp=round(float(np.median(diff)), 3),
                          ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
                          n_ref_higher=int((diff > 0).sum()), n=len(diff),
                          wilcoxon_p=round(p, 5)))
    for r, q in zip(block, bh(pvals)):
        r["bh_q"] = round(q, 5)
    rows.extend(block)

os.makedirs(RESULTS, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

label = {"full": "coordinate-bearing", "coordfree": "coordinate-free"}
name = {"gbm_d2": "GBM depth 2", "rf": "Random forest", "ridge": "Ridge"}
for spec in SPECS:
    print(f"\n=== R={RADIUS} km, {label[spec]} (reference: {REFERENCE}) ===")
    for r in [x for x in rows if x["spec"] == spec]:
        print(f"  vs {name[r['comparator']]:<15} median paired diff "
              f"{r['median_paired_diff_pp']:+7.2f} pp "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
              f"Wilcoxon p={r['wilcoxon_p']:.5f}  q={r['bh_q']:.5f}  "
              f"({r['n_ref_higher']}/{r['n']} stations higher under the reference)")
print(f"\nwrote {OUT}")

"""Aggregate the ridge and tree simulation runs, and test the learners against each other.

Two outputs, both written next to this script under results/:

  learner_check_3radii.csv   full-audit verdict rates with Wilson 95% intervals, per radius,
                             learner, beta state and specification.
  learner_direct_tests.csv   the twelve paired learner comparisons: difference in percentage
                             points, its 95% interval, Fisher's exact p, and the
                             Benjamini-Hochberg q across the family of twelve.

The comparison is the *difference between the learners*, tested directly. It is deliberately
not "do their separate confidence intervals overlap", which is the error this manuscript
objects to elsewhere (Gelman and Stern 2006) and which an earlier version of this script
committed: the overlap reading calls all twelve pairs identical, and one of them is not.

Reproduces Supplementary Table S43.

Usage:  python agg_learner.py            (reads results/simulation_summary_R*{,_hgb}.csv)
        CAMSCAST_SIMDIR=... python agg_learner.py     to point at another results directory
"""
import math
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.environ.get("CAMSCAST_SIMDIR", os.path.join(HERE, "results"))
RADII = (25, 50, 100)
SPECS = ("coordfree", "full")
BETAS = ("absent", "present")


def wilson(k, n, z=1.959963985):
    """Wilson score interval, in per cent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def cells(path, design="full"):
    """Verdict counts for one run, aggregated over the 16 scenarios of each beta state."""
    d = pd.read_csv(path)
    d = d[d.design == design]
    out = {}
    for (beta, spec), g in d.groupby(["beta_state", "spec"]):
        n = int(g.n_rep.sum())
        k = int(round((g.verdict_rate * g.n_rep).sum()))
        lo, hi = wilson(k, n)
        out[(beta, spec)] = dict(k=k, n=n, rate=100 * k / n, lo=lo, hi=hi,
                                 dtil=float(g.median_of_delta.median()))
    return out


def bh(pvals):
    """Benjamini-Hochberg step-up, returning q in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = min(1.0, prev)
    return q


rows = []
for R in RADII:
    for learner, suffix in (("ridge", ""), ("hgb", "_hgb")):
        path = os.path.join(SIMDIR, f"simulation_summary_R{R}{suffix}.csv")
        if not os.path.exists(path):
            print(f"[missing] R={R} {learner}: {path}", file=sys.stderr)
            continue
        for (beta, spec), v in cells(path).items():
            rows.append(dict(R=R, learner=learner, beta=beta, spec=spec, **v))

if not rows:
    sys.exit(f"no run summaries found under {SIMDIR}")

t = pd.DataFrame(rows)
os.makedirs(SIMDIR, exist_ok=True)
t.to_csv(os.path.join(SIMDIR, "learner_check_3radii.csv"), index=False)

print("===== full-audit verdict rates, Wilson 95% intervals =====")
for beta in BETAS:
    print(f"\n  beta {beta}")
    for _, r in t[t.beta == beta].sort_values(["spec", "R", "learner"]).iterrows():
        print(f"    R={r.R:>3} {r.spec:<10} {r.learner:<6} {r['rate']:6.2f}% "
              f"[{r.lo:5.2f}, {r.hi:5.2f}]  n={r.n:<5} dtil={r.dtil:+.4f}")

# ---------------------------------------------------------------- direct tests
comp = []
for R in RADII:
    for beta in BETAS:
        for spec in SPECS:
            a = t[(t.R == R) & (t.beta == beta) & (t.spec == spec) & (t.learner == "ridge")]
            h = t[(t.R == R) & (t.beta == beta) & (t.spec == spec) & (t.learner == "hgb")]
            if a.empty or h.empty:
                continue
            a, h = a.iloc[0], h.iloc[0]
            table = [[int(h.k), int(h.n - h.k)], [int(a.k), int(a.n - a.k)]]
            p = float(stats.fisher_exact(table)[1])
            diff = h["rate"] - a["rate"]
            se = 100 * math.sqrt(h["rate"] / 100 * (1 - h["rate"] / 100) / h.n
                                 + a["rate"] / 100 * (1 - a["rate"] / 100) / a.n)
            comp.append(dict(R=R, beta=beta, spec=spec,
                             ridge=a["rate"], hgb=h["rate"], diff=diff,
                             lo=diff - 1.96 * se, hi=diff + 1.96 * se, p=p,
                             ci_overlap="yes" if (a.lo <= h.hi and h.lo <= a.hi) else "no"))

c = pd.DataFrame(comp)
c["q"] = bh(list(c.p))
c["direct"] = ["same" if p >= 0.05 else "DIFFERENT" for p in c.p]
c = c.sort_values("p").reset_index(drop=True)
c.to_csv(os.path.join(SIMDIR, "learner_direct_tests.csv"), index=False)

print("\n===== paired learner comparisons (Fisher exact, BH over the twelve) =====")
print(f"  {'R':>4} {'beta':<8}{'spec':<11}{'ridge':>7}{'tree':>7}{'diff':>8}"
      f"{'95% CI':>18}{'p':>8}{'q':>7}  overlap")
for _, r in c.iterrows():
    print(f"  {int(r.R):>4} {r.beta:<8}{r.spec:<11}{r.ridge:7.2f}{r.hgb:7.2f}{r['diff']:+8.2f}"
          f"  [{r.lo:+6.2f},{r.hi:+6.2f}]{r.p:8.3f}{r.q:7.3f}  {r.ci_overlap}")

n_same_direct = int((c.p >= 0.05).sum())
n_same_overlap = int((c.ci_overlap == "yes").sum())
print(f"\n  interval-overlap reading calls {n_same_overlap}/12 pairs identical")
print(f"  direct test (raw p)       calls {n_same_direct}/12 pairs identical")
print(f"  after Benjamini-Hochberg  calls {int((c.q >= 0.05).sum())}/12 pairs identical")
print(f"\n  wrote learner_check_3radii.csv and learner_direct_tests.csv to {SIMDIR}")

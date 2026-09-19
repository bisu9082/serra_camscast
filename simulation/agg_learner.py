import pandas as pd, numpy as np, sys, math

def wilson(k, n, z=1.959963985):
    if n == 0: return (float('nan'),)*2
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

def cell(path, design="full"):
    d = pd.read_csv(path)
    d = d[d.design == design]
    out = {}
    for (b, spec), g in d.groupby(["beta_state", "spec"]):
        n = int(g.n_rep.sum())
        k = int(round((g.verdict_rate * g.n_rep).sum()))
        lo, hi = wilson(k, n)
        out[(b, spec)] = dict(k=k, n=n, rate=100*k/n, lo=lo, hi=hi,
                              dtil=float(g.median_of_delta.median()))
    return out

rows = []
for R in (25, 50, 100):
    for lab, path in (("ridge", f"out_R{R}/simulation_summary.csv"),
                      ("hgb",   f"out_R{R}_hgb/simulation_summary.csv")):
        try: c = cell(path)
        except FileNotFoundError: print(f"[missing] R={R} {lab}"); continue
        for (b, spec), v in c.items():
            rows.append(dict(R=R, learner=lab, beta=b, spec=spec, **v))
t = pd.DataFrame(rows)
t.to_csv("/tmp/serra/learner_check_3radii.csv", index=False)
pd.set_option("display.width", 200)
for b in ("absent", "present"):
    print(f"\n===== beta {b} : full audit =====")
    s = t[t.beta == b].sort_values(["spec", "R", "learner"])
    for _, r in s.iterrows():
        print(f"  R={r.R:>3} {r.spec:<10} {r.learner:<6} {r['rate']:6.2f}% "
              f"[{r.lo:5.2f}, {r.hi:5.2f}]  n={r.n:<5} dtil={r.dtil:+.4f}")
# overlap test
print("\n===== CI overlap between learners (full audit) =====")
for R in sorted(t.R.unique()):
    for b in ("absent", "present"):
        for spec in ("coordfree", "full"):
            a = t[(t.R==R)&(t.beta==b)&(t.spec==spec)&(t.learner=="ridge")]
            h = t[(t.R==R)&(t.beta==b)&(t.spec==spec)&(t.learner=="hgb")]
            if a.empty or h.empty: continue
            a=a.iloc[0]; h=h.iloc[0]
            ov = (a.lo <= h.hi) and (h.lo <= a.hi)
            print(f"  R={R:>3} beta={b:<7} {spec:<10} ridge {a['rate']:5.2f}[{a.lo:.2f},{a.hi:.2f}] "
                  f"vs hgb {h['rate']:5.2f}[{h.lo:.2f},{h.hi:.2f}]  -> {'OVERLAP' if ov else 'DISJOINT'}")

"""Quantify run-to-run instability of the ORIGINAL (unseeded, early_stopping='auto')
configuration, which is what produced the archived and manuscript numbers."""
import json, os, numpy as np
from loso_v2 import run, FULL, NOXY
from scipy import stats

SEEDS = list(range(10))
out = {}
for fname, cols in [("full",FULL),("coordfree",NOXY)]:
    for R in [0,25,50,100]:
        rows=[]
        for s in SEEDS:
            res = run(R, cols, seed=s, early=True)   # 'auto' -> True for n>10k; random 10% val split
            tg=sorted(res)
            raw=np.array([res[t][0] for t in tg]); tr=np.array([res[t][1] for t in tg])
            g=100*(raw-tr)/raw
            rows.append(dict(seed=s, mean=float(g.mean()), median=float(np.median(g)),
                             pooled=float(100*(raw.mean()-tr.mean())/raw.mean()),
                             wins=int((g>0).sum())))
        m=np.array([r['mean'] for r in rows]); md=np.array([r['median'] for r in rows])
        p=np.array([r['pooled'] for r in rows]); w=np.array([r['wins'] for r in rows])
        out[f"{fname}_R{R}"]=dict(seeds=rows,
            mean_across=float(m.mean()), mean_sd=float(m.std(ddof=1)), mean_range=[float(m.min()),float(m.max())],
            median_across=float(md.mean()), median_sd=float(md.std(ddof=1)), median_range=[float(md.min()),float(md.max())],
            pooled_across=float(p.mean()), pooled_sd=float(p.std(ddof=1)), pooled_range=[float(p.min()),float(p.max())],
            wins_range=[int(w.min()),int(w.max())], wins_mode=int(stats.mode(w,keepdims=False).mode))
        print(f"[{fname:9s} R={R:>3}] over 10 seeds: mean {m.mean():+6.2f}+-{m.std(ddof=1):.2f} "
              f"[{m.min():+.1f},{m.max():+.1f}] | median {md.mean():+6.2f}+-{md.std(ddof=1):.2f} "
              f"| pooled {p.mean():+6.2f}+-{p.std(ddof=1):.2f} | wins {w.min()}-{w.max()}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loso_v2_stability.json"),"w"), indent=1)
print("SAVED loso_v2_stability.json")

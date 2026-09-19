"""Heavy re-runs demanded by the second panel:
  P. coordinate-permutation placebo with a real null (200 space-only + 50 under LLTO)
  Q. source-set bootstrap at 200 replicates (was 12)
Runs long; writes audit_v4_heavy.json.
"""
import json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from loso_v2 import PM10C, DATA, hav, FULL, NOXY
from auditlib import sources, fit_predict, GUARD, NBLK

OUT = {}
IDS = sorted(PM10C)

def gbm(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                         random_state=seed, early_stopping=False)

def pooled_gain(res):
    tg = sorted(res)
    raw = np.array([res[t][0] for t in tg]); tr = np.array([res[t][1] for t in tg])
    n = np.array([res[t][2] for t in tg])
    g = 100 * (raw - tr) / raw
    return (100 * (np.average(raw, weights=n) - np.average(tr, weights=n)) / np.average(raw, weights=n),
            float(np.median(g)), int((g > 0).sum()))

def run_feats(R, cols, feats, time_block):
    out = {}
    for tgt in PM10C:
        src = sources(tgt, R, "dist")
        if len(src) < 4:
            continue
        pred = fit_predict(tgt, src, cols, feats, time_block)
        dft = DATA[tgt][0]; obs = dft["obs"].values; cams = dft["cams"].values
        ok = np.isfinite(pred)
        out[tgt] = (float(np.mean(np.abs(cams[ok] - obs[ok]))),
                    float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok]))), int(ok.sum()))
    return out

# ---------------- P. placebo with a null distribution ----------------
BASE = {i: DATA[i][3] for i in PM10C}
for design, nperm, tb in [("space_only", 200, False), ("llto", 50, True)]:
    obs_stat = pooled_gain(run_feats(0, FULL, BASE, tb))
    perms = []
    for rep in range(nperm):
        p = list(np.random.default_rng(9000 + rep).permutation(IDS))
        fk = {}
        for a, b in zip(IDS, p):
            f = DATA[a][3].copy(); f["lat"] = PM10C[b][0]; f["lon"] = PM10C[b][1]; fk[a] = f
        perms.append(pooled_gain(run_feats(0, FULL, fk, tb)))
        if (rep + 1) % 10 == 0:
            v = np.array([x[0] for x in perms])
            print(f"  [{design}] {rep+1}/{nperm}  placebo pooled mean {v.mean():+.2f} sd {v.std(ddof=1):.2f}", flush=True)
    v = np.array([x[0] for x in perms])
    # two-sided permutation p for "true coordinates beat scrambled ones"
    p_two = (np.sum(np.abs(v - v.mean()) >= abs(obs_stat[0] - v.mean())) + 1) / (nperm + 1)
    p_one = (np.sum(v >= obs_stat[0]) + 1) / (nperm + 1)
    OUT[f"placebo_{design}"] = dict(
        n_perm=nperm, observed_pooled=obs_stat[0], observed_median=obs_stat[1], observed_wins=obs_stat[2],
        perm_mean=float(v.mean()), perm_sd=float(v.std(ddof=1)),
        perm_lo=float(np.percentile(v, 2.5)), perm_hi=float(np.percentile(v, 97.5)),
        perm_min=float(v.min()), perm_max=float(v.max()),
        p_two_sided=float(p_two), p_one_sided_true_ge_perm=float(p_one),
        frac_perm_beating_true=float(np.mean(v >= obs_stat[0])))
    print(f"[{design}] observed {obs_stat[0]:+.2f} | perm {v.mean():+.2f}+-{v.std(ddof=1):.2f} "
          f"[{np.percentile(v,2.5):+.1f},{np.percentile(v,97.5):+.1f}] | p_two={p_two:.3f} "
          f"| {100*np.mean(v>=obs_stat[0]):.0f}% of permutations match or beat the true assignment", flush=True)
    json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v4_heavy.json"), "w"), indent=1)

# ---------------- Q. source-set bootstrap, 200 replicates ----------------
NB = 200
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for R in (0, 100):
        vals = []
        for b in range(NB):
            r2 = np.random.default_rng(7000 + b)
            res = {}
            for tgt in PM10C:
                src = sources(tgt, R, "dist")
                if len(src) < 4:
                    continue
                pick = list(r2.choice(src, len(src), replace=True))
                Xtr = pd.concat([DATA[j][3][cols] for j in pick])
                ytr = np.concatenate([DATA[j][0]["resid"].values for j in pick])
                pred = gbm().fit(Xtr, ytr).predict(DATA[tgt][3][cols])
                dft = DATA[tgt][0]; obs = dft["obs"].values; cams = dft["cams"].values
                res[tgt] = (float(np.mean(np.abs(cams - obs))),
                            float(np.mean(np.abs(cams + pred - obs))), len(obs))
            vals.append(pooled_gain(res)[0])
            if (b + 1) % 25 == 0:
                v = np.array(vals)
                print(f"  [{nm} R={R}] {b+1}/{NB}  {v.mean():+.2f} +- {v.std(ddof=1):.2f}", flush=True)
        v = np.array(vals)
        OUT[f"srcboot_{nm}_R{R}"] = dict(n_boot=NB, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                         lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                                         mn=float(v.min()), mx=float(v.max()))
        print(f"[srcboot {nm} R={R}] {v.mean():+.2f} +- {v.std(ddof=1):.2f} "
              f"[{np.percentile(v,2.5):+.1f},{np.percentile(v,97.5):+.1f}]", flush=True)
        json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v4_heavy.json"), "w"), indent=1)

print("DONE", flush=True)

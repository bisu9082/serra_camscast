"""Tail of the heavy re-run, sized to finish on two cores.

  P2. coordinate-permutation placebo under the LLTO design (the headline design),
      NPERM permutations against the true assignment.
  Q2. source-set bootstrap at NB replicates (audit_v3 used 12).

Merges into audit_v4_heavy.json, so the space-only placebo already written there
is preserved.
"""
import json, os, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from loso_v2 import PM10C, DATA, hav, FULL, NOXY
from auditlib import sources, fit_predict, GUARD, NBLK

NPERM = int(os.environ.get("NPERM", 30))
NB = int(os.environ.get("NB", 100))
OUTP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v4_heavy.json")
OUT = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
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


def save():
    json.dump(OUT, open(OUTP, "w"), indent=1)


# ---------------- P2. placebo under LLTO ----------------
BASE = {i: DATA[i][3] for i in PM10C}
if "placebo_llto" not in OUT:
    obs_stat = pooled_gain(run_feats(0, FULL, BASE, True))
    print(f"[llto] observed pooled {obs_stat[0]:+.2f} median {obs_stat[1]:+.2f} "
          f"wins {obs_stat[2]}/19", flush=True)
    perms = []
    for rep in range(NPERM):
        p = list(np.random.default_rng(9000 + rep).permutation(IDS))
        fk = {}
        for a, b in zip(IDS, p):
            f = DATA[a][3].copy(); f["lat"] = PM10C[b][0]; f["lon"] = PM10C[b][1]; fk[a] = f
        perms.append(pooled_gain(run_feats(0, FULL, fk, True)))
        v = np.array([x[0] for x in perms])
        print(f"  [llto] {rep+1}/{NPERM}  last {perms[-1][0]:+.2f} | "
              f"null mean {v.mean():+.2f} sd {v.std(ddof=1) if len(v)>1 else float('nan'):.2f}", flush=True)
    v = np.array([x[0] for x in perms]); vm = np.array([x[1] for x in perms])
    OUT["placebo_llto"] = dict(
        n_perm=NPERM, observed_pooled=obs_stat[0], observed_median=obs_stat[1],
        observed_wins=obs_stat[2],
        perm_mean=float(v.mean()), perm_sd=float(v.std(ddof=1)),
        perm_lo=float(np.percentile(v, 2.5)), perm_hi=float(np.percentile(v, 97.5)),
        perm_min=float(v.min()), perm_max=float(v.max()),
        perm_median_mean=float(vm.mean()),
        p_one_sided_true_ge_perm=float((np.sum(v >= obs_stat[0]) + 1) / (NPERM + 1)),
        frac_perm_beating_true=float(np.mean(v >= obs_stat[0])),
        perm_pooled=[float(x) for x in v])
    save()
    print(f"[llto placebo] observed {obs_stat[0]:+.2f} | null {v.mean():+.2f}+-{v.std(ddof=1):.2f} "
          f"[{np.percentile(v,2.5):+.1f},{np.percentile(v,97.5):+.1f}] | "
          f"{100*np.mean(v>=obs_stat[0]):.0f}% of scrambled assignments match or beat the true one",
          flush=True)

# ---------------- Q2. source-set bootstrap ----------------
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for R in (0, 100):
        key = f"srcboot_{nm}_R{R}"
        if key in OUT:
            continue
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
            if (b + 1) % 10 == 0:
                v = np.array(vals)
                print(f"  [{nm} R={R}] {b+1}/{NB}  {v.mean():+.2f} +- {v.std(ddof=1):.2f}", flush=True)
        v = np.array(vals)
        OUT[key] = dict(n_boot=NB, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                        lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                        mn=float(v.min()), mx=float(v.max()))
        save()
        print(f"[srcboot {nm} R={R}] {v.mean():+.2f} +- {v.std(ddof=1):.2f} "
              f"[{np.percentile(v,2.5):+.1f},{np.percentile(v,97.5):+.1f}]", flush=True)

print("DONE", flush=True)

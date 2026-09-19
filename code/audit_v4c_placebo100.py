"""Coordinate-permutation placebo at the buffer radius where the transfer claim
actually lives (R = 100 km), under the full control (time blocks + guard band).

The placebos reported earlier are both at R = 0, where every neighbour is still
available; at R = 0 a coordinate feature can support genuine interpolation, so a
placebo there does not test the transfer claim. This script runs the same
permutation null at R = 100 km, where no source station lies within 100 km of the
target and any coordinate contribution must be extrapolation.

Writes audit_v4c.json under 'placebo_llto_R100'.
"""
import json, os, os
import numpy as np
from loso_v2 import PM10C, DATA, FULL
from auditlib import sources, fit_predict

NPERM = int(os.environ.get("NPERM", 30))
OUTP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v4c.json")
IDS = sorted(PM10C)


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


BASE = {i: DATA[i][3] for i in PM10C}
obs_stat = pooled_gain(run_feats(100, FULL, BASE, True))
print(f"[llto R=100] observed pooled {obs_stat[0]:+.2f} median {obs_stat[1]:+.2f} "
      f"wins {obs_stat[2]}/19", flush=True)

perms = []
for rep in range(NPERM):
    p = list(np.random.default_rng(9000 + rep).permutation(IDS))
    fk = {}
    for a, b in zip(IDS, p):
        f = DATA[a][3].copy(); f["lat"] = PM10C[b][0]; f["lon"] = PM10C[b][1]; fk[a] = f
    perms.append(pooled_gain(run_feats(100, FULL, fk, True)))
    v = np.array([x[0] for x in perms])
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    print(f"  [llto R=100] {rep+1}/{NPERM}  last {perms[-1][0]:+.2f} | "
          f"null mean {v.mean():+.2f} sd {sd:.2f}", flush=True)

v = np.array([x[0] for x in perms]); vm = np.array([x[1] for x in perms])
rec = dict(n_perm=NPERM, radius_km=100,
           observed_pooled=obs_stat[0], observed_median=obs_stat[1], observed_wins=obs_stat[2],
           perm_mean=float(v.mean()), perm_sd=float(v.std(ddof=1)),
           perm_lo=float(np.percentile(v, 2.5)), perm_hi=float(np.percentile(v, 97.5)),
           perm_min=float(v.min()), perm_max=float(v.max()),
           perm_median_mean=float(vm.mean()),
           p_one_sided_true_ge_perm=float((np.sum(v >= obs_stat[0]) + 1) / (NPERM + 1)),
           frac_perm_beating_true=float(np.mean(v >= obs_stat[0])),
           perm_pooled=[float(x) for x in v], perm_median=[float(x) for x in vm])

OUT = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
OUT["placebo_llto_R100"] = rec
json.dump(OUT, open(OUTP, "w"), indent=1)
print(f"[llto R=100 placebo] observed {obs_stat[0]:+.2f} | null {v.mean():+.2f}+-{v.std(ddof=1):.2f} "
      f"[{np.percentile(v,2.5):+.1f},{np.percentile(v,97.5):+.1f}] | "
      f"{100*np.mean(v>=obs_stat[0]):.0f}% of scrambled assignments match or beat the true one",
      flush=True)
print("DONE", flush=True)

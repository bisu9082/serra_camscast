"""camscast audit v3 — the eight re-analyses demanded by the GATE 8 panel.

Adds to the v2 (space-only) audit:
  A. leave-location-and-time-out  (spatial buffer + contiguous time blocks + guard band)
  B. leave-one-grid-cell-out      (withhold sources sharing the target's CAMS cell)
  C. paired full vs coordinate-free tests at every radius
  D. |CAMS bias| vs gain correlation at every radius
  E. feature ablation (cams / +hour / +doy / coordinate-free / full)
  F. placebo test (station coordinates permuted among stations)
  G. CIs on every headline pooled value, d_z CIs, BH correction within the audit family
  H. seed diagnostic redesigned as a source-station bootstrap of the deterministic estimator

Everything runs on the deterministic estimator (random_state=0, early_stopping=False)
unless stated. Outputs audit_v3.json.
"""
import os as _os
_RAW = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "results", "per_point")
import json, os, itertools, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from loso_v2 import PM10C, DATA, hav, FULL, NOXY

CAMS = ["cams"]
HOUR = ["cams", "hsin", "hcos"]
DOY  = ["cams", "dsin", "dcos"]
SETS = {"cams": CAMS, "cams+hour": HOUR, "cams+doy": DOY, "coordfree": NOXY, "full": FULL}
RADII = [0, 25, 50, 100]
B = 2000
GUARD = pd.Timedelta(days=3)      # >= typical multi-day dust episode
NBLK = 6

CELL = {i: (round(la / 0.75), round(lo / 0.75)) for i, (la, lo) in PM10C.items()}

def gbm(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                         random_state=seed, early_stopping=False)

def sources(tgt, R, mode):
    if mode == "cell":
        return [j for j in PM10C if j != tgt and CELL[j] != CELL[tgt]]
    return [j for j in PM10C if j != tgt and hav(PM10C[tgt], PM10C[j]) > R]

def fit_predict(tgt, src, cols, feats, time_block):
    dft = DATA[tgt][0]
    if not time_block:
        Xtr = pd.concat([feats[j][cols] for j in src])
        ytr = np.concatenate([DATA[j][0]["resid"].values for j in src])
        return gbm().fit(Xtr, ytr).predict(feats[tgt][cols])
    t0, t1 = dft.index.min(), dft.index.max()
    edges = pd.date_range(t0, t1, periods=NBLK + 1)
    pred = np.full(len(dft), np.nan)
    for b in range(NBLK):
        lo, hi = edges[b], edges[b + 1]
        te = (dft.index >= lo) & (dft.index < (hi if b < NBLK - 1 else hi + pd.Timedelta("1h")))
        if te.sum() == 0:
            continue
        Xtr, ytr = [], []
        for j in src:
            dj = DATA[j][0]
            keep = (dj.index < lo - GUARD) | (dj.index > hi + GUARD)
            if keep.sum() == 0:
                continue
            Xtr.append(feats[j][cols][keep]); ytr.append(dj["resid"].values[keep])
        if not Xtr:
            continue
        pred[te] = gbm().fit(pd.concat(Xtr), np.concatenate(ytr)).predict(feats[tgt][cols][te])
    return pred

def run(R, cols, mode="dist", time_block=False, feats=None):
    feats = feats or {i: DATA[i][3] for i in PM10C}
    out = {}
    for tgt in PM10C:
        src = sources(tgt, R, mode)
        if len(src) < 4:
            continue
        pred = fit_predict(tgt, src, cols, feats, time_block)
        dft = DATA[tgt][0]; obs = dft["obs"].values; cams = dft["cams"].values
        ok = np.isfinite(pred)
        out[tgt] = (float(np.mean(np.abs(cams[ok] - obs[ok]))),
                    float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok]))),
                    len(src), int(ok.sum()))
    return out

rng = np.random.default_rng(20260907)

def infer(res):
    tg = sorted(res)
    raw = np.array([res[t][0] for t in tg]); tr = np.array([res[t][1] for t in tg])
    n = np.array([res[t][3] for t in tg]); d = raw - tr
    g = 100 * (raw - tr) / raw
    pooled = 100 * (np.average(raw, weights=n) - np.average(tr, weights=n)) / np.average(raw, weights=n)
    idx = rng.integers(0, len(tg), size=(B, len(tg)))
    bp = [100 * (np.average(raw[i], weights=n[i]) - np.average(tr[i], weights=n[i]))
          / np.average(raw[i], weights=n[i]) for i in idx]
    bd = d[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        bdz = bd.mean(1) / bd.std(ddof=1, axis=1)
    bdz = bdz[np.isfinite(bdz)]
    wins = int((g > 0).sum())
    return dict(
        n_targets=len(tg), n_src_mean=float(np.mean([res[t][2] for t in tg])),
        n_src_min=int(min(res[t][2] for t in tg)),
        pooled=float(pooled), pooled_ci=[float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
        mean=float(g.mean()), median=float(np.median(g)),
        median_ci=[float(np.percentile(np.median(g[idx], axis=1), 2.5)),
                   float(np.percentile(np.median(g[idx], axis=1), 97.5))],
        wins=wins, binom_p=float(stats.binomtest(wins, len(tg), 0.5).pvalue),
        wilcoxon_p=float(stats.wilcoxon(raw, tr).pvalue),
        d_z=float(d.mean() / d.std(ddof=1)),
        d_z_ci=[float(np.percentile(bdz, 2.5)), float(np.percentile(bdz, 97.5))],
        per_station={int(t): round(float(x), 2) for t, x in zip(tg, g)},
        mae_transfer={int(t): float(res[t][1]) for t in tg})

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    q = np.empty(m); run_min = 1.0
    for k in range(m - 1, -1, -1):
        run_min = min(run_min, p[o[k]] * m / (k + 1)); q[o[k]] = run_min
    return np.minimum(q, 1.0)

OUT = {}
S = pd.read_csv(_os.path.join(_RAW, "fig1_station_table.csv"))
S = S[S.pol == "PM10"].copy(); S["id"] = S.st.str[1:].astype(int); S = S.set_index("id")

# ---------------- E. feature ablation ----------------
print("=== E. feature ablation ===", flush=True)
OUT["ablation"] = {}
for nm, cols in SETS.items():
    for R in (0, 100):
        s = infer(run(R, cols))
        OUT["ablation"][f"{nm}_R{R}"] = s
        print(f"  {nm:<10} R={R:>3}  pooled {s['pooled']:+7.2f} [{s['pooled_ci'][0]:+.1f},{s['pooled_ci'][1]:+.1f}]"
              f"  median {s['median']:+7.2f}  wins {s['wins']}/{s['n_targets']}", flush=True)

# ---------------- A + G. LLTO ladder, both feature sets ----------------
print("\n=== A. leave-location-and-time-out (6 blocks, 3-day guard) ===", flush=True)
OUT["space_only"] = {}; OUT["llto"] = {}
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for R in RADII:
        s0 = infer(run(R, cols)); s1 = infer(run(R, cols, time_block=True))
        OUT["space_only"][f"{nm}_R{R}"] = s0; OUT["llto"][f"{nm}_R{R}"] = s1
        print(f"  {nm:<10} R={R:>3}  space-only {s0['pooled']:+7.2f} [{s0['pooled_ci'][0]:+.1f},{s0['pooled_ci'][1]:+.1f}]"
              f" d_z {s0['d_z']:+.2f} {s0['wins']}/19   ->  LLTO {s1['pooled']:+7.2f}"
              f" [{s1['pooled_ci'][0]:+.1f},{s1['pooled_ci'][1]:+.1f}] d_z {s1['d_z']:+.2f} {s1['wins']}/19", flush=True)

# ---------------- B. leave-one-grid-cell-out ----------------
print("\n=== B. leave-one-grid-cell-out ===", flush=True)
OUT["cell"] = {}
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for tb in (False, True):
        s = infer(run(0, cols, mode="cell", time_block=tb))
        OUT["cell"][f"{nm}_{'llto' if tb else 'spaceonly'}"] = s
        print(f"  {nm:<10} {'+time-block' if tb else 'space-only ':<12} pooled {s['pooled']:+7.2f}"
              f" [{s['pooled_ci'][0]:+.1f},{s['pooled_ci'][1]:+.1f}]  d_z {s['d_z']:+.2f}"
              f"  wins {s['wins']}/{s['n_targets']}  sources {s['n_src_mean']:.1f}", flush=True)

# ---------------- C. paired full vs coordinate-free ----------------
print("\n=== C. paired full vs coordinate-free ===", flush=True)
OUT["paired"] = {}
for design, store in [("space_only", OUT["space_only"]), ("llto", OUT["llto"])]:
    for R in RADII:
        a = store[f"full_R{R}"]["mae_transfer"]; b = store[f"coordfree_R{R}"]["mae_transfer"]
        k = sorted(set(a) & set(b))
        d = np.array([a[t] - b[t] for t in k])          # >0 = coordinate-free better
        i2 = rng.integers(0, len(d), size=(B, len(d)))
        ci = np.percentile(d[i2].mean(1), [2.5, 97.5])
        w = stats.wilcoxon([a[t] for t in k], [b[t] for t in k])
        OUT["paired"][f"{design}_R{R}"] = dict(
            mean_diff=float(d.mean()), ci=[float(ci[0]), float(ci[1])],
            coordfree_better=int((d > 0).sum()), n=len(d),
            wilcoxon_p=float(w.pvalue), d_z=float(d.mean() / d.std(ddof=1)))
        print(f"  {design:<11} R={R:>3}  diff {d.mean():+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
              f"  coordfree better {int((d>0).sum())}/{len(d)}  p={w.pvalue:.3f}", flush=True)

# ---------------- D. |bias| vs gain by radius ----------------
print("\n=== D. |CAMS bias| vs transfer gain, by radius ===", flush=True)
OUT["bias_corr"] = {}
for design, store in [("space_only", OUT["space_only"]), ("llto", OUT["llto"])]:
    for nm in ("full", "coordfree"):
        for R in RADII:
            ps = store[f"{nm}_R{R}"]["per_station"]
            k = sorted(ps); g = np.array([ps[t] for t in k]); ab = S.bias.reindex(k).abs().values
            sp = stats.spearmanr(ab, g); pe = stats.pearsonr(ab, g)
            OUT["bias_corr"][f"{design}_{nm}_R{R}"] = dict(
                spearman=float(sp[0]), spearman_p=float(sp[1]), pearson=float(pe[0]), pearson_p=float(pe[1]))
        row = " ".join(f"R{R}:{OUT['bias_corr'][f'{design}_{nm}_R{R}']['spearman']:+.2f}"
                       f"({OUT['bias_corr'][f'{design}_{nm}_R{R}']['spearman_p']:.3f})" for R in RADII)
        print(f"  {design:<11} {nm:<10} Spearman  {row}", flush=True)

# ---------------- F. placebo: permuted coordinates ----------------
print("\n=== F. placebo (station coordinates permuted among stations) ===", flush=True)
OUT["placebo"] = []
ids = sorted(PM10C)
for rep in range(3):
    perm = list(np.random.default_rng(100 + rep).permutation(ids))
    fk = {}
    for a, b in zip(ids, perm):
        f = DATA[a][3].copy(); f["lat"] = PM10C[b][0]; f["lon"] = PM10C[b][1]; fk[a] = f
    s = infer(run(0, FULL, feats=fk))
    OUT["placebo"].append(dict(rep=rep, pooled=s["pooled"], median=s["median"], wins=s["wins"], d_z=s["d_z"]))
    print(f"  rep {rep}: pooled {s['pooled']:+7.2f}  median {s['median']:+7.2f}  wins {s['wins']}/19  d_z {s['d_z']:+.2f}", flush=True)
print(f"  (true coordinates, R=0, full: pooled {OUT['space_only']['full_R0']['pooled']:+.2f})", flush=True)

# ---------------- H. source-station bootstrap of the deterministic estimator ----------------
print("\n=== H. source-station bootstrap (replaces the seed diagnostic) ===", flush=True)
OUT["source_boot"] = {}
NB = 12
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for R in (0, 100):
        vals = []
        for b in range(NB):
            r2 = np.random.default_rng(500 + b)
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
                res[tgt] = (float(np.mean(np.abs(cams - obs))), float(np.mean(np.abs(cams + pred - obs))),
                            len(pick), len(obs))
            tg = sorted(res); raw = np.array([res[t][0] for t in tg]); tr = np.array([res[t][1] for t in tg])
            n = np.array([res[t][3] for t in tg])
            vals.append(100 * (np.average(raw, weights=n) - np.average(tr, weights=n)) / np.average(raw, weights=n))
        v = np.array(vals)
        OUT["source_boot"][f"{nm}_R{R}"] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                                lo=float(v.min()), hi=float(v.max()), n_boot=NB)
        print(f"  {nm:<10} R={R:>3}  pooled over source-bootstrap {v.mean():+7.2f} +- {v.std(ddof=1):.2f}"
              f"  [{v.min():+.1f}, {v.max():+.1f}]", flush=True)

# ---------------- G. BH correction across the audit family ----------------
fam = [(f"{d}_{nm}_R{R}", OUT[d][f"{nm}_R{R}"]["binom_p"], OUT[d][f"{nm}_R{R}"]["wilcoxon_p"])
       for d in ("space_only", "llto") for nm in ("full", "coordfree") for R in RADII]
pb = bh([x[1] for x in fam]); pw = bh([x[2] for x in fam])
OUT["bh"] = {k: dict(binom_p=b, binom_bh=float(qb), wilcoxon_p=w, wilcoxon_bh=float(qw))
             for (k, b, w), qb, qw in zip(fam, pb, pw)}
print("\n=== G. BH correction within the audit family (16 tests) ===", flush=True)
for (k, b, w), qb, qw in zip(fam, pb, pw):
    print(f"  {k:<24} binom {b:.4f} -> {qb:.4f}   wilcoxon {w:.4f} -> {qw:.4f}", flush=True)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v3.json"), "w"), indent=1)
print("\nSAVED audit_v3.json", flush=True)

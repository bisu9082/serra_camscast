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
import json, itertools, sys
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


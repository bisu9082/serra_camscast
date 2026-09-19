"""Reviewer demands 5 and 8 of the second panel.

  R. capacity / learner sweep: is the audit's verdict a property of one tuned GBM?
     -> ridge, GBM depth 2, GBM depth 6 (reported), random forest
        crossed with {full, coordfree} x {R=0, R=100} under the LLTO design.
  S. NNDM-LOO nearest-neighbour distance matching (Mila et al. 2022) and
     AOA / dissimilarity index (Meyer & Pebesma 2021), computed on the same data,
     so the buffer ladder can be compared against the instruments the literature
     actually recommends.

Writes audit_v5.json.  Deterministic: every learner has early stopping off and a
fixed random_state; RF uses n_jobs=1 so the result does not depend on thread count.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loso_v2 import PM10C, DATA, hav, FULL, NOXY
from auditlib import sources, GUARD, NBLK

OUT = {}
IDS = sorted(PM10C)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- learners
def make(name):
    if name == "gbm_d6":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_depth=6, random_state=0,
                                             early_stopping=False)
    if name == "gbm_d2":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_depth=2, random_state=0,
                                             early_stopping=False)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                     random_state=0, n_jobs=1)
    if name == "ridge":
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 13)))
    raise ValueError(name)


def fit_predict_L(tgt, src, cols, learner):
    """Time-blocked prediction for one target, mirroring auditlib.fit_predict."""
    dft = DATA[tgt][0]
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
            Xtr.append(DATA[j][3][cols][keep]); ytr.append(dj["resid"].values[keep])
        if not Xtr:
            continue
        pred[te] = make(learner).fit(pd.concat(Xtr), np.concatenate(ytr)).predict(
            DATA[tgt][3][cols][te])
    return pred


def gains(cols, learner, R):
    out = {}
    for tgt in PM10C:
        src = sources(tgt, R, "dist")
        if len(src) < 4:
            continue
        pred = fit_predict_L(tgt, src, cols, learner)
        dft = DATA[tgt][0]
        obs, cams = dft["obs"].values, dft["cams"].values
        ok = np.isfinite(pred)
        raw = float(np.mean(np.abs(cams[ok] - obs[ok])))
        tr = float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok])))
        out[tgt] = (raw, tr, int(ok.sum()))
    tg = sorted(out)
    raw = np.array([out[t][0] for t in tg]); tr = np.array([out[t][1] for t in tg])
    n = np.array([out[t][2] for t in tg])
    g = 100 * (raw - tr) / raw
    pooled = 100 * (np.average(raw, weights=n) - np.average(tr, weights=n)) / np.average(raw, weights=n)
    # station bootstrap of the median
    bs = [np.median(g[RNG.integers(0, len(g), len(g))]) for _ in range(2000)]
    return dict(pooled=float(pooled), median=float(np.median(g)),
                median_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                wins=int((g > 0).sum()), n=len(g),
                worst3=[float(x) for x in np.sort(g)[:3]],
                per_station={str(t): round(float(x), 2) for t, x in zip(tg, g)})


import os
_prev = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v5.json"))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v5.json")) else {}
OUT["learners"] = _prev.get("learners", {})
for L in ["gbm_d6", "gbm_d2", "rf", "ridge"]:
    for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
        for R in (0, 100):
            k = f"{L}_{nm}_R{R}"
            if k in OUT["learners"]:
                continue
            OUT["learners"][k] = gains(cols, L, R)
            r = OUT["learners"][k]
            print(f"[{k:22s}] pooled {r['pooled']:+7.2f}  median {r['median']:+6.2f} "
                  f"[{r['median_ci'][0]:+6.2f},{r['median_ci'][1]:+6.2f}]  wins {r['wins']}/{r['n']} "
                  f"  worst {r['worst3'][0]:+.1f}", flush=True)
            json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v5.json"), "w"), indent=1)


# ---------------------------------------------------- NNDM-LOO and AOA
# Geographic NN distances. For a cold-start target the "prediction point" is the
# target station and the sample-to-prediction distance is its distance to the
# nearest *source* station under the buffer.
def nn_geo(tgt, R):
    src = sources(tgt, R, "dist")
    if not src:
        return np.nan, []
    d = [hav(PM10C[tgt], PM10C[j]) for j in src]
    # within-source nearest-neighbour distances (the CV distance distribution)
    w = []
    for a in src:
        dd = [hav(PM10C[a], PM10C[b]) for b in src if b != a]
        if dd:
            w.append(min(dd))
    return float(min(d)), w


nndm = {}
for R in (0, 25, 50, 100):
    s2p, cv = [], []
    for tgt in PM10C:
        m, w = nn_geo(tgt, R)
        if np.isfinite(m):
            s2p.append(m); cv += w
    s2p = np.array(s2p); cv = np.array(cv)
    # Wasserstein-1 between the two ECDFs, the quantity kNNDM minimises
    from scipy.stats import wasserstein_distance
    nndm[f"R{R}"] = dict(
        n_targets=len(s2p),
        sample_to_prediction_median=float(np.median(s2p)),
        sample_to_prediction_iqr=[float(np.percentile(s2p, 25)), float(np.percentile(s2p, 75))],
        cv_nn_median=float(np.median(cv)),
        cv_nn_iqr=[float(np.percentile(cv, 25)), float(np.percentile(cv, 75))],
        W1_km=float(wasserstein_distance(cv, s2p)))
    r = nndm[f"R{R}"]
    print(f"[NNDM R={R:3d}] sample-to-prediction NN {r['sample_to_prediction_median']:6.1f} km "
          f"| CV NN {r['cv_nn_median']:6.1f} km | W1 {r['W1_km']:6.1f} km", flush=True)
OUT["nndm"] = nndm

# AOA: feature-space dissimilarity index (Meyer & Pebesma 2021).
# DI(target) = min over training points of scaled feature distance, normalised by
# the mean of the training-set nearest-neighbour scaled distances.
# Threshold = 1.5 x IQR above the .75 quantile of the training DI (their rule).
def aoa(cols, R, nsub=4000):
    rows = {}
    for tgt in PM10C:
        src = sources(tgt, R, "dist")
        if len(src) < 4:
            continue
        Xtr = pd.concat([DATA[j][3][cols] for j in src]).values
        Xte = DATA[tgt][3][cols].values
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        A = (Xtr - mu) / sd
        B = (Xte - mu) / sd
        idx = RNG.choice(len(A), min(nsub, len(A)), replace=False)
        As = A[idx]
        # training nearest-neighbour distances (excluding self)
        d_tr = []
        for i in range(0, len(As), 500):
            blk = As[i:i + 500]
            dd = np.sqrt(((blk[:, None, :] - As[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dd[:, i:i + len(blk)], np.inf)
            d_tr.append(dd.min(1))
        d_tr = np.concatenate(d_tr)
        dbar = float(d_tr.mean())
        di_tr = d_tr / dbar
        thr = float(np.percentile(di_tr, 75) + 1.5 * (np.percentile(di_tr, 75) - np.percentile(di_tr, 25)))
        # target DI
        jdx = RNG.choice(len(B), min(1500, len(B)), replace=False)
        Bs = B[jdx]
        d_te = []
        for i in range(0, len(Bs), 500):
            blk = Bs[i:i + 500]
            dd = np.sqrt(((blk[:, None, :] - As[None, :, :]) ** 2).sum(-1))
            d_te.append(dd.min(1))
        d_te = np.concatenate(d_te) / dbar
        rows[str(tgt)] = dict(di_median=float(np.median(d_te)),
                              frac_outside=float(np.mean(d_te > thr)),
                              threshold=thr)
    fo = np.array([v["frac_outside"] for v in rows.values()])
    dm = np.array([v["di_median"] for v in rows.values()])
    return dict(per_station=rows,
                median_DI=float(np.median(dm)),
                mean_frac_outside_AOA=float(fo.mean()),
                n_stations_mostly_outside=int((fo > 0.5).sum()),
                n=len(fo))


OUT["aoa"] = {}
for nm, cols in [("full", FULL), ("coordfree", NOXY)]:
    for R in (0, 100):
        k = f"{nm}_R{R}"
        OUT["aoa"][k] = aoa(cols, R)
        r = OUT["aoa"][k]
        print(f"[AOA {k:14s}] median DI {r['median_DI']:.2f} | mean frac outside AOA "
              f"{100*r['mean_frac_outside_AOA']:5.1f}% | stations mostly outside "
              f"{r['n_stations_mostly_outside']}/{r['n']}", flush=True)
        json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v5.json"), "w"), indent=1)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v5.json"), "w"), indent=1)
print("DONE", flush=True)

"""audit_v6 — the rebuilt audit, addressing every defect the GATE-8 panel verified.

What changes relative to audit_v3:

  1. PRIMARY ESTIMAND, declared before looking at any result:
        delta = log(MAE_raw / MAE_transfer)
     positive means the transferred correction reduces error. This is symmetric:
     halving the error gives +0.693, doubling it gives -0.693, so a station that
     fails badly can no longer dominate an aggregate the way an unbounded-below
     percentage does, and pooled and median stop disagreeing for arithmetic
     reasons. Percentage gain is retained as a secondary, reported quantity so
     the earlier tables remain comparable.

  2. THE THREE-FOLD CONTROL IS ACTUALLY RUN JOINTLY. audit_v3 ran the
     leave-one-grid-cell-out stage only at R = 0. Here every radius is run under
     buffer alone, buffer + time, and buffer + time + cell, so the design the
     manuscript describes is the design that was executed.

  3. INFERENCE UNIT IS THE GRID CELL, NOT THE STATION. 16 of the 19 stations
     share a 0.75-degree CAMS cell with at least one other and the same-cell
     pairs carry an identical covariate series, so stations are not independent.
     Confidence intervals come from a block bootstrap over the 9 cells;
     the station bootstrap is retained alongside for comparison.

  4. BOTH TESTS ARE REPORTED. The exact sign test and the Wilcoxon signed-rank
     test are computed for every cell and both are Benjamini-Hochberg corrected
     across the same family. Neither is selected after the fact.

Writes audit_v6.json.
"""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats
from loso_v2 import PM10C, DATA, hav, FULL, NOXY
from auditlib import gbm, GUARD, NBLK

RADII = (0, 25, 50, 100)
DESIGNS = ("buffer", "buffer_time", "buffer_time_cell")
FEATS = (("full", FULL), ("coordfree", NOXY))
NBOOT = 2000
rng = np.random.default_rng(0)

CELL = {i: (round(la / 0.75), round(lo / 0.75)) for i, (la, lo) in PM10C.items()}
CELLS = sorted(set(CELL.values()))
CELL_IDX = {c: k for k, c in enumerate(CELLS)}
print(f"{len(PM10C)} stations in {len(CELLS)} grid cells; "
      f"{sum(1 for c in CELLS if sum(CELL[i] == c for i in PM10C) > 1)} cells hold more than one",
      flush=True)


def sources(tgt, R, design):
    """Source stations surviving the controls for this target."""
    out = []
    for j in PM10C:
        if j == tgt:
            continue
        if hav(PM10C[j], PM10C[tgt]) <= R:
            continue
        if design == "buffer_time_cell" and CELL[j] == CELL[tgt]:
            continue
        out.append(j)
    return out


def fit_predict(tgt, src, cols, time_block):
    dft = DATA[tgt][0]
    if not time_block:
        Xtr = pd.concat([DATA[j][3][cols] for j in src])
        ytr = np.concatenate([DATA[j][0]["resid"].values for j in src])
        return gbm().fit(Xtr, ytr).predict(DATA[tgt][3][cols])
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
        pred[te] = gbm().fit(pd.concat(Xtr), np.concatenate(ytr)).predict(DATA[tgt][3][cols][te])
    return pred


def run(R, cols, design):
    time_block = design in ("buffer_time", "buffer_time_cell")
    per = {}
    for tgt in PM10C:
        src = sources(tgt, R, design)
        if len(src) < 4:
            continue
        pred = fit_predict(tgt, src, cols, time_block)
        dft = DATA[tgt][0]
        obs, cams = dft["obs"].values, dft["cams"].values
        ok = np.isfinite(pred)
        raw = float(np.mean(np.abs(cams[ok] - obs[ok])))
        tr = float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok])))
        per[tgt] = dict(raw=raw, transfer=tr, n=int(ok.sum()), n_src=len(src))
    return per


def summarise(per):
    tg = sorted(per)
    raw = np.array([per[t]["raw"] for t in tg])
    tr = np.array([per[t]["transfer"] for t in tg])
    n = np.array([per[t]["n"] for t in tg])
    delta = np.log(raw / tr)                      # PRIMARY: symmetric log error ratio
    pct = 100 * (raw - tr) / raw                  # secondary, for comparability
    cellid = np.array([CELL_IDX[CELL[t]] for t in tg])

    pooled_delta = float(np.log(np.average(raw, weights=n) / np.average(tr, weights=n)))
    pooled_pct = float(100 * (np.average(raw, weights=n) - np.average(tr, weights=n))
                       / np.average(raw, weights=n))

    def boot(idx_sets):
        md, pd_, pp = [], [], []
        for i in idx_sets:
            md.append(np.median(delta[i]))
            pd_.append(np.log(np.average(raw[i], weights=n[i]) / np.average(tr[i], weights=n[i])))
            pp.append(np.median(pct[i]))
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        return q(md), q(pd_), q(pp)

    # station bootstrap (as before, for comparison)
    st_idx = [rng.integers(0, len(tg), len(tg)) for _ in range(NBOOT)]
    st_md, st_pd, st_pp = boot(st_idx)

    # grid-cell block bootstrap: resample cells, take every station in a drawn cell
    members = {c: np.where(cellid == c)[0] for c in np.unique(cellid)}
    ucells = np.array(sorted(members))
    cl_idx = []
    for _ in range(NBOOT):
        drawn = rng.integers(0, len(ucells), len(ucells))
        take = np.concatenate([members[ucells[d]] for d in drawn])
        cl_idx.append(take)
    cl_md, cl_pd, cl_pp = boot(cl_idx)

    wins = int((delta > 0).sum())
    binom = float(stats.binomtest(wins, len(delta), 0.5).pvalue)
    try:
        wil = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
    except ValueError:
        wil = float("nan")
    dz = float(np.mean(delta) / np.std(delta, ddof=1))

    return dict(
        n_targets=len(tg),
        n_src_mean=float(np.mean([per[t]["n_src"] for t in tg])),
        n_src_min=int(min(per[t]["n_src"] for t in tg)),
        n_cells=int(len(np.unique(cellid))),
        delta_median=float(np.median(delta)),
        delta_median_ci_station=st_md, delta_median_ci_cell=cl_md,
        delta_pooled=pooled_delta,
        delta_pooled_ci_station=st_pd, delta_pooled_ci_cell=cl_pd,
        delta_mean=float(np.mean(delta)),
        pct_median=float(np.median(pct)),
        pct_median_ci_station=st_pp, pct_median_ci_cell=cl_pp,
        pct_pooled=pooled_pct,
        wins=wins, binom_p=binom, wilcoxon_p=wil, d_z=dz,
        delta_min=float(delta.min()), delta_max=float(delta.max()),
        per_station={str(t): dict(delta=round(float(d), 4), pct=round(float(p), 2),
                                  raw=round(float(r), 2), transfer=round(float(x), 2),
                                  n=int(nn), cell=f"{CELL[t][0]}_{CELL[t][1]}")
                     for t, d, p, r, x, nn in zip(tg, delta, pct, raw, tr, n)})


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run_min = 1.0
    for k in range(m - 1, -1, -1):
        run_min = min(run_min, p[o[k]] * m / (k + 1)); q[o[k]] = run_min
    return np.minimum(q, 1.0)


import os
OUTFILE = os.environ.get("CAMSCAST_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v6.json"))
_prev = json.load(open(OUTFILE)) if os.path.exists(OUTFILE) else {}
OUT = {"meta": dict(
    primary_estimand="delta = log(MAE_raw / MAE_transfer); positive means the correction helps",
    secondary_estimand="pct = 100*(MAE_raw - MAE_transfer)/MAE_raw",
    inference_unit="grid cell (block bootstrap over the 9 CAMS cells); station bootstrap reported alongside",
    tests="exact sign test and Wilcoxon signed-rank, both reported, both BH-corrected across the same family",
    n_boot=NBOOT, radii=list(RADII), designs=list(DESIGNS),
    n_stations=len(PM10C), n_cells=len(CELLS))}

for design in DESIGNS:
    for nm, cols in FEATS:
        for R in RADII:
            key = f"{design}_{nm}_R{R}"
            if key in _prev:
                OUT[key] = _prev[key]
                print(f"[{key:30s}] resumed from cache", flush=True)
                continue
            per = run(R, cols, design)
            OUT[key] = summarise(per)
            s = OUT[key]
            print(f"[{key:30s}] src {s['n_src_mean']:5.1f} | "
                  f"delta med {s['delta_median']:+.3f} cell-CI [{s['delta_median_ci_cell'][0]:+.3f},{s['delta_median_ci_cell'][1]:+.3f}] "
                  f"| pooled {s['delta_pooled']:+.3f} | win {s['wins']}/{s['n_targets']} "
                  f"| sign p {s['binom_p']:.4f} wilcox p {s['wilcoxon_p']:.4f}", flush=True)
            json.dump(OUT, open(OUTFILE, "w"), indent=1)

# BH across the full family, both tests, declared in advance as the 24-cell family
fam = [k for k in OUT if k != "meta"]
qb = bh([OUT[k]["binom_p"] for k in fam])
qw = bh([OUT[k]["wilcoxon_p"] for k in fam])
OUT["bh"] = {k: dict(binom_p=OUT[k]["binom_p"], binom_bh=float(a),
                     wilcoxon_p=OUT[k]["wilcoxon_p"], wilcoxon_bh=float(b))
             for k, a, b in zip(fam, qb, qw)}
OUT["bh_family_size"] = len(fam)
json.dump(OUT, open(OUTFILE, "w"), indent=1)

print("\n=== BH across the %d-cell family (both tests) ===" % len(fam), flush=True)
for k in fam:
    r = OUT["bh"][k]
    print(f"  {k:30s} sign q {r['binom_bh']:.4f} {'PASS' if r['binom_bh']<0.05 else '    '} | "
          f"wilcoxon q {r['wilcoxon_bh']:.4f} {'PASS' if r['wilcoxon_bh']<0.05 else ''}", flush=True)
print("DONE", flush=True)

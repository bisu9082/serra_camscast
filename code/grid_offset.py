"""Does the leave-one-grid-cell-out stage depend on where the grid origin sits?

The third control withholds every source station that shares the target's CAMS cell,
because such stations carry an identical covariate series. Which stations share a cell
is a property of the 0.75-degree grid, and the grid's origin is a property of the CAMS
product rather than of the atmosphere. A reviewer is right to ask whether the control
is doing something real or is an artefact of where the cell boundaries happen to fall
relative to this particular cluster of stations.

This script shifts the grid origin over a lattice of offsets and reports, for each
offset, both the design consequence (how many stations share a cell, how many sources
survive) and the outcome (the median log error ratio and the win count) at the two
radii that bracket the argument: R = 0, where every non-cellmate neighbour is
available, and R = 100 km, where the correction must extrapolate.

An offset-dependent verdict would mean the cell stage is picking up the grid rather
than covariate sharing. A stable verdict means the opposite.

Writes grid_offset.json. Run with OMP_NUM_THREADS=1.
"""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, os, time
import numpy as np, pandas as pd
from loso_v2 import PM10C, DATA, hav, FULL
from auditlib import gbm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(HERE, "grid_offset.json"))
GUARD = pd.Timedelta(days=float(os.environ.get("GUARD_DAYS", 10)))
NBLK = int(os.environ.get("NBLK", 6))
RADII = tuple(int(x) for x in os.environ.get("RADII", "0,100").split(","))
STEP = 0.75
# offsets as fractions of a cell, applied to both axes
OFFSETS = [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25),
           (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)]


def cells_for(off):
    dy, dx = off
    return {i: (int(np.floor(la / STEP + dy)), int(np.floor(lo / STEP + dx)))
            for i, (la, lo) in PM10C.items()}


def block_edges(idx):
    srt = idx.sort_values()
    pos = np.linspace(0, len(srt) - 1, NBLK + 1).round().astype(int)
    return pd.DatetimeIndex(pd.Series(srt[pos]).drop_duplicates().values)


def fit_predict(tgt, src):
    dft = DATA[tgt][0]
    edges = block_edges(dft.index)
    pred = np.full(len(dft), np.nan)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        last = b == len(edges) - 2
        te = (dft.index >= lo) & ((dft.index <= hi) if last else (dft.index < hi))
        if te.sum() == 0:
            continue
        Xtr, ytr = [], []
        for j in src:
            dj = DATA[j][0]
            keep = (dj.index < lo - GUARD) | (dj.index > hi + GUARD)
            if keep.sum() == 0:
                continue
            Xtr.append(DATA[j][3][FULL][keep]); ytr.append(dj["resid"].values[keep])
        if not Xtr:
            continue
        pred[te] = gbm().fit(pd.concat(Xtr), np.concatenate(ytr)).predict(DATA[tgt][3][FULL][te])
    return pred


out = json.load(open(OUT)) if os.path.exists(OUT) else {}
out.setdefault("_meta", dict(step_deg=STEP, radii=list(RADII), guard_days=GUARD.days,
                             feature_set="coordinate-bearing (the specification the cell "
                                         "stage is meant to protect)"))
t0 = time.time()

for off in OFFSETS:
    CELL = cells_for(off)
    shared = sum(1 for a in PM10C for b in PM10C
                 if a < b and CELL[a] == CELL[b])
    n_cells = len(set(CELL.values()))
    tag = f"off_{off[0]:.2f}_{off[1]:.2f}"
    rec = out.setdefault(tag, dict(offset=list(off), n_cells=n_cells,
                                   same_cell_pairs=shared))
    for R in RADII:
        key = f"R{R}"
        if key in rec:
            continue
        d, ns = [], []
        for tgt in PM10C:
            src = [j for j in PM10C
                   if j != tgt and hav(PM10C[j], PM10C[tgt]) > R and CELL[j] != CELL[tgt]]
            if len(src) < 4:
                continue
            pred = fit_predict(tgt, src)
            dft = DATA[tgt][0]
            obs, cams = dft["obs"].values, dft["cams"].values
            ok = np.isfinite(pred)
            if ok.sum() == 0:
                continue
            raw = float(np.mean(np.abs(cams[ok] - obs[ok])))
            tr = float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok])))
            d.append(float(np.log(raw / tr))); ns.append(len(src))
        d = np.array(d)
        rec[key] = dict(n_targets=int(len(d)), n_src_mean=float(np.mean(ns)),
                        delta_median=float(np.median(d)), wins=int((d > 0).sum()))
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"[{tag}] cells {n_cells:2d} same-cell pairs {shared:2d} | {key}: "
              f"src {rec[key]['n_src_mean']:5.2f}  median {rec[key]['delta_median']:+.4f}  "
              f"wins {rec[key]['wins']}/{rec[key]['n_targets']}  "
              f"| {(time.time()-t0)/60:.1f} min", flush=True)

# stability summary across offsets
for R in RADII:
    key = f"R{R}"
    med = [v[key]["delta_median"] for k, v in out.items() if k != "_meta" and key in v]
    win = [v[key]["wins"] for k, v in out.items() if k != "_meta" and key in v]
    out["_meta"][f"summary_{key}"] = dict(
        n_offsets=len(med), median_min=float(np.min(med)), median_max=float(np.max(med)),
        median_range=float(np.max(med) - np.min(med)),
        wins_min=int(np.min(win)), wins_max=int(np.max(win)))
    print(f"{key}: across {len(med)} grid offsets, median delta spans "
          f"{np.min(med):+.4f} to {np.max(med):+.4f}, wins {np.min(win)}-{np.max(win)} of 19", flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("saved", os.path.basename(OUT))

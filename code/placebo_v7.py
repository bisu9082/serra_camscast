"""Coordinate-permutation placebo under the primary specification, at every radius.

The question a placebo answers here is what the latitude and longitude features are
doing. If they carry transferable structure, assigning each station a different
station's coordinates should destroy the transfer gain; if they are a lookup key,
scrambling them should not matter; and if they support interpolation but not
extrapolation, the true-minus-scrambled difference should change sign as the buffer
widens. Reading that requires a permutation null with enough draws to resolve a
p-value below 0.05, which the earlier 30-draw version could not: its smallest
attainable p was 1/31 = 0.032.

This script runs the placebo under the specification the manuscript reports --- the
flat-line quality control at K=6, quantile time blocks, a ten-day guard band, and the
full buffer + time + cell control --- on the log error ratio that is the pre-declared
estimand.

It is resumable and parallel. Each permutation is written to the output as it
finishes, so the run can be stopped and restarted, and several workers can share the
work by taking different residues of the permutation index.

Environment:
  CAMSCAST_DATA  directory of aligned records (default: aligned_qc6 beside this file)
  CAMSCAST_OUT   output json (default: placebo_v7.json beside this file)
  GUARD_DAYS     guard band in days, default 10
  NBLK           number of time blocks, default 6
  RADII          comma-separated radii in km, default 0,25,50,75,100
  NPERM          permutations per radius, default 199
  WORKER, NWORK  this worker's residue and the number of workers, default 0 and 1

Set OMP_NUM_THREADS=1 when running more than one worker. The gradient booster uses
OpenMP internally, and two workers each claiming every core oversubscribe the machine
badly enough to slow each permutation by an order of magnitude: on a two-core host a
permutation takes 53 s single-threaded and around 600 s with the default threading.
"""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, os, time
import numpy as np, pandas as pd
from loso_v2 import PM10C, DATA, hav, FULL
from auditlib import gbm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.environ.get("CAMSCAST_OUT", os.path.join(HERE, "placebo_v7.json"))
GUARD = pd.Timedelta(days=float(os.environ.get("GUARD_DAYS", 10)))
NBLK = int(os.environ.get("NBLK", 6))
RADII = tuple(int(x) for x in os.environ.get("RADII", "0,25,50,75,100").split(","))
NPERM = int(os.environ.get("NPERM", 199))
WORKER = int(os.environ.get("WORKER", 0))
NWORK = int(os.environ.get("NWORK", 1))

CELL = {i: (round(la / 0.75), round(lo / 0.75)) for i, (la, lo) in PM10C.items()}
IDS = sorted(PM10C)
LOCK = OUTFILE + f".w{WORKER}"          # each worker writes its own shard

print(f"placebo_v7 | {len(PM10C)} stations | radii {RADII} | {NPERM} permutations "
      f"| guard {GUARD.days}d | worker {WORKER}/{NWORK} | data "
      f"{os.environ.get('CAMSCAST_DATA', '(default)')}", flush=True)


def block_edges(idx):
    srt = idx.sort_values()
    pos = np.linspace(0, len(srt) - 1, NBLK + 1).round().astype(int)
    return pd.DatetimeIndex(pd.Series(srt[pos]).drop_duplicates().values)


def sources(tgt, R):
    """Full control: distance buffer, and the target's own CAMS cell withheld."""
    return [j for j in PM10C
            if j != tgt and hav(PM10C[j], PM10C[tgt]) > R and CELL[j] != CELL[tgt]]


def fit_predict(tgt, src, feats):
    """feats maps a station id to its feature frame, so a permutation is applied by
    passing frames whose lat/lon columns have been reassigned."""
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
            Xtr.append(feats[j][FULL][keep]); ytr.append(dj["resid"].values[keep])
        if not Xtr:
            continue
        pred[te] = gbm().fit(pd.concat(Xtr), np.concatenate(ytr)).predict(feats[tgt][FULL][te])
    return pred


def statistic(R, feats):
    """Median and pooled log error ratio across the targets that survive the control."""
    d, raw_all, tr_all, n_all = [], [], [], []
    for tgt in PM10C:
        src = sources(tgt, R)
        if len(src) < 4:
            continue
        pred = fit_predict(tgt, src, feats)
        dft = DATA[tgt][0]
        obs, cams = dft["obs"].values, dft["cams"].values
        ok = np.isfinite(pred)
        if ok.sum() == 0:
            continue
        raw = float(np.mean(np.abs(cams[ok] - obs[ok])))
        tr = float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok])))
        d.append(np.log(raw / tr)); raw_all.append(raw); tr_all.append(tr); n_all.append(int(ok.sum()))
    d = np.array(d); w = np.array(n_all, float)
    return dict(delta_median=float(np.median(d)),
                delta_pooled=float(np.log(np.average(raw_all, weights=w)
                                          / np.average(tr_all, weights=w))),
                wins=int((d > 0).sum()), n_targets=int(len(d)))


def permuted_features(rep):
    """Assign each station another station's coordinates. rep=-1 returns the true
    assignment, so the observed statistic goes through the identical code path."""
    if rep < 0:
        return {i: DATA[i][3] for i in PM10C}
    order = list(np.random.default_rng(20000 + rep).permutation(IDS))
    out = {}
    for a, b in zip(IDS, order):
        f = DATA[a][3].copy()
        f["lat"] = PM10C[b][0]; f["lon"] = PM10C[b][1]
        out[a] = f
    return out


shard = json.load(open(LOCK)) if os.path.exists(LOCK) else {}
t_start = time.time()
done = 0

for R in RADII:
    key = f"R{R}"
    rec = shard.setdefault(key, {"observed": None, "perm": {}})
    if rec["observed"] is None and WORKER == 0:
        rec["observed"] = statistic(R, permuted_features(-1))
        print(f"[R={R}] observed  median {rec['observed']['delta_median']:+.4f}  "
              f"pooled {rec['observed']['delta_pooled']:+.4f}  "
              f"wins {rec['observed']['wins']}/{rec['observed']['n_targets']}", flush=True)
        json.dump(shard, open(LOCK, "w"), indent=1)
    for rep in range(NPERM):
        if rep % NWORK != WORKER or str(rep) in rec["perm"]:
            continue
        s = statistic(R, permuted_features(rep))
        rec["perm"][str(rep)] = s
        done += 1
        json.dump(shard, open(LOCK, "w"), indent=1)
        el = time.time() - t_start
        print(f"[R={R}] perm {rep:4d}  median {s['delta_median']:+.4f}  "
              f"pooled {s['delta_pooled']:+.4f}  wins {s['wins']}  "
              f"| {done} done, {el/done:.0f}s each, {el/3600:.2f}h elapsed", flush=True)

print("WORKER DONE", flush=True)

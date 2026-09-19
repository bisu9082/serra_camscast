"""audit_v7 — the audit with the four defects the GATE-8 panel verified in v6 repaired.

Changes relative to audit_v6:

  1. THE PAIRED COMPARISON IS A FIRST-CLASS OUTPUT. v6 selected between the two
     feature sets by comparing each one's significance against zero, which is the
     error Gelman and Stern (2006) name: the difference between "significant" and
     "not significant" is not itself significant. Here the paired difference
     delta_coordfree - delta_full is tested directly, under all three units of
     inference, and it is that comparison the manuscript should report.

  2. TIME BLOCKS ARE QUANTILES OF THE TARGET'S OWN TIMESTAMPS, not calendar-equal
     intervals. The record has months with no data at all, so calendar blocks put
     9, 4 and 7 points into three of the six folds: a nominal six-fold control that
     is really three-fold. Quantile blocks hold equal counts by construction.

  3. THE GUARD BAND IS SET FROM THE MEASURED RESIDUAL AUTOCORRELATION rather than
     from an assumed episode duration. The residual autocorrelation of this network
     falls below 1/e at one day but plateaus near +0.2 out to a week and reaches
     +0.08 only at ten days, so the three-day guard used earlier leaves appreciable
     correlation across the block boundary. The default here is ten days, with
     three and seven reported as sensitivity.

  4. THE DATA DIRECTORY IS A PARAMETER, so the flat-line QC variants can be run
     through the identical pipeline.

Environment:
  CAMSCAST_DATA   directory of aligned records (default: this script's directory)
  CAMSCAST_OUT    output json
  GUARD_DAYS      guard band, default 10
  NBLK            number of time blocks, default 6
  BLOCKS          "quantile" (default) or "calendar"
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from loso_v2 import PM10C, DATA, hav, FULL, NOXY
from auditlib import gbm

RADII = (0, 25, 50, 100)
DESIGNS = ("buffer", "buffer_time", "buffer_time_cell")
FEATS = (("full", FULL), ("coordfree", NOXY))
NBOOT = 2000
GUARD = pd.Timedelta(days=float(os.environ.get("GUARD_DAYS", 10)))
NBLK = int(os.environ.get("NBLK", 6))
BLOCKS = os.environ.get("BLOCKS", "quantile")
OUTFILE = os.environ.get("CAMSCAST_OUT",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_v7.json"))
rng = np.random.default_rng(0)

CELL = {i: (round(la / 0.75), round(lo / 0.75)) for i, (la, lo) in PM10C.items()}
CELLS = sorted(set(CELL.values()))
CELL_IDX = {c: k for k, c in enumerate(CELLS)}
print(f"{len(PM10C)} stations in {len(CELLS)} cells | blocks={BLOCKS} nblk={NBLK} "
      f"guard={GUARD.days}d | data={os.environ.get('CAMSCAST_DATA','(default)')}", flush=True)


def block_edges(idx):
    """Boundaries of the NBLK time blocks for one target's timestamps.

    Quantile edges are taken as actual timestamps at equal-count positions rather
    than by converting to integers and back: the records carry microsecond
    resolution, and a round trip through a nanosecond constructor silently moves
    every edge to 1970.
    """
    if BLOCKS == "calendar":
        return pd.date_range(idx.min(), idx.max(), periods=NBLK + 1)
    srt = idx.sort_values()
    pos = np.linspace(0, len(srt) - 1, NBLK + 1).round().astype(int)
    return pd.DatetimeIndex(pd.Series(srt[pos]).drop_duplicates().values)


def sources(tgt, R, design):
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
        return gbm().fit(Xtr, ytr).predict(DATA[tgt][3][cols]), []
    edges = block_edges(dft.index)
    pred = np.full(len(dft), np.nan)
    fold_n = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        last = b == len(edges) - 2
        te = (dft.index >= lo) & ((dft.index <= hi) if last else (dft.index < hi))
        fold_n.append(int(te.sum()))
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
    return pred, fold_n


def run(R, cols, design):
    tb = design in ("buffer_time", "buffer_time_cell")
    per, folds, skipped = {}, [], []
    for tgt in PM10C:
        src = sources(tgt, R, design)
        if len(src) < 4:
            continue
        pred, fn = fit_predict(tgt, src, cols, tb)
        if fn:
            folds.append(fn)
        dft = DATA[tgt][0]
        obs, cams = dft["obs"].values, dft["cams"].values
        ok = np.isfinite(pred)
        if ok.sum() == 0:
            skipped.append(int(tgt))
            continue
        per[tgt] = dict(raw=float(np.mean(np.abs(cams[ok] - obs[ok]))),
                        transfer=float(np.mean(np.abs(cams[ok] + pred[ok] - obs[ok]))),
                        n=int(ok.sum()), n_src=len(src))
    if skipped:
        print(f"    (no predictable fold for stations {skipped})", flush=True)
    return per, folds


def cell_of(tg):
    return np.array([CELL_IDX[CELL[t]] for t in tg])


def boot_idx(n, cellid):
    """Station-resample and cell-block-resample index sets."""
    st = [rng.integers(0, n, n) for _ in range(NBOOT)]
    members = {c: np.where(cellid == c)[0] for c in np.unique(cellid)}
    uc = np.array(sorted(members))
    cl = [np.concatenate([members[uc[k]] for k in rng.integers(0, len(uc), len(uc))])
          for _ in range(NBOOT)]
    return st, cl


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarise(per, folds):
    tg = sorted(per)
    raw = np.array([per[t]["raw"] for t in tg]); tr = np.array([per[t]["transfer"] for t in tg])
    n = np.array([per[t]["n"] for t in tg]); cid = cell_of(tg)
    delta = np.log(raw / tr)
    st, cl = boot_idx(len(tg), cid)
    pooled = float(np.log(np.average(raw, weights=n) / np.average(tr, weights=n)))
    md_st = ci([np.median(delta[i]) for i in st])
    md_cl = ci([np.median(delta[i]) for i in cl])
    pl_cl = ci([np.log(np.average(raw[i], weights=n[i]) / np.average(tr[i], weights=n[i])) for i in cl])
    cellmed = np.array([np.median(delta[cid == c]) for c in np.unique(cid)])
    wins = int((delta > 0).sum())
    out = dict(
        n_targets=len(tg), n_src_mean=float(np.mean([per[t]["n_src"] for t in tg])),
        n_src_min=int(min(per[t]["n_src"] for t in tg)), n_cells=int(len(np.unique(cid))),
        n_points=int(n.sum()),
        delta_median=float(np.median(delta)), delta_median_ci_station=md_st,
        delta_median_ci_cell=md_cl, delta_pooled=pooled, delta_pooled_ci_cell=pl_cl,
        wins=wins, binom_p=float(stats.binomtest(wins, len(delta), 0.5).pvalue),
        wilcoxon_p=float(stats.wilcoxon(delta).pvalue),
        cell_median=float(np.median(cellmed)),
        cells_positive=int((cellmed > 0).sum()),
        cell_binom_p=float(stats.binomtest(int((cellmed > 0).sum()), len(cellmed), 0.5).pvalue),
        per_station={str(t): dict(delta=round(float(d), 4), raw=round(float(r), 2),
                                  transfer=round(float(x), 2), n=int(nn),
                                  cell=f"{CELL[t][0]}_{CELL[t][1]}")
                     for t, d, r, x, nn in zip(tg, delta, raw, tr, n)})
    if folds:
        fl = np.array([f for f in folds if len(f) == len(folds[0])])
        if fl.size:
            out["fold_n_mean"] = [float(x) for x in fl.mean(0)]
            out["fold_n_min"] = int(fl.min())
            out["fold_imbalance_ratio"] = float(fl.mean(0).max() / max(fl.mean(0).min(), 1e-9))
    return out


def paired(perA, perB):
    """delta_coordfree - delta_full on the stations both specifications share."""
    tg = sorted(set(perA) & set(perB))
    dA = np.array([np.log(perA[t]["raw"] / perA[t]["transfer"]) for t in tg])
    dB = np.array([np.log(perB[t]["raw"] / perB[t]["transfer"]) for t in tg])
    dif = dB - dA
    cid = cell_of(tg)
    st, cl = boot_idx(len(tg), cid)
    cellmed = np.array([np.median(dif[cid == c]) for c in np.unique(cid)])
    wins = int((dif > 0).sum())
    return dict(
        n=len(tg), median_diff=float(np.median(dif)), mean_diff=float(np.mean(dif)),
        coordfree_better=wins,
        station_binom_p=float(stats.binomtest(wins, len(dif), 0.5).pvalue),
        station_wilcoxon_p=float(stats.wilcoxon(dif).pvalue),
        station_ci=ci([np.median(dif[i]) for i in st]),
        cell_ci=ci([np.median(dif[i]) for i in cl]),
        cells_favouring_coordfree=int((cellmed > 0).sum()), n_cells=int(len(cellmed)),
        cell_binom_p=float(stats.binomtest(int((cellmed > 0).sum()), len(cellmed), 0.5).pvalue),
        d_z=float(np.mean(dif) / np.std(dif, ddof=1)))


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        run = min(run, p[o[k]] * m / (k + 1)); q[o[k]] = run
    return np.minimum(q, 1.0)


OUT = {"meta": dict(
    primary_estimand="delta = log(MAE_raw / MAE_transfer)",
    headline_comparison="paired difference delta_coordfree - delta_full, tested under three units of inference",
    blocks=BLOCKS, n_blocks=NBLK, guard_days=GUARD.days, n_boot=NBOOT,
    data_dir=os.environ.get("CAMSCAST_DATA", "default"),
    n_stations=len(PM10C), n_cells=len(CELLS))}

store = {}
for design in DESIGNS:
    for nm, cols in FEATS:
        for R in RADII:
            key = f"{design}_{nm}_R{R}"
            per, folds = run(R, cols, design)
            store[key] = per
            OUT[key] = summarise(per, folds)
            s = OUT[key]
            print(f"[{key:32s}] med {s['delta_median']:+.3f} cell-CI "
                  f"[{s['delta_median_ci_cell'][0]:+.3f},{s['delta_median_ci_cell'][1]:+.3f}] "
                  f"pooled {s['delta_pooled']:+.3f} win {s['wins']}/{s['n_targets']} "
                  f"cells {s['cells_positive']}/{s['n_cells']}", flush=True)
            json.dump(OUT, open(OUTFILE, "w"), indent=1)

OUT["paired"] = {}
for design in DESIGNS:
    for R in RADII:
        k = f"{design}_R{R}"
        OUT["paired"][k] = paired(store[f"{design}_full_R{R}"], store[f"{design}_coordfree_R{R}"])
        p = OUT["paired"][k]
        print(f"[paired {k:26s}] coordfree better {p['coordfree_better']}/{p['n']} "
              f"sign p {p['station_binom_p']:.3f} wilcox p {p['station_wilcoxon_p']:.3f} | "
              f"cells {p['cells_favouring_coordfree']}/{p['n_cells']} p {p['cell_binom_p']:.3f} | "
              f"cell-CI [{p['cell_ci'][0]:+.3f},{p['cell_ci'][1]:+.3f}]", flush=True)

fam = [k for k in OUT if k not in ("meta", "paired")]
OUT["bh"] = {k: dict(binom_bh=float(a), wilcoxon_bh=float(b))
             for k, a, b in zip(fam, bh([OUT[k]["binom_p"] for k in fam]),
                                bh([OUT[k]["wilcoxon_p"] for k in fam]))}
OUT["bh_paired"] = {k: float(q) for k, q in
                    zip(OUT["paired"], bh([OUT["paired"][k]["station_binom_p"] for k in OUT["paired"]]))}
json.dump(OUT, open(OUTFILE, "w"), indent=1)
print("DONE", flush=True)

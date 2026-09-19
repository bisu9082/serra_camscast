"""camscast — distance-buffered LOSO transfer, REPRODUCIBLE re-run (v2).

Fixes vs original exp_distance_loso.py:
  * HistGradientBoostingRegressor now takes random_state and early_stopping=False
    -> fully deterministic (original left random_state=None with early_stopping='auto',
       which splits off a random 10% validation set when n>10,000; that is why two
       archived runs of the identical R=0 computation disagreed: 16/19 vs 17/19).
  * Adds the coordinate-free ablation (lat/lon removed) that the manuscript claims
    but for which no code or output existed.
  * Adds station-cluster bootstrap CIs, exact Wilcoxon, Cohen's d_z, exact binomial
    win-rate test, and per-buffer source-station counts.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats

import os
# Data directory. Defaults to the directory holding this script, so a clone runs
# without editing; override with CAMSCAST_DATA to point at a QC variant.
DDIR = os.environ.get("CAMSCAST_DATA", os.path.dirname(os.path.abspath(__file__)))
PM10C = {369953:(23.6522,53.7038),369954:(24.4819,54.3691),369955:(23.0957,53.6064),
369957:(24.4889,54.3637),369958:(24.3471,54.5028),369959:(24.2258,55.7658),369960:(24.4665,55.3428),
369961:(24.4199,54.5781),369962:(23.8355,52.8103),369963:(24.4035,54.516),369965:(24.0908,52.7548),
369966:(24.3213,54.6359),369967:(23.7504,53.7452),369968:(24.219,55.7348),369969:(24.1634,55.7021),
369971:(23.5311,55.4859),369972:(24.2591,55.7048),369973:(24.0351,53.8853),369974:(24.287,54.5861)}
NAME = {369965:"Ruwais"}
FULL = ["cams","hsin","hcos","dsin","dcos","lat","lon"]
NOXY = ["cams","hsin","hcos","dsin","dcos"]

def hav(a,b):
    la1,lo1,la2,lo2 = map(np.radians,[a[0],a[1],b[0],b[1]])
    d = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(d))

def feat(df,la,lo):
    h = df.index.hour; doy = df.index.dayofyear
    return pd.DataFrame({"cams":df["cams"].values,
        "hsin":np.sin(2*np.pi*h/24),"hcos":np.cos(2*np.pi*h/24),
        "dsin":np.sin(2*np.pi*doy/365),"dcos":np.cos(2*np.pi*doy/365),
        "lat":la,"lon":lo}, index=df.index)

DATA = {}
for i,(la,lo) in PM10C.items():
    d = pd.read_csv(f"{DDIR}/aligned_pm10_{i}.csv", index_col=0, parse_dates=True).dropna()
    DATA[i] = (d, la, lo, feat(d,la,lo))

def run(R, cols, seed, early):
    """Return dict target -> (mae_raw, mae_transfer, n_src)."""
    out = {}
    for tgt in PM10C:
        src = [j for j in PM10C if j != tgt and hav(PM10C[tgt],PM10C[j]) > R]
        if len(src) < 4:
            continue
        Xtr = pd.concat([DATA[j][3][cols] for j in src])
        ytr = np.concatenate([DATA[j][0]["resid"].values for j in src])
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                          random_state=seed, early_stopping=early)
        pred = m.fit(Xtr,ytr).predict(DATA[tgt][3][cols])
        obs = DATA[tgt][0]["obs"].values; cams = DATA[tgt][0]["cams"].values
        out[tgt] = (float(np.mean(np.abs(cams-obs))),
                    float(np.mean(np.abs(cams+pred-obs))), len(src))
    return out

def infer(res, B=2000, seed=20260829):
    tg = sorted(res)
    raw = np.array([res[t][0] for t in tg]); tr = np.array([res[t][1] for t in tg])
    gain = 100*(raw-tr)/raw
    diff = raw-tr                       # positive = transfer better
    n = len(tg); rng = np.random.default_rng(seed)
    idx = rng.integers(0,n,size=(B,n))
    bm = gain[idx].mean(1); bmed = np.median(gain[idx],axis=1)
    bd = diff[idx]
    with np.errstate(invalid="ignore",divide="ignore"):
        bdz = bd.mean(1)/bd.std(ddof=1,axis=1)
    bdz = bdz[np.isfinite(bdz)]
    wins = int((gain>0).sum())
    d_z = diff.mean()/diff.std(ddof=1)
    return dict(
        n_targets=n,
        n_src_min=int(min(res[t][2] for t in tg)),
        n_src_med=float(np.median([res[t][2] for t in tg])),
        pooled_gain_pct=float(100*(raw.mean()-tr.mean())/raw.mean()),
        mean_gain_pct=float(gain.mean()),
        mean_ci=[float(np.percentile(bm,2.5)),float(np.percentile(bm,97.5))],
        median_gain_pct=float(np.median(gain)),
        median_ci=[float(np.percentile(bmed,2.5)),float(np.percentile(bmed,97.5))],
        wins=wins, binom_p=float(stats.binomtest(wins,n,0.5).pvalue),
        wilcoxon_p=float(stats.wilcoxon(raw,tr,mode="exact").pvalue),
        d_z=float(d_z),
        d_z_ci=[float(np.percentile(bdz,2.5)),float(np.percentile(bdz,97.5))],
        worst_station=int(tg[int(np.argmin(gain))]),
        worst_gain_pct=float(gain.min()),
        mean_excl_worst=float(np.delete(gain,int(np.argmin(gain))).mean()),
        per_station={int(t):round(float(g),2) for t,g in zip(tg,gain)},
    )

if __name__ == "__main__":
    RADII = [0,25,50,100]
    results = {"spec":"HistGradientBoostingRegressor(max_iter=300, lr=0.05, max_depth=6, "
                      "random_state=0, early_stopping=False)  [deterministic]",
               "primary":{}}
    for fname, cols in [("full",FULL),("coordfree",NOXY)]:
        for R in RADII:
            res = run(R, cols, seed=0, early=False)
            st = infer(res)
            results["primary"][f"{fname}_R{R}"] = st
            print(f"[{fname:9s} R={R:>3}] n={st['n_targets']} src>={st['n_src_min']} "
                  f"pooled={st['pooled_gain_pct']:+6.2f}% mean={st['mean_gain_pct']:+7.2f}% "
                  f"CI[{st['mean_ci'][0]:+.1f},{st['mean_ci'][1]:+.1f}] "
                  f"med={st['median_gain_pct']:+6.2f}% CI[{st['median_ci'][0]:+.1f},{st['median_ci'][1]:+.1f}] "
                  f"win={st['wins']}/{st['n_targets']} p={st['binom_p']:.4f} "
                  f"W p={st['wilcoxon_p']:.4f} d_z={st['d_z']:+.2f}", flush=True)
    json.dump(results, open(f"{DDIR}/loso_v2_primary.json","w"), indent=1)
    print("\nSAVED loso_v2_primary.json")

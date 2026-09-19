import os as _os
_RAW = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "results", "per_point")
import os
"""Reconstruct bench_withobs.py's rolling origins exactly from aligned_*.csv,
so that signed errors, contingency tables and detection scores can be recomputed
for the deterministic models (raw_cams, persistence, snaive, biascorr).

The foundation-model columns cannot be reconstructed without re-running Chronos;
they are handled separately.
"""
import glob, os, json
import numpy as np, pandas as pd

DDIR = os.path.dirname(os.path.abspath(__file__))
K, H, STRIDE, MAX_ORIGINS = 64, 8, 6, 120

def station_files():
    pm25 = [("pm25", f) for f in sorted(glob.glob(f"{DDIR}/aligned_[A-Z]*.csv"))]
    pm10 = [("pm10", f) for f in sorted(glob.glob(f"{DDIR}/aligned_pm10_*.csv"))]
    return pm25 + pm10

def build(pol, f):
    st = os.path.basename(f).split("aligned_")[-1].replace(".csv", "")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="3h")
    g = df.reindex(grid)
    obs, cams, resid = g["obs"], g["cams"], g["resid"]
    obs_i, resid_i = obs.interpolate(limit=4), resid.interpolate(limit=4)
    p95 = np.nanpercentile(obs.values, 95)
    origins = [t for t in range(K, len(grid) - H, STRIDE)
               if obs_i.iloc[t-K:t].notna().mean() > 0.95
               and obs.iloc[t:t+H].notna().all()
               and cams.iloc[t:t+H].notna().all()]
    if len(origins) > MAX_ORIGINS:
        sel = np.linspace(0, len(origins)-1, MAX_ORIGINS).astype(int)
        origins = [origins[i] for i in sel]
    if len(origins) < 25:
        return None
    obs_t  = np.array([[obs.iloc[t+h]  for h in range(H)] for t in origins])
    cams_t = np.array([[cams.iloc[t+h] for h in range(H)] for t in origins])
    bias   = np.array([np.nanmean(resid_i.iloc[t-K:t].values) for t in origins])[:, None]
    pers   = np.array([[obs.iloc[t-1]] * H for t in origins])
    sn     = np.array([[obs.iloc[t+h-8] if t+h-8 >= 0 else np.nan for h in range(H)]
                       for t in origins])
    return dict(pol=pol, st=st, p95=float(p95), n_origins=len(origins),
                obs=obs_t, preds={"raw_cams": cams_t, "persistence": pers,
                                  "snaive": sn, "biascorr": cams_t + bias})

if __name__ == "__main__":
    D = pd.read_csv(_os.path.join(_RAW, "bench_withobs_detail.csv"))
    ref = D.groupby(["pol", "st", "model"]).ae.agg(["count", "mean"])
    ok = bad = 0
    rows = []
    for pol, f in station_files():
        b = build(pol, f)
        if b is None:
            print("skip", pol, f); continue
        for m, P in b["preds"].items():
            ae = np.abs(P - b["obs"])
            fin = np.isfinite(ae)
            n_new, mae_new = int(fin.sum()), float(ae[fin].mean())
            try:
                n_ref, mae_ref = ref.loc[(pol, b["st"], m)]
            except KeyError:
                print("no ref", pol, b["st"], m); continue
            match = (n_new == int(n_ref)) and abs(mae_new - mae_ref) < 1e-6
            ok += match; bad += (not match)
            if not match:
                print(f"MISMATCH {pol} {b['st']:14s} {m:12s} n {n_new} vs {int(n_ref)} "
                      f"MAE {mae_new:.4f} vs {mae_ref:.4f}")
            rows.append(dict(pol=pol, st=b["st"], model=m, n=n_new, mae=mae_new))
    print(f"\nreconstruction check: {ok} matched, {bad} mismatched "
          f"(deterministic models only)")

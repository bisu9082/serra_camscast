import os as _os
_RAW = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "results", "per_point")
import os
"""Re-run the zero-shot foundation-model forecasts on the reconstructed origins and
check them against the archived per-point absolute errors in bench_withobs_detail.csv."""
import sys, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from recon_origins import station_files, build, K, H

MODEL = sys.argv[1] if len(sys.argv) > 1 else "bolt"
SPEC = {"bolt": ("amazon/chronos-bolt-small", "fm_bolt"),
        "c2":   ("amazon/chronos-2",          "fm_c2")}[MODEL]
repo, col = SPEC
kw = dict(device_map="cpu")
if MODEL == "bolt": kw["torch_dtype"] = torch.float32
pipe = BaseChronosPipeline.from_pretrained(repo, **kw)
print("loaded", repo, flush=True)

def fc(ctxs):
    q, m = pipe.predict_quantiles(
        inputs=[torch.tensor(c, dtype=torch.float32) for c in ctxs],
        prediction_length=H, quantile_levels=[0.5])
    a = np.asarray(m, dtype=float)
    return a.reshape(a.shape[0], -1)[:, -H:]

D = pd.read_csv(_os.path.join(_RAW, "bench_withobs_detail.csv"))
ref = D[D.model == col].set_index(["pol", "st", "origin", "h"]).ae

rows = []
for pol, f in station_files():
    b = build(pol, f)
    if b is None: continue
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="3h")
    obs_i = df.reindex(grid)["obs"].interpolate(limit=4)
    # rebuild the origin index list identically
    from recon_origins import STRIDE, MAX_ORIGINS
    g = df.reindex(grid); obs, cams = g["obs"], g["cams"]
    origins = [t for t in range(K, len(grid)-H, STRIDE)
               if obs_i.iloc[t-K:t].notna().mean() > 0.95
               and obs.iloc[t:t+H].notna().all() and cams.iloc[t:t+H].notna().all()]
    if len(origins) > MAX_ORIGINS:
        sel = np.linspace(0, len(origins)-1, MAX_ORIGINS).astype(int)
        origins = [origins[i] for i in sel]
    P = fc([obs_i.iloc[t-K:t].ffill().bfill().values for t in origins])
    ae = np.abs(P - b["obs"])
    got, exp = [], []
    for i in range(len(origins)):
        for h in range(H):
            try: r = ref.loc[(pol, b["st"], i, h+1)]
            except KeyError: continue
            got.append(ae[i, h]); exp.append(float(r))
    got, exp = np.array(got), np.array(exp)
    d = np.abs(got - exp)
    rows.append(dict(pol=pol, st=b["st"], n=len(got), mae_new=got.mean(),
                     mae_ref=exp.mean(), max_abs_diff=d.max(), mean_abs_diff=d.mean()))
    print(f"{pol} {b['st']:14s} n={len(got):5d} MAE new {got.mean():7.3f} "
          f"ref {exp.mean():7.3f}  max|Δ| {d.max():.4f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(fos.path.join(os.path.dirname(os.path.abspath(__file__)), "fm_repro_{MODEL}.csv"), index=False)
print("\n==== summary ====")
for pol in ["pm25", "pm10"]:
    s = R[R.pol == pol]
    w = np.average(s.mae_new, weights=s.n); wr = np.average(s.mae_ref, weights=s.n)
    print(f"{pol}: pooled MAE new {w:.3f} vs archived {wr:.3f} | "
          f"worst per-point |Δ| {s.max_abs_diff.max():.4f}")

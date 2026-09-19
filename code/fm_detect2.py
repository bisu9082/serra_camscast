"""Regenerate FM forecasts and complete Tables S12/S13 with the foundation-model rows."""
import json, os, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from recon_origins import station_files, build, K, H, STRIDE, MAX_ORIGINS

WHO = {"pm10": 45.0, "pm25": 15.0}
B = 2000; rng = np.random.default_rng(20260829)

pipes = {"fm_c2": BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu"),
         "fm_bolt": BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                                        device_map="cpu", torch_dtype=torch.float32)}
def fc(p, ctxs):
    q, m = p.predict_quantiles(inputs=[torch.tensor(c, dtype=torch.float32) for c in ctxs],
                               prediction_length=H, quantile_levels=[0.5])
    a = np.asarray(m, dtype=float); return a.reshape(a.shape[0], -1)[:, -H:]

S = []
for pol, f in station_files():
    b = build(pol, f)
    if b is None: continue
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="3h"); g = df.reindex(grid)
    obs, cams = g["obs"], g["cams"]; obs_i = obs.interpolate(limit=4)
    origins = [t for t in range(K, len(grid)-H, STRIDE)
               if obs_i.iloc[t-K:t].notna().mean() > 0.95
               and obs.iloc[t:t+H].notna().all() and cams.iloc[t:t+H].notna().all()]
    if len(origins) > MAX_ORIGINS:
        sel = np.linspace(0, len(origins)-1, MAX_ORIGINS).astype(int)
        origins = [origins[i] for i in sel]
    ctx = [obs_i.iloc[t-K:t].ffill().bfill().values for t in origins]
    for nm, p in pipes.items():
        b["preds"][nm] = fc(p, ctx)
    S.append(b); print("done", pol, b["st"], flush=True)

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preds.npz"), **{f"{b['pol']}|{b['st']}|{k}": v for b in S for k,v in list(b["preds"].items())+[("OBS",b["obs"])]})
MODELS = ["raw_cams", "biascorr", "persistence", "snaive", "fm_bolt", "fm_c2"]
s13 = []
for pol in ["pm25", "pm10"]:
    sub = [b for b in S if b["pol"] == pol]
    for m in MODELS:
        err = np.concatenate([(b["preds"][m] - b["obs"]).ravel() for b in sub])
        obs = np.concatenate([b["obs"].ravel() for b in sub])
        fin = np.isfinite(err); err, obs = err[fin], obs[fin]
        per_st = [np.nanmean(np.abs(b["preds"][m]-b["obs"])) for b in sub]
        s13.append(dict(pol=pol, model=m, n=int(fin.sum()),
                        MAE=round(float(np.abs(err).mean()),2),
                        RMSE=round(float(np.sqrt((err**2).mean())),2),
                        Bias=round(float(err.mean()),2),
                        nMAE_pct=round(float(100*np.abs(err).mean()/obs.mean()),1),
                        MacroMAE=round(float(np.mean(per_st)),2),
                        MedianMAE=round(float(np.median(per_st)),2)))
print("\n=== Table S13 (complete) ===")
print(pd.DataFrame(s13).to_string(index=False))

def scores(h, m_, f_, c):
    n = h+m_+f_+c
    pod = h/(h+m_) if h+m_ else np.nan
    far = f_/(h+f_) if h+f_ else np.nan
    csi = h/(h+m_+f_) if h+m_+f_ else np.nan
    ar = (h+f_)*(h+m_)/n
    ets = (h-ar)/(h+m_+f_-ar) if (h+m_+f_-ar) else np.nan
    exp = ((h+f_)*(h+m_)+(c+m_)*(c+f_))/n
    hss = (h+c-exp)/(n-exp) if (n-exp) else np.nan
    return pod, far, csi, ets, hss

print("\n=== Table S12 (complete): WHO 24-h mean exceedance ===")
s12 = []
for pol in ["pm10", "pm25"]:
    sub = [b for b in S if b["pol"] == pol]; thr = WHO[pol]
    obs_ex = np.concatenate([np.nanmean(b["obs"], axis=1) > thr for b in sub])
    st_id = np.concatenate([[b["st"]]*b["obs"].shape[0] for b in sub])
    print(f"-- {pol}: thr {thr}, base {int(obs_ex.sum())}/{len(obs_ex)} = {100*obs_ex.mean():.1f}%")
    p, fa, cs, et, hs = scores(int(obs_ex.sum()), 0, int((~obs_ex).sum()), 0)
    print(f"   {'always-alarm':13s} POD {p:.2f} FAR {fa:.2f} CSI {cs:.2f} ETS {et:+.3f} HSS {hs:+.3f}")
    for mdl in MODELS:
        pm = np.concatenate([np.nanmean(b["preds"][mdl], axis=1) for b in sub])
        keep = np.isfinite(pm); pr = pm > thr
        o, q, s_ = obs_ex[keep], pr[keep], st_id[keep]
        h = int((o&q).sum()); m_ = int((o&~q).sum()); f_ = int((~o&q).sum()); c = int((~o&~q).sum())
        p, fa, cs, et, hs = scores(h, m_, f_, c)
        sts = np.unique(s_); idx = {t: np.where(s_==t)[0] for t in sts}; bp=[]; bc=[]
        for _ in range(B):
            ii = np.concatenate([idx[t] for t in rng.choice(sts, len(sts), replace=True)])
            oo, qq = o[ii], q[ii]
            hh, mm, ff = (oo&qq).sum(), (oo&~qq).sum(), ((~oo)&qq).sum()
            if hh+mm: bp.append(hh/(hh+mm))
            if hh+mm+ff: bc.append(hh/(hh+mm+ff))
        lo, hi = np.percentile(bp, [2.5, 97.5]); clo, chi = np.percentile(bc, [2.5, 97.5])
        s12.append(dict(pol=pol, model=mdl, hit=h, miss=m_, fa=f_, cn=c, POD=round(p,3),
                        POD_lo=round(lo,3), POD_hi=round(hi,3), CSI_lo=round(clo,3), CSI_hi=round(chi,3), FAR=round(fa,3),
                        CSI=round(cs,3), ETS=round(et,3), HSS=round(hs,3)))
        print(f"   {mdl:13s} POD {p:.2f} [{lo:.2f},{hi:.2f}] FAR {fa:.2f} CSI {cs:.2f} [{clo:.2f},{chi:.2f}] "
              f"ETS {et:+.3f} HSS {hs:+.3f}  (H{h}/M{m_}/F{f_}/C{c})")
json.dump({"s13": s13, "s12": s12}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables_s12_s13_full2.json"),"w"), indent=1)
print("\nSAVED tables_s12_s13_full2.json")

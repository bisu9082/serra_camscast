"""Recompute Table S13 (signed metrics) and Table S12 (WHO 24-h detection incl.
chance-corrected ETS/HSS) from reconstructed origins, for the deterministic models."""
import json, os
import numpy as np, pandas as pd
from recon_origins import station_files, build

WHO = {"pm10": 45.0, "pm25": 15.0}
B = 2000
rng = np.random.default_rng(20260829)

S = [b for b in (build(p, f) for p, f in station_files()) if b is not None]

# ---------- Table S13: signed error metrics ----------
print("=== Table S13 (deterministic models; FM columns need Chronos re-run) ===")
s13 = []
for pol in ["pm25", "pm10"]:
    sub = [b for b in S if b["pol"] == pol]
    for m in ["raw_cams", "biascorr", "persistence", "snaive"]:
        err = np.concatenate([(b["preds"][m] - b["obs"]).ravel() for b in sub])
        obs = np.concatenate([b["obs"].ravel() for b in sub])
        fin = np.isfinite(err)
        err, obs = err[fin], obs[fin]
        per_st = [np.nanmean(np.abs(b["preds"][m] - b["obs"])) for b in sub]
        s13.append(dict(pol=pol, model=m, n=int(fin.sum()),
                        MAE=round(float(np.abs(err).mean()), 2),
                        RMSE=round(float(np.sqrt((err**2).mean())), 2),
                        Bias=round(float(err.mean()), 2),
                        nMAE_pct=round(float(100*np.abs(err).mean()/obs.mean()), 1),
                        MacroMAE=round(float(np.mean(per_st)), 2),
                        MedianMAE=round(float(np.median(per_st)), 2)))
print(pd.DataFrame(s13).to_string(index=False))

# ---------- Table S12: WHO 24-h mean exceedance ----------
def scores(hit, miss, fa, cn):
    n = hit + miss + fa + cn
    pod = hit/(hit+miss) if hit+miss else np.nan
    far = fa/(hit+fa) if hit+fa else np.nan
    csi = hit/(hit+miss+fa) if hit+miss+fa else np.nan
    ar = (hit+fa)*(hit+miss)/n                      # random hits
    ets = (hit-ar)/(hit+miss+fa-ar) if (hit+miss+fa-ar) else np.nan
    exp = ((hit+fa)*(hit+miss) + (cn+miss)*(cn+fa))/n
    hss = (hit+cn-exp)/(n-exp) if (n-exp) else np.nan
    return pod, far, csi, ets, hss

print("\n=== Table S12: WHO 24-h mean exceedance (origin-level) ===")
s12 = []
for pol in ["pm10", "pm25"]:
    sub = [b for b in S if b["pol"] == pol]
    thr = WHO[pol]
    obs_ex = np.concatenate([np.nanmean(b["obs"], axis=1) > thr for b in sub])
    st_id = np.concatenate([[b["st"]]*b["obs"].shape[0] for b in sub])
    base = obs_ex.mean()
    print(f"-- {pol}: threshold {thr} ug/m3, {int(obs_ex.sum())} of {len(obs_ex)} "
          f"origin-level 24-h means exceed ({100*base:.1f}%)")
    # always-alarm reference
    h, m_, f_, c = int(obs_ex.sum()), 0, int((~obs_ex).sum()), 0
    p, fa, cs, et, hs = scores(h, m_, f_, c)
    print(f"   {'always-alarm':14s} POD {p:.2f} FAR {fa:.2f} CSI {cs:.2f} ETS {et:+.3f} HSS {hs:+.3f}")
    for mdl in ["raw_cams", "biascorr", "persistence", "snaive"]:
        pr = np.concatenate([np.nanmean(b["preds"][mdl], axis=1) > thr for b in sub])
        keep = np.isfinite(np.concatenate([np.nanmean(b["preds"][mdl], axis=1) for b in sub]))
        o, q, s_ = obs_ex[keep], pr[keep], st_id[keep]
        h = int((o & q).sum()); m_ = int((o & ~q).sum())
        f_ = int((~o & q).sum()); c = int((~o & ~q).sum())
        p, fa, cs, et, hs = scores(h, m_, f_, c)
        # station-cluster bootstrap on POD
        sts = np.unique(s_); idx = {t: np.where(s_ == t)[0] for t in sts}
        bp = []
        for _ in range(B):
            pick = rng.choice(sts, len(sts), replace=True)
            ii = np.concatenate([idx[t] for t in pick])
            oo, qq = o[ii], q[ii]
            hh, mm = (oo & qq).sum(), (oo & ~qq).sum()
            if hh+mm: bp.append(hh/(hh+mm))
        lo, hi = np.percentile(bp, [2.5, 97.5])
        s12.append(dict(pol=pol, model=mdl, hit=h, miss=m_, fa=f_, cn=c,
                        POD=round(p,3), POD_lo=round(lo,3), POD_hi=round(hi,3),
                        FAR=round(fa,3), CSI=round(cs,3), ETS=round(et,3), HSS=round(hs,3)))
        print(f"   {mdl:14s} POD {p:.2f} [{lo:.2f},{hi:.2f}] FAR {fa:.2f} CSI {cs:.2f} "
              f"ETS {et:+.3f} HSS {hs:+.3f}   (H{h}/M{m_}/F{f_}/C{c})")

json.dump({"s13": s13, "s12": s12}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables_s12_s13.json"), "w"), indent=1)
print("\nSAVED tables_s12_s13.json")

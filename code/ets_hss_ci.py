"""Station cluster-bootstrap intervals for the chance-corrected detection scores.

Table S12 reports the equitable threat score and the Heidke skill score as point
estimates, while POD and CSI beside them carry intervals. That asymmetry is not
defensible: a chance-corrected score is a ratio of differences of contingency
counts, so it is at least as sensitive to which stations happen to be in the
sample as the raw scores are. This script supplies the missing intervals by the
same resampling used everywhere else in the paper --- stations drawn with
replacement, the contingency table rebuilt from the drawn stations, the score
recomputed --- so the intervals in that table come from one procedure.

Writes ets_hss_ci.json.
"""
import json, os
import numpy as np
from recon_origins import station_files, build

WHO = {"pm10": 45.0, "pm25": 15.0}
B = int(os.environ.get("NBOOT", 2000))
OUT = os.environ.get("CAMSCAST_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "ets_hss_ci.json"))
rng = np.random.default_rng(20260908)

S = [b for b in (build(p, f) for p, f in station_files()) if b is not None]
print(f"reconstructed {len(S)} station-pollutant records", flush=True)
MODELS = sorted({m for b in S for m in b["preds"]})
print("models available:", MODELS, flush=True)


def scores(hit, miss, fa, cn):
    n = hit + miss + fa + cn
    if n == 0:
        return {k: np.nan for k in ("POD", "FAR", "CSI", "ETS", "HSS")}
    pod = hit / (hit + miss) if hit + miss else np.nan
    far = fa / (hit + fa) if hit + fa else np.nan
    csi = hit / (hit + miss + fa) if hit + miss + fa else np.nan
    ar = (hit + fa) * (hit + miss) / n
    ets = (hit - ar) / (hit + miss + fa - ar) if (hit + miss + fa - ar) else np.nan
    exp = ((hit + miss) * (hit + fa) + (cn + miss) * (cn + fa)) / n
    hss = (hit + cn - exp) / (n - exp) if (n - exp) else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, ETS=ets, HSS=hss)


def station_table(b, model, thr):
    """Contingency counts for one station-pollutant record under the 24-h mean rule."""
    p = b["preds"].get(model)
    if p is None:
        return None
    fc = np.nanmean(p, axis=1)          # eight-step forecast mean per origin
    ob = np.nanmean(b["obs"], axis=1)   # observed 24-h mean per origin
    ok = np.isfinite(fc) & np.isfinite(ob)
    fc, ob = fc[ok], ob[ok]
    if len(fc) == 0:
        return None
    f, o = fc >= thr, ob >= thr
    return np.array([int((f & o).sum()), int((~f & o).sum()),
                     int((f & ~o).sum()), int((~f & ~o).sum())], float)


out = {}
for pol, thr in WHO.items():
    sub = [b for b in S if b["pol"] == pol]
    for model in MODELS:
        tabs = [t for t in (station_table(b, model, thr) for b in sub) if t is not None]
        if len(tabs) < 3:
            continue
        T = np.array(tabs)
        pt = scores(*T.sum(axis=0))
        draws = {k: [] for k in pt}
        for _ in range(B):
            idx = rng.integers(0, len(T), len(T))
            s = scores(*T[idx].sum(axis=0))
            for k, v in s.items():
                draws[k].append(v)
        rec = dict(n_stations=len(T),
                   hit=int(T[:, 0].sum()), miss=int(T[:, 1].sum()),
                   fa=int(T[:, 2].sum()), cn=int(T[:, 3].sum()))
        for k, v in pt.items():
            a = np.array(draws[k], float)
            a = a[np.isfinite(a)]
            rec[k] = round(float(v), 3)
            rec[k + "_lo"] = round(float(np.percentile(a, 2.5)), 3) if len(a) else None
            rec[k + "_hi"] = round(float(np.percentile(a, 97.5)), 3) if len(a) else None
        out[f"{pol}:{model}"] = rec
        print(f"  {pol:5s} {model:14s} n={rec['n_stations']:2d}  "
              f"ETS {rec['ETS']:.3f} [{rec['ETS_lo']},{rec['ETS_hi']}]  "
              f"HSS {rec['HSS']:.3f} [{rec['HSS_lo']},{rec['HSS_hi']}]", flush=True)

out["_meta"] = dict(n_boot=B, resampling="stations drawn with replacement, contingency "
                                         "table rebuilt from the drawn stations",
                    rule="eight-step forecast mean against the WHO 24-h guideline value",
                    thresholds=WHO)
json.dump(out, open(OUT, "w"), indent=1)
print("saved", os.path.basename(OUT))

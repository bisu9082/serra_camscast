"""Recompute SI Tables S16, S17 and S18 from the verified per-point record and the
station inventory, so that they carry the same provenance as the rest of the paper."""
import os as _os
_RAW = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "results", "per_point")
import json, os
import numpy as np, pandas as pd
from recon_origins import station_files, build, K, H, STRIDE, MAX_ORIGINS

RAW = _RAW
D = pd.read_csv(f"{RAW}/bench_withobs_detail.csv")
S = pd.read_csv(f"{RAW}/fig1_station_table.csv")
S10 = S[S.pol == "PM10"].copy(); S10["id"] = S10.st.str[1:].astype(int); S10 = S10.set_index("id")
rng = np.random.default_rng(20260907)
out = {}

# ---------------- origin timestamps, for episode blocking ----------------
stamps = {}
for pol, f in station_files():
    if pol != "pm10":
        continue
    st = f.split("aligned_")[-1].replace(".csv", "")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="3h"); g = df.reindex(grid)
    obs, cams = g["obs"], g["cams"]; obs_i = obs.interpolate(limit=4)
    origins = [t for t in range(K, len(grid) - H, STRIDE)
               if obs_i.iloc[t-K:t].notna().mean() > 0.95
               and obs.iloc[t:t+H].notna().all() and cams.iloc[t:t+H].notna().all()]
    if len(origins) > MAX_ORIGINS:
        sel = np.linspace(0, len(origins)-1, MAX_ORIGINS).astype(int)
        origins = [origins[i] for i in sel]
    stamps[st] = {i: grid[t] for i, t in enumerate(origins)}

P = D[D.pol == "pm10"].copy()
P["t"] = [stamps[s].get(o) for s, o in zip(P.st, P.origin)]
P = P.dropna(subset=["t"])
t0 = P["t"].min()
# episode = contiguous multi-day block; cut the record into blocks separated by >36 h
allt = np.sort(P["t"].unique())
gaps = np.diff(allt).astype("timedelta64[h]").astype(float)
cuts = np.where(gaps > 36)[0]
edges = np.concatenate([[allt[0]], allt[cuts + 1], [allt[-1] + np.timedelta64(1, "h")]])
# if the record is continuous, fall back to fixed 8-day blocks (~16 blocks over Sep-Dec)
if len(edges) - 1 < 8:
    edges = pd.date_range(pd.Timestamp(allt[0]).floor("D"),
                          pd.Timestamp(allt[-1]).ceil("D") + pd.Timedelta(days=8),
                          freq="8D").values
P["ep"] = np.searchsorted(edges, P["t"].values, side="right") - 1
eps = sorted(P.ep.unique())
print(f"episode blocks: {len(eps)}  (span {pd.Timestamp(allt[0]).date()} to {pd.Timestamp(allt[-1]).date()})")

MODELS = ["raw_cams", "biascorr", "persistence", "snaive", "fm_bolt", "fm_c2"]
piv = {m: P[P.model == m].groupby("ep").ae.mean() for m in MODELS}
n_ep = {m: P[P.model == m].groupby("ep").ae.size() for m in MODELS}
B = 2000
rows = []
for m in MODELS:
    pts = P[P.model == m]
    mae = pts.ae.mean()
    grp = {e: v.ae.values for e, v in pts.groupby("ep")}
    ee = list(grp)
    bs = [np.concatenate([grp[e] for e in rng.choice(ee, len(ee), replace=True)]).mean()
          for _ in range(B)]
    rows.append(dict(model=m, MAE=round(float(mae), 1),
                     lo=round(float(np.percentile(bs, 2.5)), 1),
                     hi=round(float(np.percentile(bs, 97.5)), 1)))
# paired gap raw_cams - fm_c2, resampled by episode
gc = {e: v.ae.values for e, v in P[P.model == "raw_cams"].groupby("ep")}
gf = {e: v.ae.values for e, v in P[P.model == "fm_c2"].groupby("ep")}
ee = sorted(set(gc) & set(gf))
gapb = []
for _ in range(B):
    pick = rng.choice(ee, len(ee), replace=True)
    gapb.append(np.concatenate([gc[e] for e in pick]).mean()
                - np.concatenate([gf[e] for e in pick]).mean())
gap = P[P.model == "raw_cams"].ae.mean() - P[P.model == "fm_c2"].ae.mean()
out["S16"] = dict(n_episodes=len(eps), rows=rows,
                  gap=round(float(gap), 1),
                  gap_ci=[round(float(np.percentile(gapb, 2.5)), 1),
                          round(float(np.percentile(gapb, 97.5)), 1)],
                  per_episode_wins=int(sum(gc[e].mean() > gf[e].mean() for e in ee)),
                  n_ep_compared=len(ee))
print("\n=== Table S16 (episode-block bootstrap) ===")
print(pd.DataFrame(rows).to_string(index=False))
print(f"raw CAMS - Chronos-2 gap {gap:.1f} CI {out['S16']['gap_ci']} | "
      f"CAMS worse in {out['S16']['per_episode_wins']}/{len(ee)} episodes")

# ---------------- S17: within-cell representativeness ----------------
GRID_DEG = 0.75
# nearest 0.75 deg grid centre, matching the CAMS extraction and audit_v3.py CELL
S10["cy"] = np.round(S10.lat / GRID_DEG).astype(int)
S10["cx"] = np.round(S10.lon / GRID_DEG).astype(int)
obs_mean = P[P.model == "raw_cams"].groupby("st").apply(lambda d: np.nan)  # placeholder
# station mean observed concentration from the aligned records
smean = {}
for pol, f in station_files():
    if pol != "pm10":
        continue
    sid = int(f.split("aligned_pm10_")[-1].replace(".csv", ""))
    smean[sid] = pd.read_csv(f, index_col=0, parse_dates=True)["obs"].mean()
S10["obs_mean"] = pd.Series(smean)
cells = []
for (cy, cx), g in S10.groupby(["cy", "cx"]):
    if len(g) < 2:
        continue
    cells.append(dict(cell=f"{cy*GRID_DEG:.2f}N/{cx*GRID_DEG:.2f}E", n=len(g),
                      mean_obs=round(float(g.obs_mean.mean()), 1),
                      rng=round(float(g.obs_mean.max() - g.obs_mean.min()), 1),
                      sd=round(float(g.obs_mean.std(ddof=1)), 1)))
cells.sort(key=lambda r: -r["rng"])
out["S17"] = dict(cells=cells,
                  median_range=round(float(np.median([c["rng"] for c in cells])), 1),
                  max_range=round(max(c["rng"] for c in cells), 1),
                  n_cells=len(cells), n_stations_in_multi=int(sum(c["n"] for c in cells)))
print("\n=== Table S17 (within-cell representativeness) ===")
print(pd.DataFrame(cells).to_string(index=False))
print(f"median within-cell range {out['S17']['median_range']} ug/m3, max {out['S17']['max_range']}")

# ---------------- S18: headline on well-represented stations ----------------
low = S10.index[S10.bias.abs() <= 15]
high = S10.index[S10.bias.abs() > 15]
def sub(ids):
    keys = {f"pm10_{i}" for i in ids}
    q = P[P.st.isin(keys)]
    r = q.groupby("model").ae.mean()
    return r, len(keys)
r_low, n_low = sub(low); r_high, n_high = sub(high); r_all, n_all = sub(S10.index)
def red(r): return 100 * (r["raw_cams"] - r["fm_c2"]) / r["raw_cams"]
out["S18"] = dict(
    low=dict(n=n_low, mean_abs_bias=round(float(S10.bias[low].abs().mean()), 1),
             raw=round(float(r_low["raw_cams"]), 1), fm=round(float(r_low["fm_c2"]), 1),
             pers=round(float(r_low["persistence"]), 1), reduction=round(float(red(r_low)), 0)),
    high=dict(n=n_high, mean_abs_bias=round(float(S10.bias[high].abs().mean()), 1),
              raw=round(float(r_high["raw_cams"]), 1), fm=round(float(r_high["fm_c2"]), 1),
              pers=round(float(r_high["persistence"]), 1), reduction=round(float(red(r_high)), 0)),
    all=dict(n=n_all, raw=round(float(r_all["raw_cams"]), 1), fm=round(float(r_all["fm_c2"]), 1),
             pers=round(float(r_all["persistence"]), 1), reduction=round(float(red(r_all)), 0)))
print("\n=== Table S18 (well-represented subset) ===")
for k, v in out["S18"].items():
    print(f"  {k:5s} n={v['n']:2d} raw {v['raw']:6.1f}  Chronos-2 {v['fm']:5.1f}  "
          f"persistence {v['pers']:5.1f}  reduction {v['reduction']:.0f}%")

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "si_S16_S17_S18.json"), "w"), indent=1)
print("\nSAVED si_S16_S17_S18.json")

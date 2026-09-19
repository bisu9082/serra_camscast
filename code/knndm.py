"""Is a kNNDM fold assignment available for this network?

Nearest-neighbour distance matching (NNDM, Mila et al. 2022) and its k-fold form
(kNNDM, Linnenbrink et al. 2024) choose cross-validation folds so that the
distribution of distances from a held-out point to its nearest training point
matches the distribution of distances from a prediction point to its nearest
training point. The method is the right one to ask about here, because the whole
question of this audit is whether a fold design reproduces the geometry of the
deployment it stands for.

Whether it can be applied is an empirical question about this network, and this
script answers it with the two quantities the method needs:

  G(d)  the ECDF of the distance from each station to its nearest other station
        (what a leave-one-station-out fold exposes a model to), and
  Gj(d) the ECDF of the distance from a prediction location to its nearest
        station (what deployment exposes it to).

kNNDM works by moving stations between folds until those two curves agree. The
number of distinct fold assignments available is what decides whether that search
has anything to search over, so the script reports it alongside the buffered
designs the manuscript uses instead.

Writes knndm.json.
"""
import json, os, itertools
import numpy as np
from loso_v2 import PM10C, hav

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(HERE, "knndm.json"))
RADII = (0, 25, 50, 75, 100)

ids = sorted(PM10C)
n = len(ids)
D = np.array([[hav(PM10C[a], PM10C[b]) for b in ids] for a in ids])
np.fill_diagonal(D, np.inf)

# G(d): station to its nearest other station
nn = D.min(axis=1)

# Gj(d): a prediction location to its nearest station. The deployment target is an
# unmonitored site in the same region, so we sample the network's bounding box.
lat = np.array([PM10C[i][0] for i in ids]); lon = np.array([PM10C[i][1] for i in ids])
rng = np.random.default_rng(0)
NS = 20000
plat = rng.uniform(lat.min(), lat.max(), NS)
plon = rng.uniform(lon.min(), lon.max(), NS)
gj = np.array([min(hav((a, b), PM10C[i]) for i in ids) for a, b in zip(plat, plon)])

# What each buffered design exposes the model to: distance from the target to its
# nearest SURVIVING source under that radius and the cell exclusion.
CELL = {i: (round(la / 0.75), round(lo / 0.75)) for i, (la, lo) in PM10C.items()}
by_radius = {}
for R in RADII:
    ds = []
    for k, t in enumerate(ids):
        surv = [m for m, j in enumerate(ids)
                if j != t and D[k, m] > R and CELL[j] != CELL[t]]
        if len(surv) < 4:
            continue
        ds.append(float(D[k, surv].min()))
    by_radius[f"R{R}"] = dict(n_targets=len(ds), median_nn_to_source=float(np.median(ds)),
                              min_nn_to_source=float(np.min(ds)), max_nn_to_source=float(np.max(ds)))

# How much room a kNNDM search has: with n stations and k folds the assignment space
# is finite and small, and the method needs enough of it to move the ECDF.
def stirling_partitions(n, k):
    """Number of ways to split n labelled stations into k non-empty folds."""
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    return S[n][k]

out = dict(
    n_stations=n,
    nn_station_to_station=dict(
        median=float(np.median(nn)), mean=float(nn.mean()),
        min=float(nn.min()), max=float(nn.max()),
        q25=float(np.percentile(nn, 25)), q75=float(np.percentile(nn, 75))),
    nn_prediction_to_station=dict(
        n_sample=NS, median=float(np.median(gj)), mean=float(gj.mean()),
        min=float(gj.min()), max=float(gj.max()),
        q25=float(np.percentile(gj, 25)), q75=float(np.percentile(gj, 75))),
    buffered_designs=by_radius,
    fold_assignments_available={f"k={k}": stirling_partitions(n, k) for k in (5, 6, 10)},
    leave_one_out_folds=n,
)

# The comparison that decides the question: LOSO exposes the model to G(d); the
# deployment case the manuscript is about (an unmonitored site) exposes it to Gj(d).
out["gap_median_km"] = out["nn_prediction_to_station"]["median"] - out["nn_station_to_station"]["median"]

print(f"stations {n}")
print(f"  station -> nearest other station : median {out['nn_station_to_station']['median']:.1f} km "
      f"(IQR {out['nn_station_to_station']['q25']:.1f}-{out['nn_station_to_station']['q75']:.1f}, "
      f"max {out['nn_station_to_station']['max']:.1f})")
print(f"  prediction point -> nearest stn  : median {out['nn_prediction_to_station']['median']:.1f} km "
      f"(IQR {out['nn_prediction_to_station']['q25']:.1f}-{out['nn_prediction_to_station']['q75']:.1f}, "
      f"max {out['nn_prediction_to_station']['max']:.1f})")
print(f"  gap in medians                   : {out['gap_median_km']:+.1f} km")
print("  buffered designs (target -> nearest surviving source):")
for k, v in by_radius.items():
    print(f"    {k:<6} n={v['n_targets']:2d}  median {v['median_nn_to_source']:6.1f} km  "
          f"range {v['min_nn_to_source']:.1f}-{v['max_nn_to_source']:.1f}")
json.dump(out, open(OUT, "w"), indent=1)
print("saved", os.path.basename(OUT))

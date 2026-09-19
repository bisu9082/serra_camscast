"""Residual spatial autocorrelation of the PM10 network, by distance band.

Reported in the Methods to justify the buffer ladder: the radii are chosen to
span the estimated range of residual spatial autocorrelation rather than by
convention. Writes moran.json.

Moran's I is computed on the per-station mean observation-minus-CAMS residual,
with binary weights within each distance band and, separately, inverse-distance
weights. Significance is a permutation test on the station labels.
"""
import json, os
import numpy as np, pandas as pd
from loso_v2 import PM10C, DATA, hav

NPERM = 9999
rng = np.random.default_rng(0)

ids = sorted(PM10C)
x = np.array([DATA[i][0]["resid"].mean() for i in ids])
n = len(ids)
D = np.array([[hav(PM10C[a], PM10C[b]) for b in ids] for a in ids])


def moran(W, z):
    np.fill_diagonal(W, 0.0)
    S0 = W.sum()
    if S0 == 0:
        return np.nan
    return (n / S0) * (z @ W @ z) / (z @ z)


z = x - x.mean()
out = {"n_stations": n, "mean_residual": float(x.mean()),
       "expected_I": float(-1 / (n - 1)), "n_perm": NPERM, "bands": {}}

for band in (15, 25, 50, 100):
    W = (D <= band).astype(float)
    np.fill_diagonal(W, 0.0)
    npairs = int(W.sum() / 2)
    I = moran(W.copy(), z)
    perm = np.empty(NPERM)
    for k in range(NPERM):
        zz = rng.permutation(z)
        perm[k] = moran(W.copy(), zz)
    p = (np.sum(np.abs(perm - perm.mean()) >= abs(I - perm.mean())) + 1) / (NPERM + 1)
    out["bands"][f"<= {band} km"] = dict(I=float(I), n_pairs=npairs, p_perm=float(p))
    print(f"  <= {band:3d} km   I = {I:+.3f}   pairs {npairs:3d}   p = {p:.4f}", flush=True)

# inverse-distance weights over all pairs
Wi = np.zeros_like(D)
nz = D > 0
Wi[nz] = 1.0 / D[nz]
I = moran(Wi.copy(), z)
perm = np.array([moran(Wi.copy(), rng.permutation(z)) for _ in range(NPERM)])
p = (np.sum(np.abs(perm - perm.mean()) >= abs(I - perm.mean())) + 1) / (NPERM + 1)
out["inverse_distance"] = dict(I=float(I), p_perm=float(p))
print(f"  inverse distance   I = {I:+.3f}   p = {p:.4f}", flush=True)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "moran.json"), "w"), indent=1)
print("expected I under no autocorrelation:", round(out["expected_I"], 4))
print("saved moran.json")

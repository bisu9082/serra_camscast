"""Nearest-neighbour distance structure and area of applicability for the simulated audit.

The empirical supplement reports these two diagnostics for the observed network
(Tables S27 and S28). A reviewer is entitled to ask the same of the simulated networks,
because the simulation is what carries the paper's generality claim: if the simulated
validation ladder does not span the same deployment geometry as the real one, the
operating characteristics it reports are about a different problem.

Two quantities are computed for each geometry (dense, sparse) and each rung of the ladder.

  1. kNNDM-relevant distance structure.  For every target station, the distance to its
     nearest surviving source, and the nearest-neighbour distance within the surviving
     source set.  The Wasserstein-1 distance between those two distributions is the
     discrepancy kNNDM minimises (Milà et al. 2022; Linnenbrink et al. 2024).  At a zero
     buffer the two coincide, which is the formal statement that plain leave-one-station-out
     validates interpolation.

  2. Unweighted dissimilarity index (Meyer and Pebesma 2021).  For each target prediction
     point, the minimum scaled distance to any training point, divided by the mean
     training-to-training distance; the area-of-applicability threshold is the 0.75
     quantile of the training-to-training nearest-neighbour distances plus 1.5 times their
     interquartile range, applied to the same scaling.  The variant used here is
     unweighted: the feature-importance weighting of the original formulation requires a
     fitted model, and the point at issue is the design, not one corrector.

No Monte Carlo is needed.  The station geometry and the cell assignment are deterministic
given the geometry factor, and the feature matrix for a realization is a deterministic
function of the simulated field, so one seeded realization per cell is sufficient and is
reported as such.

Writes aoa_knndm_sim.csv.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "sim", os.path.join(HERE, "serra_simulation_v2.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

OUT = os.environ.get("CAMSCAST_OUT", os.path.join(HERE, "aoa_knndm_sim.csv"))
RADII = (0.0, 25.0, 50.0, 100.0)
SEED = 20260910


def wasserstein1(a, b):
    """W1 between two empirical distributions, by quantile matching."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    q = np.linspace(0.0, 1.0, 512)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def surviving_sources(D, cell, target, design, radius):
    n = D.shape[0]
    src = np.array([j for j in range(n) if j != target], int)
    if design in ("space", "space_time", "full"):
        src = src[D[target, src] > radius]
    if design == "full":
        src = src[cell[src] != cell[target]]
    return src


def di_unweighted(Xtr, Xte):
    """Dissimilarity index of prediction points relative to a training set.

    Scaling is the training-set standard deviation, as in the reference implementation;
    the denominator is the mean training-to-training nearest-neighbour distance.
    """
    s = Xtr.std(axis=0, ddof=1)
    s[s < 1e-12] = 1.0
    A = Xtr / s
    B = Xte / s

    d_tr = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_tr, np.inf)
    nn_tr = d_tr.min(axis=1)
    denom = float(nn_tr.mean())
    if denom <= 0:
        return np.nan, np.nan, np.nan

    d_te = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    di = d_te.min(axis=1) / denom

    scaled_nn = nn_tr / denom
    q75, q25 = np.percentile(scaled_nn, [75, 25])
    threshold = float(q75 + 1.5 * (q75 - q25))
    return float(np.median(di)), threshold, float(np.mean(di > threshold))


rows = []
for geometry in ("dense", "sparse"):
    dat = sim.simulate_field(
        seed=SEED, beta=1.0, spatial_on=True, temporal_on=True, grid_on=True,
        geometry=geometry, n_days=120, sigma_noise=0.75)
    D = sim.pairdist(dat["x"], dat["y"])
    cell = dat["cell"]
    n, T = dat["residual"].shape
    station_index = np.repeat(np.arange(n), T)

    for spec in ("coordfree", "full"):
        X = sim.rbf_features(dat, spec)
        for radius in RADII:
            for design in ("station", "space", "space_time", "full"):
                if design == "station" and radius != 0.0:
                    continue
                tgt_nn, src_nn, di_med, di_out = [], [], [], []
                for target in range(n):
                    src = surviving_sources(D, cell, target, design, radius)
                    if len(src) < 4:
                        continue
                    tgt_nn.append(float(D[target, src].min()))
                    sub = D[np.ix_(src, src)].copy()
                    np.fill_diagonal(sub, np.inf)
                    src_nn.extend(sub.min(axis=1).tolist())

                    tr = np.isin(station_index, src)
                    te = station_index == target
                    m, thr, out = di_unweighted(X[tr][::7], X[te][::7])
                    if np.isfinite(m):
                        di_med.append(m)
                        di_out.append(out)
                if not tgt_nn:
                    continue
                rows.append(dict(
                    geometry=geometry, spec=spec, radius_km=radius, design=design,
                    n_targets=len(tgt_nn),
                    target_to_source_nn_median=float(np.median(tgt_nn)),
                    within_source_nn_median=float(np.median(src_nn)),
                    w1_km=wasserstein1(tgt_nn, src_nn),
                    di_median=float(np.median(di_med)) if di_med else np.nan,
                    share_outside_aoa=float(np.mean(di_out)) if di_out else np.nan,
                ))
                print(f"{geometry:6s} {spec:9s} R={radius:5.0f} {design:11s} "
                      f"tgt-nn {rows[-1]['target_to_source_nn_median']:6.1f} km  "
                      f"src-nn {rows[-1]['within_source_nn_median']:6.1f} km  "
                      f"W1 {rows[-1]['w1_km']:6.1f} km  "
                      f"DI {rows[-1]['di_median']:.3f}  "
                      f"outside {100*rows[-1]['share_outside_aoa']:5.1f}%", flush=True)

import csv
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nsaved", os.path.basename(OUT), f"({len(rows)} rows)")

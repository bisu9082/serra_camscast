#!/usr/bin/env python
"""
SERRA ground-truth simulation for camscast-audit (v2).

Purpose
-------
Test whether four validation designs distinguish structural transferable signal
from dependence-driven interpolation in an environmental residual-correction model.

Primary design
--------------
DGP:
    r_i(t) = beta * F_i(t) + S_i + T(t) + G_c(i)(t) + eps_i(t)
    y_i(t) = m_i(t) + r_i(t)

where:
- F_i(t): globally transferable function of the mechanistic field/time;
- S_i: spatially correlated station effect;
- T(t): temporally autocorrelated common residual;
- G_c(i)(t): shared-grid-cell residual component;
- eps_i(t): iid noise.

Validation ladder:
    station      target station held out, source stations at all times allowed
    space        + sources within R km excluded
    space_time   + contiguous target test block and data-adaptive temporal guard
    full         + source stations sharing the target mechanistic-grid cell excluded

A single contiguous test block is used per Monte Carlo replicate. Repeated
simulation, rather than six OOF folds, supplies the sampling distribution. This
keeps the simulation computationally tractable while directly testing the
dependence pathways of interest.

The primary corrector is deliberately transparent: ridge regression on a
nonlinear RBF basis of mechanistic/time features, optionally including spatial
RBF features. The real-data manuscript retains HistGradientBoostingRegressor;
the simulation tests the validation design rather than reproducing a single
learner.

Outputs
-------
simulation_results.csv
simulation_summary.csv
simulation_config.json

Example
-------
python serra_simulation.py --n-rep 100 --jobs 8 --outdir results/serra_sim
python serra_simulation.py --smoke --outdir results/serra_sim_smoke
"""

from __future__ import annotations
import argparse
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


# Same 19-station geometry as the PM10 audit.
PM10_COORDS = {
    369953:(23.6522,53.7038),369954:(24.4819,54.3691),369955:(23.0957,53.6064),
    369957:(24.4889,54.3637),369958:(24.3471,54.5028),369959:(24.2258,55.7658),
    369960:(24.4665,55.3428),369961:(24.4199,54.5781),369962:(23.8355,52.8103),
    369963:(24.4035,54.5160),369965:(24.0908,52.7548),369966:(24.3213,54.6359),
    369967:(23.7504,53.7452),369968:(24.2190,55.7348),369969:(24.1634,55.7021),
    369971:(23.5311,55.4859),369972:(24.2591,55.7048),369973:(24.0351,53.8853),
    369974:(24.2870,54.5861),
}

DESIGNS = ("station", "space", "space_time", "full")
SPECS = ("coordfree", "full")


def local_xy_km(scale: float = 1.0):
    ids = np.array(sorted(PM10_COORDS))
    lat = np.array([PM10_COORDS[i][0] for i in ids], float)
    lon = np.array([PM10_COORDS[i][1] for i in ids], float)
    lat0, lon0 = lat.mean(), lon.mean()
    x = (lon - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 111.0
    return ids, x * scale, y * scale


def pairdist(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)


def ar1(n, rho, rng):
    z = np.empty(n, float)
    z[0] = rng.normal()
    innovation = rng.normal(size=n)
    s = math.sqrt(max(1.0 - rho * rho, 1e-12))
    for k in range(1, n):
        z[k] = rho * z[k - 1] + s * innovation[k]
    z -= z.mean()
    z /= z.std(ddof=1) + 1e-12
    return z


def temporal_component(n, strong, rng):
    if not strong:
        return ar1(n, 0.10, rng)
    # Mixture chosen so the theoretical ACF is approximately:
    # lag 1 d = 0.35, lag 3 d = 0.19, lag 10 d = 0.08,
    # matching the scale that motivated the real-data 10-day guard.
    fast = ar1(n, 0.15, rng)
    slow = ar1(n, 0.885, rng)
    z = math.sqrt(0.73) * fast + math.sqrt(0.27) * slow
    z -= z.mean()
    z /= z.std(ddof=1) + 1e-12
    return z


def spatial_component(x, y, strong, rng, range_km=70.0):
    n = len(x)
    if not strong:
        return np.zeros(n)
    d = pairdist(x, y)
    cov = np.exp(-d / range_km) + np.eye(n) * 1e-8
    z = rng.multivariate_normal(np.zeros(n), cov)
    z -= z.mean()
    z /= z.std(ddof=1) + 1e-12
    return z


def observed_cams_grid_ids(ids):
    """Return the exact 0.75-degree CAMS cell assignment used in audit_v7.

    The assignment is held fixed for dense and sparse simulation geometries so
    that the geometry factor changes source-target separation without
    simultaneously changing the shared-grid factor. The observed 19-station
    network occupies nine cells under this rule.
    """
    labels = np.array([
        f"{round(PM10_COORDS[int(i)][0] / 0.75)}_{round(PM10_COORDS[int(i)][1] / 0.75)}"
        for i in ids
    ])
    unique = np.unique(labels)
    lut = {c: k for k, c in enumerate(unique)}
    cidx = np.array([lut[c] for c in labels], int)
    if len(ids) == 19:
        assert len(unique) == 9, f"expected 9 observed CAMS cells, got {len(unique)}"
    return labels, cidx, len(unique)


def simulate_field(seed, beta, spatial_on, temporal_on, grid_on,
                   geometry, n_days=120, sigma_noise=0.75):
    rng = np.random.default_rng(seed)
    scale = 1.0 if geometry == "dense" else 2.0
    ids, x, y = local_xy_km(scale)
    n = len(ids)
    cells, cidx, ncell = observed_cams_grid_ids(ids)

    # Mechanistic field m_i(t): same CAMS-like predictor series for every
    # station assigned to the same 0.75-degree cell. This reproduces the shared
    # predictor support that motivates leave-one-grid-cell-out in the empirical
    # audit. There is deliberately no station-specific perturbation inside m.
    weather = ar1(n_days, 0.70, rng)
    cell_fingerprint = rng.normal(size=ncell)
    cell_local = np.vstack([ar1(n_days, 0.45, rng) for _ in range(ncell)])
    m_cell = (
        60.0
        + 12.0 * weather[None, :]
        + 8.0 * cell_fingerprint[:, None]
        + 6.0 * cell_local
    )
    m = m_cell[cidx, :]

    # Structural invariant: same-cell stations must share an identical
    # mechanistic predictor series, exactly as in the real CAMS extraction.
    for c in np.unique(cidx):
        members = np.where(cidx == c)[0]
        if len(members) > 1:
            assert np.allclose(m[members], m[members[0]][None, :])
    mz = (m - m.mean()) / (m.std(ddof=1) + 1e-12)

    # Globally transferable component. It has the same functional relationship
    # at every station and remains available after spatial/time/grid separation.
    day = np.arange(n_days)
    season = np.sin(2.0 * np.pi * day / 365.0)[None, :]
    f = mz + 0.25 * (mz ** 2 - 1.0) + 0.20 * season
    f = (f - f.mean()) / (f.std(ddof=1) + 1e-12)
    transferable = beta * f

    s = spatial_component(x, y, spatial_on, rng)[:, None]
    if not spatial_on:
        s[:] = 0.0

    t = temporal_component(n_days, temporal_on, rng)[None, :]
    if not temporal_on:
        t[:] = 0.0

    if grid_on:
        static = rng.normal(scale=0.75, size=ncell)
        dynamic = np.vstack([
            0.60 * temporal_component(n_days, True, rng) for _ in range(ncell)
        ])
        g = static[cidx, None] + dynamic[cidx, :]
    else:
        g = np.zeros((n, n_days))

    eps = rng.normal(scale=sigma_noise, size=(n, n_days))
    residual = transferable + s + t + g + eps
    observation = m + residual

    return {
        "ids": ids, "x": x, "y": y, "cell": cells,
        "mechanistic": m, "residual": residual, "observation": observation,
        "n_days": n_days,
    }


def rbf_features(dat, spec):
    m = dat["mechanistic"]
    x, y = dat["x"], dat["y"]
    n, T = m.shape
    day = np.arange(T)
    mz = (m - m.mean()) / (m.std(ddof=1) + 1e-12)

    cols = [
        mz.reshape(-1),
        (mz ** 2 - 1.0).reshape(-1),
        np.tile(np.sin(2.0 * np.pi * day / 365.0), n),
        np.tile(np.cos(2.0 * np.pi * day / 365.0), n),
    ]

    # Flexible date basis: a transparent analogue of a nonlinear learner
    # partitioning day-of-year. Exact/nearby source times can therefore be
    # exploited unless the temporal block + guard removes them.
    t_centres = np.linspace(0, T - 1, 13)
    t_bw = max(T / 18.0, 4.0)
    for c in t_centres:
        cols.append(np.tile(np.exp(-0.5 * ((day - c) / t_bw) ** 2), n))

    if spec == "full":
        # Smooth spatial basis, analogous to nonlinear use of lat/lon.
        xs = np.linspace(x.min(), x.max(), 3)
        ys = np.linspace(y.min(), y.max(), 3)
        xx = np.repeat(x, T)
        yy = np.repeat(y, T)
        s_bw = max(np.ptp(x), np.ptp(y)) / 3.0
        for a in xs:
            for b in ys:
                cols.append(np.exp(-0.5 * (((xx - a) / s_bw) ** 2 +
                                           ((yy - b) / s_bw) ** 2)))
    return np.column_stack(cols)


def raw_features(dat, spec):
    """Feature matrix for the tree learner.

    The ridge learner is given a fixed nonlinear basis because a linear model cannot build
    one for itself. A gradient-boosted tree builds its own partition, so it is given the
    features the empirical audit actually uses: the mechanistic value, the seasonal pair, the
    date index, and -- for the coordinate-bearing specification -- the coordinates. The date
    index is included deliberately: over a single simulated year it is a bijection onto the
    day, which is the pathway the temporal guard exists to close, and withholding it would
    make the tree easier to audit than the model in the empirical arm.
    """
    m = dat["mechanistic"]
    x, y = dat["x"], dat["y"]
    n, T = m.shape
    day = np.arange(T)
    cols = [
        m.reshape(-1),
        np.tile(np.sin(2.0 * np.pi * day / 365.0), n),
        np.tile(np.cos(2.0 * np.pi * day / 365.0), n),
        np.tile(day.astype(float), n),
    ]
    if spec == "full":
        cols.append(np.repeat(x, T))
        cols.append(np.repeat(y, T))
    return np.column_stack(cols)


def design_matrix(dat, spec, learner):
    return rbf_features(dat, spec) if learner == "ridge" else raw_features(dat, spec)


def make_model(learner, ridge_alpha):
    if learner == "ridge":
        return Ridge(alpha=ridge_alpha)
    # Matches the empirical audit's estimator settings: deterministic, no internal
    # early-stopping holdout, so a rerun reproduces bit-for-bit.
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                         random_state=0, early_stopping=False)


def acf_at(z, lag):
    if lag <= 0 or lag >= len(z):
        return np.nan
    a, b = z[:-lag], z[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def estimate_guard_days(residual, source_stations, threshold=0.10, max_days=14):
    # Mirrors the empirical logic: per-station ACF, then the network median.
    # Only source stations are used; the target outcome does not tune its guard.
    if len(source_stations) == 0:
        return max_days
    for lag in range(1, max_days + 1):
        vals = [acf_at(residual[j], lag) for j in source_stations]
        vals = np.array([v for v in vals if np.isfinite(v)], float)
        if len(vals) and abs(float(np.median(vals))) <= threshold:
            return lag
    return max_days


def fit_one_target(dat, X, target, design, test_start, test_len,
                   radius_km=50.0, ridge_alpha=2.0, learner="ridge"):
    n, T = dat["residual"].shape
    D = pairdist(dat["x"], dat["y"])
    src = np.array([j for j in range(n) if j != target], int)

    if design in ("space", "space_time", "full"):
        src = src[D[target, src] > radius_km]
    if design == "full":
        src = src[dat["cell"][src] != dat["cell"][target]]
    if len(src) < 4:
        return None

    guard = 0
    if design in ("space_time", "full"):
        guard = estimate_guard_days(dat["residual"], src)

    station_index = np.repeat(np.arange(n), T)
    time_index = np.tile(np.arange(T), n)
    train = np.isin(station_index, src)

    lo, hi = test_start, test_start + test_len
    if design in ("space_time", "full"):
        # Remove the held-out target period and its measured guard from every
        # source station. Strict inequalities retain data outside the guard.
        train &= ((time_index < max(0, lo - guard)) |
                  (time_index >= min(T, hi + guard)))

    ytrain = dat["residual"].reshape(-1)[train]
    if len(ytrain) < 50:
        return None

    model = make_model(learner, ridge_alpha)
    model.fit(X[train], ytrain)

    target_rows = np.where(station_index == target)[0]
    test_rows = target_rows[lo:hi]
    pred = model.predict(X[test_rows])

    raw_err = np.mean(np.abs(dat["residual"][target, lo:hi]))
    corrected_err = np.mean(np.abs(dat["residual"][target, lo:hi] - pred))
    delta = float(np.log(raw_err / corrected_err))
    return delta, guard, len(src)


def evaluate(dat, spec, design, test_start, test_len,
             radius_km=50.0, ridge_alpha=2.0, learner="ridge"):
    X = design_matrix(dat, spec, learner)
    deltas, guards, nsrc = [], [], []
    for target in range(len(dat["ids"])):
        z = fit_one_target(dat, X, target, design, test_start, test_len,
                           radius_km, ridge_alpha, learner)
        if z is None:
            continue
        d, g, ns = z
        deltas.append(d)
        guards.append(g)
        nsrc.append(ns)

    d = np.asarray(deltas, float)
    if len(d) < 5:
        return {
            "n_targets": len(d), "median_delta": np.nan, "mean_delta": np.nan,
            "win_rate": np.nan, "sign_p": np.nan, "positive_verdict": False,
            "guard_days_median": np.nan, "n_src_median": np.nan,
        }

    wins = int((d > 0).sum())
    sign_p = float(stats.binomtest(wins, len(d), 0.5,
                                   alternative="greater").pvalue)
    median_delta = float(np.median(d))
    return {
        "n_targets": len(d),
        "median_delta": median_delta,
        "mean_delta": float(np.mean(d)),
        "win_rate": float(wins / len(d)),
        "sign_p": sign_p,
        "positive_verdict": bool(median_delta > 0 and sign_p < 0.05),
        "guard_days_median": float(np.median(guards)) if guards else 0.0,
        "n_src_median": float(np.median(nsrc)),
    }


def scenario_grid():
    keys = ("beta_state", "spatial", "temporal", "grid", "geometry")
    for values in itertools.product(
        ("absent", "present"),
        ("off", "on"),
        ("off", "on"),
        ("off", "on"),
        ("dense", "sparse"),
    ):
        yield dict(zip(keys, values))


def run_task(task):
    scenario, rep, base_seed, cfg = task
    beta = 0.0 if scenario["beta_state"] == "absent" else cfg["beta_signal"]
    seed = int(base_seed + rep + 10000 * task[0]["_scenario_id"])

    dat = simulate_field(
        seed=seed,
        beta=beta,
        spatial_on=scenario["spatial"] == "on",
        temporal_on=scenario["temporal"] == "on",
        grid_on=scenario["grid"] == "on",
        geometry=scenario["geometry"],
        n_days=cfg["n_days"],
        sigma_noise=cfg["sigma_noise"],
    )

    rng = np.random.default_rng(seed + 777)
    margin = cfg["test_margin_days"]
    test_len = cfg["test_len_days"]
    test_start = int(rng.integers(margin, cfg["n_days"] - margin - test_len + 1))

    rows = []
    wanted = cfg.get("designs") or ",".join(DESIGNS)
    wanted = [d.strip() for d in wanted.split(",") if d.strip()]
    for spec in SPECS:
        for design in [d for d in DESIGNS if d in wanted]:
            out = evaluate(
                dat, spec, design, test_start, test_len,
                radius_km=cfg["radius_km"],
                ridge_alpha=cfg["ridge_alpha"],
                learner=cfg.get("learner", "ridge"),
            )
            row = {
                "scenario_id": scenario["_scenario_id"],
                "replicate": rep,
                "seed": seed,
                "beta_state": scenario["beta_state"],
                "beta": beta,
                "spatial": scenario["spatial"],
                "temporal": scenario["temporal"],
                "grid": scenario["grid"],
                "geometry": scenario["geometry"],
                "spec": spec,
                "design": design,
                "learner": cfg.get("learner", "ridge"),
                "test_start": test_start,
                "test_len_days": test_len,
                **out,
            }
            rows.append(row)
    return rows


def summarize(df):
    group = ["beta_state", "spatial", "temporal", "grid",
             "geometry", "spec", "design"]
    out = (
        df.groupby(group, dropna=False)
          .agg(
              n_rep=("replicate", "nunique"),
              verdict_rate=("positive_verdict", "mean"),
              median_of_delta=("median_delta", "median"),
              delta_q25=("median_delta", lambda x: np.nanpercentile(x, 25)),
              delta_q75=("median_delta", lambda x: np.nanpercentile(x, 75)),
              median_win_rate=("win_rate", "median"),
              median_guard_days=("guard_days_median", "median"),
              median_sources=("n_src_median", "median"),
          )
          .reset_index()
    )
    out["rate_type"] = np.where(
        out["beta_state"].eq("absent"),
        "false-positive transfer verdict rate",
        "true-transfer detection rate",
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outdir", default="results/serra_sim")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--smoke", action="store_true",
                    help="2 scenarios x 1 replicate for an integrity check")
    ap.add_argument("--radius-km", type=float, default=50.0)
    ap.add_argument("--beta-signal", type=float, default=1.0)
    ap.add_argument("--n-days", type=int, default=120)
    ap.add_argument("--test-len-days", type=int, default=20)
    ap.add_argument("--test-margin-days", type=int, default=20)
    ap.add_argument("--ridge-alpha", type=float, default=2.0)
    ap.add_argument("--designs", default="",
                    help="comma-separated subset of the validation ladder to run "
                         "(station,space,space_time,full). Empty = all four.")
    ap.add_argument("--learner", choices=["ridge", "hgb"], default="ridge",
                    help="ridge: the reported RBF-basis ridge learner. "
                         "hgb: histogram gradient boosting on the raw feature set, "
                         "matching the empirical audit's estimator.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "version": "v2",
        "n_rep": args.n_rep,
        "base_seed": args.seed,
        "radius_km": args.radius_km,
        "beta_signal": args.beta_signal,
        "n_days": args.n_days,
        "test_len_days": args.test_len_days,
        "test_margin_days": args.test_margin_days,
        "ridge_alpha": args.ridge_alpha,
        "learner": args.learner,
        "designs": args.designs or ",".join(DESIGNS),
        "sigma_noise": 0.75,
        "guard_rule": "first source-network median per-station |ACF| <= 0.10; max 14 d",
        "grid_assignment": "observed audit_v7 rule: round(lat/0.75), round(lon/0.75); 9 cells",
        "spatial_range_km": 70.0,
        "temporal_strong_theoretical_acf": {
            "1_day": 0.35, "3_days": 0.19, "10_days": 0.08
        },
        "primary_designs": list(DESIGNS),
        "feature_specs": list(SPECS),
        "primary_radius_sensitivity": [25, 50, 100],
        "note": (
            "Primary run uses R=50 km. Repeat --radius-km 25 and 100 as "
            "pre-declared sensitivity runs, not as separate hypothesis fishing."
        ),
        "structural_invariants": [
            "19 observed station coordinates",
            "9 observed CAMS 0.75-degree cells",
            "same-cell stations share an exactly identical mechanistic predictor series",
            "dense and sparse geometries retain the same cell membership to isolate geometry",
        ],
    }

    # Fail-fast structural validation before any Monte Carlo work.
    _ids, _x, _y = local_xy_km(1.0)
    _cells, _cidx, _ncell = observed_cams_grid_ids(_ids)
    _d = pairdist(_x, _y)
    _nn = np.min(np.where(_d == 0, np.inf, _d), axis=1)
    assert len(_ids) == 19
    assert _ncell == 9
    if args.smoke:
        _probe = simulate_field(args.seed, 0.0, False, False, False, "dense",
                                n_days=min(args.n_days, 40))
        same_cell_maxdiff = 0.0
        for c in np.unique(_probe["cell"]):
            members = np.where(_probe["cell"] == c)[0]
            if len(members) > 1:
                ref = _probe["mechanistic"][members[0]]
                same_cell_maxdiff = max(
                    same_cell_maxdiff,
                    float(np.max(np.abs(_probe["mechanistic"][members] - ref)))
                )
        assert same_cell_maxdiff == 0.0
        print(f"STRUCTURE OK: stations={len(_ids)} cells={_ncell} "
              f"median_NN={np.median(_nn):.2f} km same_cell_maxdiff={same_cell_maxdiff:.1f}")

    scenarios = list(scenario_grid())
    for i, s in enumerate(scenarios):
        s["_scenario_id"] = i

    if args.smoke:
        # Four corner scenarios: clean/all-dependence x absent/present signal.
        # This checks both false-transfer and positive-control code paths.
        scenarios = [
            s for s in scenarios
            if (s["geometry"] == "dense"
                and ((s["spatial"], s["temporal"], s["grid"]) == ("off","off","off")
                     or (s["spatial"], s["temporal"], s["grid"]) == ("on","on","on")))
        ]
        n_rep = 1
    else:
        n_rep = args.n_rep

    tasks = [
        (s, rep, args.seed, cfg)
        for s in scenarios for rep in range(n_rep)
    ]

    rows = []
    if args.jobs == 1:
        for task in tasks:
            rows.extend(run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for fut in as_completed(futs):
                rows.extend(fut.result())

    df = pd.DataFrame(rows)
    summary = summarize(df)

    df.to_csv(outdir / "simulation_results.csv", index=False)
    summary.to_csv(outdir / "simulation_summary.csv", index=False)
    with open(outdir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote {len(df):,} result rows")
    print(outdir / "simulation_results.csv")
    print(outdir / "simulation_summary.csv")
    print(outdir / "simulation_config.json")

    if args.smoke:
        print("\nSMOKE SUMMARY")
        show = summary[[
            "beta_state","spatial","temporal","grid","geometry",
            "spec","design","verdict_rate","median_of_delta","median_guard_days"
        ]]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

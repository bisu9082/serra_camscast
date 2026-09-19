#!/usr/bin/env python
import importlib.util
import numpy as np

import os
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serra_simulation_v2.py")
spec = importlib.util.spec_from_file_location("serra_sim_v2", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 1) Exact observed station and cell structure.
ids, x, y = m.local_xy_km(1.0)
cells, cidx, ncell = m.observed_cams_grid_ids(ids)
assert len(ids) == 19
assert ncell == 9
D = m.pairdist(x, y)
nn = np.min(np.where(D == 0, np.inf, D), axis=1)
assert abs(float(np.median(nn)) - 6.5499) < 0.02

# 2) Dense and sparse geometry isolate distance while retaining grid membership.
ids2, xs, ys = m.local_xy_km(2.0)
cells2, cidx2, ncell2 = m.observed_cams_grid_ids(ids2)
assert np.array_equal(ids, ids2)
assert np.array_equal(cells, cells2)
assert ncell2 == 9
Ds = m.pairdist(xs, ys)
nns = np.min(np.where(Ds == 0, np.inf, Ds), axis=1)
assert abs(float(np.median(nns)) / float(np.median(nn)) - 2.0) < 1e-10

# 3) Same-cell CAMS-like predictor identity is exact in both geometries.
for geom in ("dense", "sparse"):
    dat = m.simulate_field(12345, 0.0, True, True, True, geom, n_days=60)
    assert len(np.unique(dat["cell"])) == 9
    for c in np.unique(dat["cell"]):
        members = np.where(dat["cell"] == c)[0]
        if len(members) > 1:
            ref = dat["mechanistic"][members[0]]
            assert np.max(np.abs(dat["mechanistic"][members] - ref)) == 0.0

# 4) Full-audit source rule removes same-cell stations beyond a permissive radius.
dat = m.simulate_field(123, 0.0, True, True, True, "dense", n_days=60)
counts = [np.sum(dat["cell"] == c) for c in dat["cell"]]
target = int(np.argmax(counts))
D = m.pairdist(dat["x"], dat["y"])
space_src = np.array([j for j in range(19) if j != target and D[target, j] > 0.0])
full_src = space_src[dat["cell"][space_src] != dat["cell"][target]]
assert not np.any(dat["cell"][full_src] == dat["cell"][target])
assert len(full_src) < len(space_src)

# 5) Positive-control beta=1.0 produces a structural transferable component.
d0 = m.simulate_field(777, 0.0, False, False, False, "dense", n_days=60)
d1 = m.simulate_field(777, 1.0, False, False, False, "dense", n_days=60)
assert not np.allclose(d0["residual"], d1["residual"])

# 6) Data-adaptive temporal guard uses source residuals and returns legal bounds.
guard = m.estimate_guard_days(d1["residual"], np.arange(1, 19), threshold=0.10, max_days=14)
assert 1 <= guard <= 14

print("ALL V2 STRUCTURAL TESTS PASSED")
print(
    f"stations={len(ids)} cells={ncell} "
    f"dense_median_NN={np.median(nn):.4f} km "
    f"sparse_median_NN={np.median(nns):.4f} km guard_check={guard} d"
)

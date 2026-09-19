"""Three-line check that a fresh clone runs. Exits non-zero if it does not.

    pip install -r requirements.txt
    python smoke_test.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
os.environ.setdefault("CAMSCAST_DATA",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed"))

from loso_v2 import DATA, PM10C                      # noqa: E402

n_st = len(DATA)
n_pt = sum(len(v[0]) for v in DATA.values())
print(f"loaded {n_st} stations, {n_pt} matched three-hourly samples")
assert n_st == 19, f"expected 19 stations, got {n_st}"
assert n_pt == 20818, f"expected 20818 samples, got {n_pt}"

import numpy as np                                    # noqa: E402
for sid, (df, *_rest) in DATA.items():
    assert df.index.is_monotonic_increasing, f"{sid}: timestamps not sorted"
    assert not df.isna().any().any(), f"{sid}: missing values present"
    assert np.abs(df["resid"] - (df["obs"] - df["cams"])).max() < 1e-4, f"{sid}: resid mismatch"
print("integrity checks passed")
print("OK")

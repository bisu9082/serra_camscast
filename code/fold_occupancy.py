"""Fold occupancy under calendar-equal versus equal-count (quantile) time blocks.

The second of the three conventions the paper is about. Calendar-equal blocks are
equal in time, not in data: this record has four months of 2023 with no
observations at all, so three of six calendar folds are nearly empty at every
station and the nominal six-fold temporal control is three-fold in substance.

Reports, over the primary quality-controlled records (flat-line threshold K=6):
the fold sizes at one representative station, the smallest and largest fold
anywhere in the network, and the within-station ratio of largest to smallest fold
under each blocking scheme.

Writes fold_occupancy.json.
"""
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DDIR = os.environ.get("CAMSCAST_DATA", os.path.join(ROOT, "data_processed", "aligned_qc6"))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(ROOT, "results", "fold_occupancy.json"))
NBLK = 6
EXAMPLE = "369959"

files = sorted(glob.glob(os.path.join(DDIR, "aligned_pm10_*.csv")))
assert files, f"no aligned PM10 records under {DDIR}"


def occupancy(idx, edges):
    n = len(edges) - 1
    return [int(((idx >= edges[b]) &
                 (idx < (edges[b + 1] if b < n - 1 else edges[b + 1] + pd.Timedelta("1h")))).sum())
            for b in range(n)]


rows = []
for f in files:
    sid = os.path.basename(f).split("_")[-1].replace(".csv", "")
    idx = pd.read_csv(f, index_col=0, parse_dates=True).index
    cal = occupancy(idx, pd.date_range(idx.min(), idx.max(), periods=NBLK + 1))
    srt = idx.sort_values()
    pos = np.linspace(0, len(srt) - 1, NBLK + 1).round().astype(int)
    qe = pd.DatetimeIndex(pd.Series(srt[pos]).drop_duplicates().values)
    qnt = occupancy(idx, qe)
    rows.append(dict(sid=sid, calendar=cal, quantile=qnt,
                     calendar_ratio=max(cal) / max(min(cal), 1),
                     quantile_ratio=max(qnt) / max(min(qnt), 1)))

d = pd.DataFrame(rows)
ex = d[d.sid == EXAMPLE]
out = dict(
    n_stations=len(d), n_blocks=NBLK, data_dir=os.path.basename(DDIR),
    calendar_min_fold=int(min(min(r["calendar"]) for r in rows)),
    calendar_max_fold=int(max(max(r["calendar"]) for r in rows)),
    calendar_ratio_median=float(d["calendar_ratio"].median()),
    calendar_ratio_min=float(d["calendar_ratio"].min()),
    calendar_ratio_max=float(d["calendar_ratio"].max()),
    quantile_min_fold=int(min(min(r["quantile"]) for r in rows)),
    quantile_max_fold=int(max(max(r["quantile"]) for r in rows)),
    quantile_ratio_median=float(d["quantile_ratio"].median()),
    quantile_ratio_max=float(d["quantile_ratio"].max()),
    example_station=EXAMPLE,
    example_calendar_folds=[int(x) for x in ex["calendar"].values[0]] if len(ex) else None,
    example_quantile_folds=[int(x) for x in ex["quantile"].values[0]] if len(ex) else None,
    per_station={r["sid"]: dict(calendar=r["calendar"], quantile=r["quantile"]) for r in rows})

print(f"calendar : folds range {out['calendar_min_fold']} to {out['calendar_max_fold']}, "
      f"within-station ratio median {out['calendar_ratio_median']:.0f} "
      f"({out['calendar_ratio_min']:.0f} to {out['calendar_ratio_max']:.0f})")
print(f"quantile : folds range {out['quantile_min_fold']} to {out['quantile_max_fold']}, "
      f"within-station ratio median {out['quantile_ratio_median']:.3f}")
if out["example_calendar_folds"]:
    print(f"station {EXAMPLE}: calendar {out['example_calendar_folds']}  "
          f"quantile {out['example_quantile_folds']}")

json.dump(out, open(OUT, "w"), indent=1)
print("saved", os.path.basename(OUT))

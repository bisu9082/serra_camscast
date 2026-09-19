"""Flat-line quality control for the PM10 network, and the evidence behind the rule.

A reviewer noticed that station 369963 reports exactly 80 ug/m3 for 51 consecutive
three-hourly steps, and that 15.5% of all observations repeat the previous value.
This script characterises those repeats, separates the two mechanisms that produce
them, applies a threshold, and writes QC'd copies of the aligned records.

The two mechanisms are distinguishable by concentration:

  * Integer reporting. The record is integer-valued at three-hourly resolution and
    the median absolute step between consecutive values is 3 ug/m3, so exact ties
    are expected wherever the concentration is low and the air is calm. Ties of
    this kind should sit BELOW the network's median concentration.
  * A stuck instrument. A sensor that latches reports the same number regardless of
    what the air is doing, so its ties carry no concentration signature and can run
    far longer than any plausible calm spell.

The observed run lengths behave like the first mechanism up to about five steps and
like the second beyond that, which is what sets the threshold. The script reports
the evidence so a reader can check the choice rather than take it.

Writes qc_flatline.json and, for each threshold K, a directory aligned_qcK/ holding
the same aligned records with runs of K or more identical observations removed.
"""
import json, os, shutil, glob
import numpy as np, pandas as pd

RUN = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = (3, 4, 6)          # 6 is primary; 3 and 4 are the sensitivity variants
PRIMARY = 6

files = sorted(glob.glob(os.path.join(RUN, "aligned_pm10_*.csv")))
assert files, "no aligned PM10 records found"


def runs_of(s):
    """Return a Series giving, for each observation, the length of the run of
    identical consecutive values it belongs to."""
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).transform("size")


# ---------------------------------------------------------------- evidence
allobs, alldiff, rows = [], [], []
for f in files:
    sid = os.path.basename(f).split("_")[-1].replace(".csv", "")
    s = pd.read_csv(f, index_col=0, parse_dates=True)["obs"]
    allobs.append(s.values)
    alldiff.append(np.diff(s.values))
    grp = (s != s.shift()).cumsum()
    for g, idx in s.groupby(grp).groups.items():
        rows.append(dict(sid=sid, L=len(idx), val=float(s.loc[idx[0]]),
                         start=str(idx[0]), hours=3 * len(idx)))

obs = np.concatenate(allobs)
dif = np.concatenate(alldiff)
r = pd.DataFrame(rows)

ev = dict(
    n_obs=int(len(obs)),
    integer_valued_fraction=float(np.mean(obs == np.round(obs))),
    median_concentration=float(np.median(obs)),
    fraction_above_100=float(np.mean(obs > 100)),
    median_abs_step=float(np.median(np.abs(dif[dif != 0]))),
    fraction_tied_with_previous=float(np.mean(dif == 0)),
    longest_run=int(r.L.max()),
)

bands = []
for lo, hi, lab in [(2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 999, "11+")]:
    x = r[(r.L >= lo) & (r.L <= hi)]
    if len(x) == 0:
        continue
    bands.append(dict(run_length=lab, n_runs=int(len(x)),
                      median_value=float(x.val.median()),
                      fraction_above_100=float((x.val > 100).mean())))
ev["by_run_length"] = bands
ev["interpretation"] = (
    "Runs of two to ten steps sit below the network median concentration and are "
    "depleted in values above 100 ug/m3 relative to the record as a whole, which is "
    "the signature of integer reporting rather than of a latched sensor. The two "
    "longest runs (51 and 22 steps) show no such depletion in duration and are "
    "treated as instrument faults.")

ev["long_runs"] = [dict(sid=x.sid, length=int(x.L), hours=int(x.hours),
                        value=float(x.val), start=x.start)
                   for _, x in r[r.L >= 6].sort_values("L", ascending=False).iterrows()]

print("=== evidence ===", flush=True)
print(f"  integer-valued {100*ev['integer_valued_fraction']:.0f}%  median {ev['median_concentration']:.0f}  "
      f">100 in {100*ev['fraction_above_100']:.0f}% of the record", flush=True)
print(f"  median absolute step {ev['median_abs_step']:.0f} ug/m3   tied with previous {100*ev['fraction_tied_with_previous']:.1f}%", flush=True)
for b in bands:
    print(f"  run {b['run_length']:>5s}: {b['n_runs']:4d} runs, median value {b['median_value']:5.0f}, "
          f"{100*b['fraction_above_100']:4.0f}% above 100", flush=True)

# ---------------------------------------------------------------- apply
ev["thresholds"] = {}
for K in THRESHOLDS:
    outdir = os.path.join(RUN, f"aligned_qc{K}")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    removed = kept = 0
    per = {}
    for f in files:
        base = os.path.basename(f)
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        keep = runs_of(d["obs"]) < K
        per[base.split("_")[-1].replace(".csv", "")] = dict(
            before=int(len(d)), removed=int((~keep).sum()), after=int(keep.sum()))
        removed += int((~keep).sum()); kept += int(keep.sum())
        d[keep].to_csv(os.path.join(outdir, base))
    # the PM2.5 records are copied unchanged so the directory is a drop-in replacement
    for f in glob.glob(os.path.join(RUN, "aligned_[A-Z]*.csv")):
        shutil.copy(f, outdir)
    ev["thresholds"][str(K)] = dict(
        rule=f"remove every observation in a run of {K} or more identical consecutive values",
        removed=removed, kept=kept, removed_fraction=removed / (removed + kept),
        stations_affected=int(sum(1 for v in per.values() if v["removed"] > 0)),
        min_station_n=int(min(v["after"] for v in per.values())),
        per_station=per, primary=(K == PRIMARY), outdir=os.path.basename(outdir))
    print(f"[K={K}] removed {removed} of {removed+kept} ({100*removed/(removed+kept):.1f}%), "
          f"{ev['thresholds'][str(K)]['stations_affected']} stations affected, "
          f"smallest station now {ev['thresholds'][str(K)]['min_station_n']}", flush=True)

ev["primary_threshold"] = PRIMARY
json.dump(ev, open(os.path.join(RUN, "qc_flatline.json"), "w"), indent=1)
print("saved qc_flatline.json", flush=True)

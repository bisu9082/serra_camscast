"""Merge the placebo worker shards and compute the permutation p-values.

placebo_v7.py runs several workers, each writing its own shard so that two
processes never contend for one file. This merges the shards into placebo_v7.json
and reports, for each radius, the observed statistic against the permutation null.

The p-value is the standard permutation estimate (r+1)/(n+1), which is why the
number of draws bounds how small it can be: 30 draws cannot go below 0.032, and
99 cannot go below 0.010. The count actually attained at each radius is reported
alongside the p-value so a reader can see the resolution rather than assume it.

Two statistics are carried: the median log error ratio across the surviving
targets, which is the paper's primary estimand, and the pooled one.

Writes placebo_v7.json.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(HERE, "placebo_v7.json"))
SHARDS = sorted(glob.glob(OUT + ".w*"))
assert SHARDS, f"no worker shards beside {OUT}"

merged = {}
for s in SHARDS:
    d = json.load(open(s))
    for R, rec in d.items():
        m = merged.setdefault(R, {"observed": None, "perm": {}})
        if rec.get("observed") and m["observed"] is None:
            m["observed"] = rec["observed"]
        m["perm"].update(rec.get("perm", {}))

# A radius is reported only when it carries the full permutation count. A partial
# radius is kept in the shards so the run can be resumed, but reporting it would mix
# resolutions within one table: 11 draws cannot go below p = 0.083.
COMPLETE = int(os.environ.get("MIN_PERM", 99))

out = {"_meta": dict(
    shards=[os.path.basename(s) for s in SHARDS],
    min_perm_to_report=COMPLETE,
    specification="flat-line QC K=6, quantile time blocks, ten-day guard, "
                  "buffer + time + cell control, coordinate-bearing feature set",
    estimand="delta = log(MAE_raw / MAE_transfer), median across surviving targets",
    p_definition="(number of permutations at least as extreme + 1) / (permutations + 1)",
    null="each station is assigned another station's latitude and longitude")}

print(f"{'radius':>8}{'perms':>7}{'min p':>8}{'observed':>10}{'null mean':>11}"
      f"{'null 95% band':>22}{'p(true>=perm)':>15}{'p(two-sided)':>14}")
for R in sorted(merged, key=lambda k: int(k[1:])):
    rec = merged[R]
    if rec["observed"] is None or len(rec["perm"]) < COMPLETE:
        print(f"{R:>8}  not reported: {len(rec['perm'])} of {COMPLETE} permutations"
              f"{' (observed statistic not yet computed)' if rec['observed'] is None else ''}")
        out.setdefault("_partial", {})[R] = len(rec["perm"])
        continue
    o = rec["observed"]["delta_median"]
    v = np.array([p["delta_median"] for p in rec["perm"].values()])
    n = len(v)
    p_hi = (np.sum(v >= o) + 1) / (n + 1)          # scrambling does at least as well
    p_two = (np.sum(np.abs(v - v.mean()) >= abs(o - v.mean())) + 1) / (n + 1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    out[R] = dict(
        n_perm=n, min_attainable_p=1.0 / (n + 1),
        observed_median=float(o),
        observed_pooled=float(rec["observed"]["delta_pooled"]),
        observed_wins=rec["observed"]["wins"], n_targets=rec["observed"]["n_targets"],
        null_mean=float(v.mean()), null_sd=float(v.std(ddof=1)),
        null_median=float(np.median(v)), null_lo=float(lo), null_hi=float(hi),
        p_true_ge_perm=float(p_hi), p_two_sided=float(p_two),
        frac_perm_at_least_true=float(np.mean(v >= o)),
        perm_median=[float(x) for x in v])
    print(f"{R:>8}{n:>7}{1/(n+1):>8.3f}{o:>+10.4f}{v.mean():>+11.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p_hi:>15.4f}{p_two:>14.4f}")

json.dump(out, open(OUT, "w"), indent=1)
print("\nsaved", os.path.basename(OUT))
print("Reading: p(true>=perm) small means scrambled coordinates rarely match the true "
      "assignment, so the coordinate features carry structure the permutation destroys.")

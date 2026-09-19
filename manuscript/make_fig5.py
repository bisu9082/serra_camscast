"""Figure 5 --- the three checks that ask whether the audit's own controls are real.

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
Panel (a) asks what geometry each design actually validates, against the geometry an
unmonitored site in this region actually presents. Panel (b) asks what the coordinate
features are doing, against a permutation null with enough draws to resolve a p-value.
Panel (c) asks whether the leave-one-grid-cell-out stage is picking up covariate
sharing or merely the position of the CAMS grid.

Every value is read from a computed artefact; nothing is illustrative.

Palette matches Figure 4: #C94F4A (with coordinates) against #2E8F94 (coordinate-free),
identity repeated by marker and direct label rather than carried by colour alone.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RUN = os.environ.get("CAMSCAST_RUN", _ROOT + "/results")
OUTPNG = os.environ.get("CAMSCAST_FIG", _HERE + "/Fig5.png")

WITH, FREE = "#C94F4A", "#2E8F94"
INK, MUTED, GRID = "#2F2A24", "#6B655C", "#DCD7CE"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#4A4A4A", "axes.linewidth": 1.0,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
})
FS_L, FS_T, FS_TT, FS_P, FS_LEG = 14, 12, 13, 20, 11.5

K = json.load(open(f"{RUN}/knndm.json"))
PL = json.load(open(f"{RUN}/placebo_v7.json")) if os.path.exists(f"{RUN}/placebo_v7.json") else None
GO = json.load(open(f"{RUN}/grid_offset.json")) if os.path.exists(f"{RUN}/grid_offset.json") else None

fig = plt.figure(figsize=(16.5, 5.4))
G = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                      left=0.055, right=0.985, top=0.86, bottom=0.16)


def tag(ax, s, dx=-0.13):
    ax.text(dx, 1.13, s, transform=ax.transAxes, fontsize=FS_P, fontweight="bold",
            color=INK, va="top", ha="left")


# ---------------------------------------------------------------- (a) geometry
axa = fig.add_subplot(G[0, 0])
labels, vals, cols = [], [], []
labels.append("plain LOSO\n(no buffer, no cell stage)")
vals.append(K["nn_station_to_station"]["median"]); cols.append("#B0A99F")
for R in (0, 25, 50, 75, 100):
    k = f"R{R}"
    if k in K["buffered_designs"]:
        labels.append(f"audited design\n$R$ = {R} km")
        vals.append(K["buffered_designs"][k]["median_nn_to_source"]); cols.append(FREE)
y = np.arange(len(vals))[::-1]
axa.barh(y, vals, color=cols, edgecolor=cols, height=0.62, zorder=3)
for yi, v in zip(y, vals):
    axa.text(v + 2, yi, f"{v:.1f}", va="center", fontsize=11.5, color=INK, fontweight="bold")
dep = K["nn_prediction_to_station"]["median"]
lo, hi = K["nn_prediction_to_station"]["q25"], K["nn_prediction_to_station"]["q75"]
axa.axvspan(lo, hi, color=WITH, alpha=0.13, zorder=1)
axa.axvline(dep, color=WITH, lw=2.2, ls="--", zorder=4)
axa.text(dep + 1.5, y[0] + 0.62,
         f"deployment: an unmonitored site\nin this region sits {dep:.0f} km\nfrom the nearest monitor",
         fontsize=10.5, color=WITH, fontweight="bold", va="top")
axa.set_yticks(y); axa.set_yticklabels(labels, fontsize=11)
axa.set_xlabel("Median distance from the held-out target\nto its nearest training station (km)", fontsize=FS_L)
axa.set_xlim(0, 132)
axa.set_title("Plain leave-one-station-out validates\ninterpolation, not deployment", fontsize=FS_TT, color="#333", pad=9)
axa.grid(axis="y", visible=False)
for sp in ("top", "right"):
    axa.spines[sp].set_visible(False)
tag(axa, "a", dx=-0.42)

# ---------------------------------------------------------------- (b) placebo
axb = fig.add_subplot(G[0, 1])
if PL:
    radii = sorted(int(k[1:]) for k in PL
                   if k.startswith("R") and isinstance(PL[k], dict) and "perm_median" in PL[k])
    obs, pvals, nperm, nulls = [], [], [], []
    for R in radii:
        rec = PL[f"R{R}"]
        obs.append(rec["observed_median"])
        nulls.append(np.array(rec["perm_median"]))
        nperm.append(rec["n_perm"])
        pvals.append(rec["p_two_sided"])
    x = np.arange(len(radii))
    rng = np.random.default_rng(1)
    for xi, v in zip(x, nulls):
        jit = (rng.random(len(v)) - 0.5) * 0.26
        axb.scatter(xi + jit, v, s=14, color=MUTED, alpha=0.42, edgecolors="none", zorder=2,
                    label="scrambled coordinates" if xi == 0 else None)
        axb.plot([xi - 0.21, xi + 0.21], [np.median(v)] * 2, color=MUTED, lw=2.4, zorder=3,
                 label="null median" if xi == 0 else None)
    axb.plot(x, obs, color=WITH, lw=2.6, marker="o", ms=12, zorder=5, label="true coordinates")
    for xi, o, p, v in zip(x, obs, pvals, nulls):
        pct = 100 * np.mean(v < o)
        axb.annotate(f"{pct:.0f}th percentile\nof the null,  $p$={p:.2f}", (xi, o),
                     textcoords="offset points", xytext=(0, 17), ha="center",
                     fontsize=10.5, color=WITH, fontweight="bold")
    axb.axhline(0, color="#555", lw=1.1, ls="--", zorder=1)
    axb.set_xticks(x)
    axb.set_xticklabels([f"$R$ = {r} km" for r in radii], fontsize=FS_T)
    axb.set_xlim(-0.5, len(radii) - 0.5)
    axb.set_ylabel(r"Median $\Delta=\log(\mathrm{MAE_{raw}}/\mathrm{MAE_{transfer}})$",
                   fontsize=FS_L)
    axb.legend(fontsize=FS_LEG - 1, frameon=False, loc="lower left")
    axb.set_title(f"Coordinates are inert while neighbours remain,\n"
                  f"a liability once the buffer bites ({nperm[0]} permutations each)",
                  fontsize=FS_TT, color="#333", pad=9)
    axb.grid(axis="x", visible=False)
else:
    axb.text(0.5, 0.5, "placebo_v7.json not yet written", ha="center", va="center",
             transform=axb.transAxes, fontsize=13, color=MUTED)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)
tag(axb, "b")

# ---------------------------------------------------------------- (c) grid offset
axc = fig.add_subplot(G[0, 2])
if GO:
    # offsets are stored absolute; (0.5, 0.5) is the CAMS grid the manuscript uses, so
    # they are shown relative to it, matching Table S33
    keys = sorted((k for k in GO if k.startswith("off_")),
                  key=lambda k: (GO[k]["offset"][0] - 0.5, GO[k]["offset"][1] - 0.5))
    rad = [r for r in ("R0", "R100") if all(r in GO[k] for k in keys)]
    xs = np.arange(len(keys))
    lbl = [f"({GO[k]['offset'][0]-0.5:+.2f},\n{GO[k]['offset'][1]-0.5:+.2f})" for k in keys]
    for r, col, mk in zip(rad, (FREE, WITH), ("o", "D")):
        v = [GO[k][r]["delta_median"] for k in keys]
        axc.plot(xs, v, color=col, lw=2.2, marker=mk, ms=8, zorder=4,
                 label=f"$R$ = {r[1:]} km")
        axc.annotate(f"span {max(v)-min(v):.3f}", (xs[-1], v[-1]),
                     textcoords="offset points", xytext=(-6, 12 if r == "R0" else -20),
                     ha="right", fontsize=10.5, color=col, fontweight="bold")
    # mark the offset that is the CAMS grid the manuscript actually uses
    home = [i for i, k in enumerate(keys)
            if abs(GO[k]["offset"][0] - 0.5) < 1e-9 and abs(GO[k]["offset"][1] - 0.5) < 1e-9]
    if home:
        i = home[0]
        axc.axvline(i, color=INK, lw=1.0, ls=":", zorder=1)
        axc.annotate("the CAMS grid\nused in this paper", (i, GO[keys[i]]["R0"]["delta_median"]),
                     textcoords="offset points", xytext=(-8, -34), ha="right",
                     fontsize=10, color=INK, style="italic")
    axc.axhline(0, color="#555", lw=1.1, ls="--", zorder=1)
    axc.set_xticks(xs); axc.set_xticklabels(lbl, fontsize=10)
    axc.set_xlabel("Grid origin offset from the CAMS grid\n(fraction of a $0.75^\\circ$ cell)", fontsize=FS_L)
    axc.set_ylabel(r"Median $\Delta$, coordinate-bearing", fontsize=FS_L)
    axc.legend(fontsize=FS_LEG, frameon=False, loc="best")
    axc.set_title("Moving the CAMS grid origin does not\nmove the verdict", fontsize=FS_TT, color="#333", pad=9)
else:
    axc.text(0.5, 0.5, "grid_offset.json not yet written", ha="center", va="center",
             transform=axc.transAxes, fontsize=13, color=MUTED)
for sp in ("top", "right"):
    axc.spines[sp].set_visible(False)
tag(axc, "c")

fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print("saved", os.path.basename(OUTPNG))
if PL:
    print("panel b:", [(f"R{r}", round(o, 4), round(p, 3), n)
                       for r, o, p, n in zip(radii, obs, pvals, nperm)])
if GO:
    print("panel c:", {r: [round(GO[k][r]["delta_median"], 4) for k in keys] for r in rad})

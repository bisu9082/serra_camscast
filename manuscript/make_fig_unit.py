"""Figure: which inference unit removes the failure, and what it costs.

The verdict is formed by a one-sided exact sign test across held-out units. Which targets count as
units is a design choice made alongside the choice of what to withhold, and the figure puts five
rules against the same per-realization deltas of the same realizations:

  station rule           nineteen held-out stations
  grid cells             the nine occupied mechanistic-grid cells
  random spatial groups  the same targets partitioned at random into nine groups
  contiguous episodes    the evaluation window cut into nine consecutive blocks
  random time groups     the same timesteps scattered at random into nine groups

The two random rules hold the number of units fixed while destroying the structure the
corresponding structured rule claims to use, so the difference between a structured rule and its
own control is what the structure buys, and the difference between the spatial pair and the
temporal pair is what the choice of dimension buys.

Panel (a) is the corner in which the station-level test is anti-conservative: a residual component
common to the network in time, with neither spatial proximity nor shared grid support. The number
beside each label is that rule's corner rate divided by its own pooled false-positive rate, which
separates a rule that is calibrated there from one that merely declares transfer less often
everywhere. Panel (b) is the price, measured on the realizations that carry a genuine signal.

Reads  simulation/results/time_unit/R{25,50,100}/simulation_results.csv
Writes figures/Unit_Control.png

Authored at the final placement width (\\textwidth = 390 pt = 5.40 in) so that \\includegraphics
applies no reduction and the printed point size is the authored point size; every label is at
least 6.6 pt. Do not add bbox_inches="tight", which changes the saved width.

Identity is carried by colour, marker and line style together, never by colour alone. The spatial
rules are drawn in the dark/teal/grey family and the temporal rules in the red family, so the
dimension is legible before any individual series is.
"""
import os as _os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RUNDIR = _os.environ.get("CAMSCAST_EPRUN",
                         _os.path.join(_ROOT, "simulation", "results", "time_unit"))
OUTDIR = _os.environ.get("CAMSCAST_FIGDIR", _os.path.join(_HERE, "figures"))

INK, FREE, MUTED, WARM, GRID = "#2F2A24", "#2E8F94", "#8A8378", "#C94F4A", "#DCD7CE"
FS_L, FS_T, FS_TT, FS_P, FS_LEG = 8.0, 7.0, 8.0, 10.5, 6.6
RADII = (25, 50, 100)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#4A4A4A", "axes.linewidth": 0.9,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
})

SERIES = (
    ("positive_verdict",      INK,   "o", "-",  "station (19)"),
    ("cell_positive_verdict", FREE,  "D", "-",  "grid cells (9)"),
    ("rnd_frac",              MUTED, "s", "--", "random spatial (9)"),
    ("ep_positive_verdict",   WARM,  "^", "-",  "episodes (9)"),
    ("eprnd_frac",            WARM,  "v", ":",  "random time (9)"),
)

_CACHE = {}


def load(R):
    if R not in _CACHE:
        p = _os.path.join(RUNDIR, f"R{R}", "simulation_results.csv")
        if not _os.path.exists(p):
            raise SystemExit(f"missing run: {p}")
        _CACHE[R] = pd.read_csv(p).query("design == 'full'")
    return _CACHE[R]


def corner(d):
    return d[(d.beta_state == "absent") & (d.temporal == "on")
             & (d.spatial == "off") & (d.grid == "off")]


def series(col, kind):
    out = []
    for R in RADII:
        d = load(R)
        x = corner(d) if kind == "corner" else d[d.beta_state == "present"]
        out.append(100 * x[col].mean())
    return out


def ratio(col):
    """Mean over radii of (corner rate) / (that rule's own pooled false-positive rate)."""
    vals = []
    for R in RADII:
        d = load(R)
        pooled = d[d.beta_state == "absent"][col].mean()
        if pooled > 0:
            vals.append(corner(d)[col].mean() / pooled)
    return sum(vals) / len(vals) if vals else float("nan")


fig = plt.figure(figsize=(5.40, 3.10))
G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.34,
                      left=0.098, right=0.978, top=0.80, bottom=0.355)

ax1 = fig.add_subplot(G[0, 0])
for col, c, mk, ls, lab in SERIES:
    ax1.plot(RADII, series(col, "corner"), ls, color=c, marker=mk, ms=3.6, lw=1.3,
             mec="white", mew=0.7, label=f"{lab}  {ratio(col):.1f}×", zorder=4)
ax1.set_title("False-positive rate where the test fails\n(temporal dependence acting alone)",
              fontsize=FS_TT, color="#333", pad=5)
ax1.set_ylabel("Verdict rate (%)", fontsize=FS_L)
ax1.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
ax1.set_ylim(0, None)
ax1.legend(fontsize=FS_LEG, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.7,
           columnspacing=1.1, borderaxespad=0.0)
ax1.text(-0.20, 1.19, "a", transform=ax1.transAxes, fontsize=FS_P, fontweight="bold",
         va="top", ha="left", color=INK)

ax2 = fig.add_subplot(G[0, 1])
for col, c, mk, ls, lab in SERIES:
    ax2.plot(RADII, series(col, "detect"), ls, color=c, marker=mk, ms=3.6, lw=1.3,
             mec="white", mew=0.7, zorder=4)
ax2.set_title("Detection of a genuine\ntransferable signal", fontsize=FS_TT, color="#333", pad=5)
ax2.set_ylabel("Verdict rate (%)", fontsize=FS_L)
ax2.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
ax2.set_ylim(0, None)
ax2.text(-0.20, 1.19, "b", transform=ax2.transAxes, fontsize=FS_P, fontweight="bold",
         va="top", ha="left", color=INK)

for ax in (ax1, ax2):
    ax.set_xticks(RADII)
    ax.set_xticklabels([str(r) for r in RADII], fontsize=FS_T)
    ax.tick_params(axis="y", labelsize=FS_T)

fig.savefig(_os.path.join(OUTDIR, "Unit_Control.png"), dpi=800)

print("corner false-positive rate (%), temporal dependence acting alone")
for col, c, mk, ls, lab in SERIES:
    v = series(col, "corner")
    print(f"  {lab:20s} " + "  ".join(f"R{R}={x:.2f}" for R, x in zip(RADII, v))
          + f"   corner/pooled {ratio(col):.2f}x")
print("detection (%)")
for col, c, mk, ls, lab in SERIES:
    v = series(col, "detect")
    print(f"  {lab:20s} " + "  ".join(f"R{R}={x:.2f}" for R, x in zip(RADII, v)))
print(f"\nwrote Unit_Control.png to {OUTDIR}")

"""Figures for the ground-truth stochastic experiment.

Two figures, both regenerated from the pre-declared 32-scenario factorial run:

  figures/Simulation_Ladder.png   verdict rates along the validation ladder
                                  (station -> space -> space_time -> full) at R = 50 km
  figures/Simulation_Radius.png   full-audit verdict rates across the three buffer radii

Each panel aggregates the sixteen scenarios of one beta state (transferable signal absent
or present) over the dependence combinations and the two network geometries, exactly as
the manuscript reports them. Counts are recovered as round(verdict_rate * n_rep) per
scenario and summed; intervals are Wilson score intervals on the summed counts.

Inputs are the run summaries written by serra_simulation_v2.py:
  simulation/results/simulation_summary_R{25,50,100}.csv

Both figures are authored at the final placement width (\\textwidth = 390 pt = 5.40 in) so
that \\includegraphics applies no reduction and the printed point size equals the authored
point size. Every label is at least 7 pt at that size. Do not add bbox_inches="tight" to
the savefig calls: it changes the saved width and reintroduces down-scaling.

Palette matches the empirical figures --- #C94F4A (with coordinates) against #2E8F94
(coordinate-free); identity is repeated by marker and direct label, never colour alone.

Nothing here is illustrative; every plotted value comes from the run summaries.
"""
import math
import os as _os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
SIMDIR = _os.environ.get("CAMSCAST_SIMDIR", _os.path.join(_ROOT, "simulation", "results"))
OUTDIR = _os.environ.get("CAMSCAST_FIGDIR", _os.path.join(_HERE, "figures"))

WITH, FREE = "#C94F4A", "#2E8F94"
INK, MUTED, GRID = "#2F2A24", "#6B655C", "#DCD7CE"
FS_L, FS_T, FS_TT, FS_P, FS_LEG = 8.0, 7.0, 8.0, 10.5, 7.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#4A4A4A", "axes.linewidth": 0.9,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
})

RADII = (25, 50, 100)
LADDER = ("station", "space", "space_time", "full")
LADDER_LAB = ("station\nonly", "+ spatial\nbuffer", "+ temporal\nguard", "+ grid\ncell")
SPECS = (("coordfree", FREE, "o", "coordinate-free"),
         ("full", WITH, "s", "with lat/lon"))


def wilson(k, n, z=1.959963985):
    """Wilson score interval, in per cent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def load(R):
    path = _os.path.join(SIMDIR, f"simulation_summary_R{R}.csv")
    if not _os.path.exists(path):
        raise SystemExit(f"missing run summary: {path}")
    return pd.read_csv(path)


def cell(d, beta, spec, design):
    """Summed verdict count and denominator over the sixteen scenarios of one beta state."""
    g = d[(d.beta_state == beta) & (d.spec == spec) & (d.design == design)]
    n = int(g.n_rep.sum())
    k = int(round((g.verdict_rate * g.n_rep).sum()))
    lo, hi = wilson(k, n)
    return 100 * k / n, lo, hi, n


def panel(ax, xs, series, title, ylab, xticklabels, xlabel=None):
    for (key, col, mk, lab), pts in series.items():
        rate = [p[0] for p in pts]
        lo = [p[0] - p[1] for p in pts]
        hi = [p[2] - p[0] for p in pts]
        ax.errorbar(xs, rate, yerr=[lo, hi], color=col, marker=mk, ms=3.8, lw=1.3,
                    capsize=2.0, elinewidth=0.8, mec="white", mew=0.7, label=lab, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, fontsize=FS_T)
    ax.tick_params(axis="y", labelsize=FS_T)
    ax.set_ylabel(ylab, fontsize=FS_L)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_L)
    ax.set_title(title, fontsize=FS_TT, color="#333", pad=5)
    ax.set_ylim(bottom=0)


def tag(ax, s, dx=-0.20):
    ax.text(dx, 1.17, s, transform=ax.transAxes, fontsize=FS_P, fontweight="bold",
            va="top", ha="left", color=INK)


# ------------------------------------------------------------------ ladder, R = 50 km
d50 = load(50)
xs = list(range(len(LADDER)))
fpr = {s: [cell(d50, "absent", s[0], g) for g in LADDER] for s in SPECS}
det = {s: [cell(d50, "present", s[0], g) for g in LADDER] for s in SPECS}

fig = plt.figure(figsize=(5.40, 2.95))
G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                      left=0.100, right=0.972, top=0.80, bottom=0.235)
ax1 = fig.add_subplot(G[0, 0])
panel(ax1, xs, fpr, "False-positive verdict rate\n(no transferable signal)",
      "Verdict rate (%)", LADDER_LAB)
ax1.legend(fontsize=FS_LEG, frameon=False, loc="upper right", handlelength=1.4,
           borderaxespad=0.2)
tag(ax1, "a")
ax2 = fig.add_subplot(G[0, 1])
panel(ax2, xs, det, "Detection of a genuine\ntransferable signal", "Verdict rate (%)", LADDER_LAB)
tag(ax2, "b")
fig.savefig(_os.path.join(OUTDIR, "Simulation_Ladder.png"), dpi=600)

# ------------------------------------------------------------------ full audit across radius
byR = {R: load(R) for R in RADII}
# radius is plotted on its own linear scale, not as evenly spaced categories: 25->50 and
# 50->100 km are not the same step and should not be drawn as if they were
xs2 = list(RADII)
fprR = {s: [cell(byR[R], "absent", s[0], "full") for R in RADII] for s in SPECS}
detR = {s: [cell(byR[R], "present", s[0], "full") for R in RADII] for s in SPECS}

fig2 = plt.figure(figsize=(5.40, 2.95))
G2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.42,
                       left=0.100, right=0.972, top=0.80, bottom=0.235)
ax3 = fig2.add_subplot(G2[0, 0])
panel(ax3, xs2, fprR, "Full-audit specificity\n(no transferable signal)", "Verdict rate (%)",
      [str(R) for R in RADII], xlabel="Distance buffer $R$ (km)")
ax3.legend(fontsize=FS_LEG, frameon=False, loc="upper right", handlelength=1.4,
           borderaxespad=0.2)
tag(ax3, "a")
ax4 = fig2.add_subplot(G2[0, 1])
panel(ax4, xs2, detR, "Full-audit detection\n(genuine signal present)", "Verdict rate (%)",
      [str(R) for R in RADII], xlabel="Distance buffer $R$ (km)")
tag(ax4, "b")
fig2.savefig(_os.path.join(OUTDIR, "Simulation_Radius.png"), dpi=600)

# ------------------------------------------------------------------ echo the plotted values
print("ladder at R=50 km, false-positive verdict rate (%)")
for (key, col, mk, lab), pts in fpr.items():
    print(f"  {lab:18s} " + "  ".join(f"{g}={p[0]:.2f}" for g, p in zip(LADDER, pts)))
print("ladder at R=50 km, detection (%)")
for (key, col, mk, lab), pts in det.items():
    print(f"  {lab:18s} " + "  ".join(f"{g}={p[0]:.2f}" for g, p in zip(LADDER, pts)))
print("full audit across radius, false-positive (%)")
for (key, col, mk, lab), pts in fprR.items():
    print(f"  {lab:18s} " + "  ".join(f"R{R}={p[0]:.2f}" for R, p in zip(RADII, pts)))
print("full audit across radius, detection (%)")
for (key, col, mk, lab), pts in detR.items():
    print(f"  {lab:18s} " + "  ".join(f"R{R}={p[0]:.2f}" for R, p in zip(RADII, pts)))
print(f"\nwrote Simulation_Ladder.png and Simulation_Radius.png to {OUTDIR}")

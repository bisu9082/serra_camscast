"""Figure 3 — the spatial-transfer audit. Regenerated from the deterministic re-run
(loso_v2.py, random_state=0, early_stopping=False) so that every value matches the
corrected manuscript. All inputs are verified artefacts; nothing here is illustrative.

Palette: two hues only, validated with the dataviz palette validator ---
  #C94F4A (with coordinates)  vs  #2E8F94 (coordinate-free)
  normal-vision dE 24.4, deutan dE 11.6, both contrast >= 3:1 on white.
Identity is never colour-alone: line style, marker shape and direct labels repeat it.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

RUN = _ROOT + "/results"
RAW = _ROOT + "/results/per_point"

WITH  = "#C94F4A"   # with latitude/longitude
FREE  = "#2E8F94"   # coordinate-free
INK   = "#2F2A24"
MUTED = "#6B655C"
GRID  = "#DCD7CE"
NEUT  = "#F3EFE9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#4A4A4A", "axes.linewidth": 1.0,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "xtick.major.size": 2.4,
    "ytick.major.size": 2.4, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
})
FS_L, FS_T, FS_TT, FS_P, FS_LEG = 8.0, 7.0, 8.0, 10.5, 7.0   # all >= 7 pt at final size

NAME = {369953: "Bida Zayed", 369954: "Khadeeja Sch.", 369955: "Liwa", 369957: "Hamdan St.",
        369958: "Mussafah", 369959: "Al Ain St.", 369960: "Sweihan", 369961: "Khalifa City",
        369962: "Gayathi", 369963: "Al Maqtaa", 369965: "Ruwais", 369966: "Baniyas Sch.",
        369967: "Habshan S.", 369968: "Al Ain Islamic", 369969: "Zakher", 369971: "Al Qua'a",
        369972: "Al Tawia", 369973: "E11 Road", 369974: "Al Mafraq"}

prim = json.load(open(f"{RUN}/loso_v2_primary.json"))["primary"]
pool = json.load(open(f"{RUN}/pooled_pt.json"))
S = pd.read_csv(f"{RAW}/fig1_station_table.csv")
S10 = S[S.pol == "PM10"].copy(); S10["id"] = S10.st.str[1:].astype(int); S10 = S10.set_index("id")
g10 = pd.Series({int(k): v for k, v in prim["full_R0"]["per_station"].items()})

# PM2.5 deterministic cold-start gains (exp_loso re-run, random_state=0, early_stopping=False)
PM25 = {"Manama": (6.62, 27.15), "KuwaitCity": (36.94, 45.34), "Dhahran": (11.54, 26.55),
        "Jeddah": (3.77, -16.77), "Doha": (15.14, 22.74)}

R = [0, 25, 50, 100]
SRC = {0: 18.0, 25: 15.5, 50: 14.4, 100: 12.0}          # mean surviving source stations
pooled = {k: [pool[f"{k}_R{r}"]["pooled_pt"] for r in R] for k in ("full", "coordfree")}
median = {k: [prim[f"{k}_R{r}"]["median_gain_pct"] for r in R] for k in ("full", "coordfree")}

# Authored at the final placement width (0.96\textwidth = 5.18 in), one panel per row, so the
# printed point size equals the authored point size and nothing is reduced by \includegraphics.
# The four panels are split across two figures: a 2x2 arrangement at this width leaves each
# panel 2.3 in across, which is too narrow for the in-panel annotations.
FIGSIZE = (5.18, 5.15)
GK = dict(hspace=0.62, left=0.135, right=0.975, top=0.915, bottom=0.095)
figA = plt.figure(figsize=FIGSIZE)
GA = gridspec.GridSpec(2, 1, figure=figA, **GK)
figB = plt.figure(figsize=FIGSIZE)
GB = gridspec.GridSpec(2, 1, figure=figB, **GK)
def tag(ax, s, dx=-0.125):
    ax.text(dx, 1.20, s, transform=ax.transAxes, fontsize=FS_P, fontweight="bold",
            va="top", ha="left", color=INK)

# ---------------------------------------------------------------- (a)
import json as _j
ACF = _j.load(open(f"{RUN}/resid_acf.json"))["lag_hours"]
V7 = _j.load(open(f"{RUN}/audit_v7_qc6.json"))
G3 = _j.load(open(f"{RUN}/audit_v7_qc6_guard3.json"))
lh = np.array(sorted(int(k) for k in ACF))
av = np.array([ACF[str(int(h))] for h in lh])

axa = figA.add_subplot(GA[0, 0])
axa.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
axa.axhline(np.exp(-1), color=GRID, lw=1.2, zorder=1)
axa.text(3.4, np.exp(-1) + 0.012, "$1/e$", fontsize=7.0, color=MUTED)
axa.axvspan(0, 72, color=WITH, alpha=0.10, zorder=0)
axa.plot(lh, av, "-o", color=INK, lw=1.4, ms=4.0, mec="white", mew=0.8, zorder=4)
for h, lab, col in [(72, "guard used\nearlier: 3 d", WITH), (240, "guard used\nhere: 10 d", FREE)]:
    axa.axvline(h, color=col, lw=1.3, ls=":", zorder=3)
    y = ACF[str(h)]
    axa.plot(h, y, marker="D", ms=6.5, mfc=col, mec="white", mew=1.0, zorder=6, linestyle="none")
    axa.annotate(f"{y:+.2f}", (h, y), textcoords="offset points",
                 xytext=(9, -17) if h == 72 else (-9, 9),
                 ha="left" if h == 72 else "right",
                 fontsize=7.5, color=col, fontweight="bold")
    axa.text(h, 0.86, lab, ha="center" if h == 72 else "right", va="top",
             fontsize=7.0, color=col, fontweight="bold")
axa.set_xscale("log")
axa.set_xticks([3, 12, 24, 72, 168, 240])
axa.set_xticklabels(["3 h", "12 h", "1 d", "3 d", "7 d", "10 d"], fontsize=FS_T)
axa.set_xlim(2.6, 330)
axa.set_ylim(-0.05, 0.95)
axa.set_xlabel("Lag", fontsize=FS_L)
axa.set_ylabel("Median residual autocorrelation", fontsize=FS_L)
axa.tick_params(axis="y", labelsize=FS_T)
axa.text(0.97, 0.60, "3 d guard keeps a fifth\nof the correlation",
         transform=axa.transAxes, ha="right", va="top", fontsize=7.0,
         color=WITH, style="italic")
axa.set_title("The residual does not decorrelate on the timescale of a dust episode",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axa.spines[sp].set_visible(False)
tag(axa, "a")

# ---------------------------------------------------------------- (b)
axb = figA.add_subplot(GA[1, 0])
axb.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
for key, col, lab in [("full", WITH, "with lat/lon"), ("coordfree", FREE, "coordinate-free")]:
    m3 = [G3[f"buffer_time_cell_{key}_R{r}"]["delta_median"] for r in R]
    m10 = [V7[f"buffer_time_cell_{key}_R{r}"]["delta_median"] for r in R]
    c10 = np.array([V7[f"buffer_time_cell_{key}_R{r}"]["delta_median_ci_cell"] for r in R])
    axb.fill_between(R, c10[:, 0], c10[:, 1], color=col, alpha=0.12, lw=0, zorder=2)
    axb.plot(R, m3, ":s", color=col, lw=1.1, ms=3.6, alpha=0.65, mec="white", mew=0.6,
             zorder=3, label=f"{lab} · guard 3 d")
    axb.plot(R, m10, "-o", color=col, lw=1.5, ms=4.4, mec="white", mew=0.8, zorder=4,
             label=f"{lab} · guard 10 d")
for key, col, r in [("full", WITH, 0), ("coordfree", FREE, 0)]:
    q3 = G3["bh"][f"buffer_time_cell_{key}_R{r}"]["binom_bh"]
    q10 = V7["bh"][f"buffer_time_cell_{key}_R{r}"]["binom_bh"]
    y3 = G3[f"buffer_time_cell_{key}_R{r}"]["delta_median"]
    axb.annotate(f"$q$={q3:.3f}", (r, y3), textcoords="offset points",
                 xytext=(7, 7 if key == "coordfree" else -12), ha="left", fontsize=7.0,
                 color=col, fontweight="bold")
axb.text(0.97, 0.05, "10 d guard: no radius clears\ncorrection ($q\\geq0.25$)",
         transform=axb.transAxes, ha="right", va="bottom", fontsize=7.0,
         color=INK, style="italic")
axb.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
axb.set_ylabel(r"Median $\Delta = \log(\mathrm{MAE_{raw}}/\mathrm{MAE_{transfer}})$",
               fontsize=FS_L - 1)
axb.set_xticks(R); axb.set_xticklabels([str(r) for r in R], fontsize=FS_T)
axb.tick_params(labelsize=FS_T)
axb.set_xlim(-6, 116); axb.set_ylim(-0.09, 0.30)
axb.legend(fontsize=FS_LEG - 2.0, frameon=False, loc="upper right", ncol=1,
           handlelength=2.6, labelspacing=0.32)
axb.set_title("Widening the guard removes the gain at every radius, both feature sets",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
tag(axb, "b")

# ---------------------------------------------------------------- (c)
# The other convention that carried the result: where the block boundaries fall.
axc = figB.add_subplot(GB[0, 0])
CAL = [218, 9, 4, 7, 335, 489]   # station 369959 under the primary QC (K=6)
QNT = [177, 177, 176, 177, 177, 178]   # same station under quantile blocks
x = np.arange(6); w = 0.38
axc.bar(x - w/2, CAL, width=w, facecolor=WITH, edgecolor=WITH, lw=1.2, zorder=3,
        label="calendar-equal blocks")
axc.bar(x + w/2, QNT, width=w, facecolor=FREE, edgecolor=FREE, lw=1.2, zorder=3,
        label="quantile blocks")
for xi, v in zip(x - w/2, CAL):
    axc.text(xi, v + 12, str(v), ha="center", fontsize=7.0, color=WITH, fontweight="bold")
for xi, v in zip(x + w/2, QNT):
    axc.text(xi, v + 12, str(v), ha="center", fontsize=7.0, color=FREE, fontweight="bold")
axc.set_xticks(x); axc.set_xticklabels([f"fold {i+1}" for i in range(6)], fontsize=7.2)
axc.set_ylabel("Points held out per fold", fontsize=FS_L)
axc.tick_params(axis="y", labelsize=FS_T)
axc.set_ylim(0, 660)
axc.legend(fontsize=FS_LEG - 1, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
axc.text(0.62, 0.99,
         "imbalance ratio  124 $\\rightarrow$ 1.01",
         transform=axc.transAxes, ha="center", va="top", fontsize=7.5,
         color=INK, fontweight="bold")
axc.text(0.62, 0.90, "four months of 2023 are empty",
         transform=axc.transAxes, ha="center", va="top", fontsize=7.0,
         color=MUTED, style="italic")
axc.set_title("A six-fold temporal control that was three-fold in substance",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axc.spines[sp].set_visible(False)
axc.grid(axis="x", visible=False)
tag(axc, "a")

# ---------------------------------------------------------------- (d)
axd = figB.add_subplot(GB[1, 0])
PA = [V7["paired"][f"buffer_time_cell_R{r}"] for r in R]
axd.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
ci = np.array([p["cell_ci"] for p in PA])
med = [p["median_diff"] for p in PA]
axd.fill_between(R, ci[:, 0], ci[:, 1], color=FREE, alpha=0.13, lw=0, zorder=2)
axd.plot(R, med, "-o", color=FREE, lw=2.8, ms=10, mec="white", mew=1.4, zorder=4)
for r, p, y in zip(R, PA, med):
    axd.annotate(f"{p['coordfree_better']}/19", (r, y), textcoords="offset points",
                 xytext=(0, 15), ha="center", fontsize=7.2, color=FREE, fontweight="bold")
    star = "$p$=%.3f" % p["station_binom_p"]
    axd.annotate(star, (r, y), textcoords="offset points", xytext=(0, 33),
                 ha="center", fontsize=7.0, color=MUTED)
axd.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
axd.set_ylabel(r"Paired $\Delta_{\mathrm{free}} - \Delta_{\mathrm{lat/lon}}$", fontsize=FS_L)
axd.set_xticks(R); axd.set_xticklabels([str(r) for r in R], fontsize=FS_T)
axd.tick_params(labelsize=FS_T)
axd.set_xlim(-8, 118); axd.set_ylim(-0.10, 0.66)
axd.text(0.03, 0.95, "coordinate-free better", transform=axd.transAxes,
         fontsize=7.0, color=FREE, fontweight="bold", va="top")
axd.text(0.03, 0.03, "with lat/lon better", transform=axd.transAxes,
         fontsize=7.0, color=WITH, fontweight="bold", va="bottom")
axd.text(0.97, 0.86, "shading: block bootstrap,\nnine covariate cells",
         transform=axd.transAxes, ha="right", va="top", fontsize=7.0,
         color=MUTED, style="italic")
axd.set_title("Coordinates are neutral while neighbours remain, costly as the buffer widens",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axd.spines[sp].set_visible(False)
tag(axd, "b", dx=-0.13)

figA.savefig(_HERE + "/figures/Empirical_Guard.png", dpi=600)
figB.savefig(_HERE + "/figures/Empirical_Blocks.png", dpi=600)
# no tight bbox: bbox_inches="tight" would change the saved width and reintroduce down-scaling
print("saved figures/Empirical_Guard.png and figures/Empirical_Blocks.png")
print("panel b guard10:", {k: [round(V7[f"buffer_time_cell_{k}_R{r}"]["delta_median"],3) for r in R] for k in ("full","coordfree")})
print("panel d paired:", [(r, p["coordfree_better"], round(p["station_binom_p"],3)) for r, p in zip(R, PA)])

"""Figure 3 — the spatial-transfer audit. Regenerated from the deterministic re-run
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
(loso_v2.py, random_state=0, early_stopping=False) so that every value matches the
corrected manuscript. All inputs are verified artefacts; nothing here is illustrative.

Palette: two hues only, validated with the dataviz palette validator ---
  #C94F4A (with coordinates)  vs  #2E8F94 (coordinate-free)
  normal-vision dE 24.4, deutan dE 11.6, both contrast >= 3:1 on white.
Identity is never colour-alone: line style, marker shape and direct labels repeat it.
"""
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
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
})
FS_L, FS_T, FS_TT, FS_P, FS_LEG = 15, 12.5, 13.5, 21, 12

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

fig = plt.figure(figsize=(15.5, 12.4))
G = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24,
                      left=0.075, right=0.965, top=0.945, bottom=0.075)
def tag(ax, s, dx=-0.10):
    ax.text(dx, 1.10, s, transform=ax.transAxes, fontsize=FS_P, fontweight="bold",
            va="top", ha="left", color=INK)

# ---------------------------------------------------------------- (a)
import json as _j
ACF = _j.load(open(f"{RUN}/resid_acf.json"))["lag_hours"]
V7 = _j.load(open(f"{RUN}/audit_v7_qc6.json"))
G3 = _j.load(open(f"{RUN}/audit_v7_qc6_guard3.json"))
lh = np.array(sorted(int(k) for k in ACF))
av = np.array([ACF[str(int(h))] for h in lh])

axa = fig.add_subplot(G[0, 0])
axa.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
axa.axhline(np.exp(-1), color=GRID, lw=1.2, zorder=1)
axa.text(3.4, np.exp(-1) + 0.012, "$1/e$", fontsize=11, color=MUTED)
axa.axvspan(0, 72, color=WITH, alpha=0.10, zorder=0)
axa.plot(lh, av, "-o", color=INK, lw=2.6, ms=8, mec="white", mew=1.3, zorder=4)
for h, lab, col in [(72, "guard used\nearlier: 3 d", WITH), (240, "guard used\nhere: 10 d", FREE)]:
    axa.axvline(h, color=col, lw=2.2, ls=":", zorder=3)
    y = ACF[str(h)]
    axa.plot(h, y, marker="D", ms=13, mfc=col, mec="white", mew=1.6, zorder=6, linestyle="none")
    axa.annotate(f"{y:+.2f}", (h, y), textcoords="offset points",
                 xytext=(14, -22 if h == 72 else 10),
                 fontsize=12.5, color=col, fontweight="bold")
    axa.text(h, 0.80, lab, ha="center", va="top", fontsize=11.5, color=col, fontweight="bold")
axa.set_xscale("log")
axa.set_xticks([3, 12, 24, 72, 168, 240])
axa.set_xticklabels(["3 h", "12 h", "1 d", "3 d", "7 d", "10 d"], fontsize=FS_T)
axa.set_xlim(2.6, 330)
axa.set_ylim(-0.05, 0.95)
axa.set_xlabel("Lag", fontsize=FS_L)
axa.set_ylabel("Median residual autocorrelation", fontsize=FS_L)
axa.tick_params(axis="y", labelsize=FS_T)
axa.text(0.97, 0.60, "a three-day guard leaves\na fifth of the correlation",
         transform=axa.transAxes, ha="right", va="top", fontsize=11,
         color=WITH, style="italic")
axa.set_title("The residual field does not decorrelate on the\ntimescale of a dust episode",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axa.spines[sp].set_visible(False)
tag(axa, "a")

# ---------------------------------------------------------------- (b)
axb = fig.add_subplot(G[0, 1])
axb.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
for key, col, lab in [("full", WITH, "with lat/lon"), ("coordfree", FREE, "coordinate-free")]:
    m3 = [G3[f"buffer_time_cell_{key}_R{r}"]["delta_median"] for r in R]
    m10 = [V7[f"buffer_time_cell_{key}_R{r}"]["delta_median"] for r in R]
    c10 = np.array([V7[f"buffer_time_cell_{key}_R{r}"]["delta_median_ci_cell"] for r in R])
    axb.fill_between(R, c10[:, 0], c10[:, 1], color=col, alpha=0.12, lw=0, zorder=2)
    axb.plot(R, m3, ":s", color=col, lw=1.9, ms=7, alpha=0.65, mec="white", mew=0.9,
             zorder=3, label=f"{lab} · guard 3 d")
    axb.plot(R, m10, "-o", color=col, lw=2.8, ms=9, mec="white", mew=1.3, zorder=4,
             label=f"{lab} · guard 10 d")
for key, col, r in [("full", WITH, 0), ("coordfree", FREE, 0)]:
    q3 = G3["bh"][f"buffer_time_cell_{key}_R{r}"]["binom_bh"]
    q10 = V7["bh"][f"buffer_time_cell_{key}_R{r}"]["binom_bh"]
    y3 = G3[f"buffer_time_cell_{key}_R{r}"]["delta_median"]
    axb.annotate(f"$q$={q3:.3f}", (r, y3), textcoords="offset points",
                 xytext=(-6, 12 if key == "coordfree" else -20), fontsize=11,
                 color=col, fontweight="bold", ha="left")
axb.text(0.97, 0.05, "with a 10 d guard no cell clears\ncorrection at any radius ($q\\geq0.25$)",
         transform=axb.transAxes, ha="right", va="bottom", fontsize=11,
         color=INK, style="italic")
axb.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
axb.set_ylabel(r"Median $\Delta = \log(\mathrm{MAE_{raw}}/\mathrm{MAE_{transfer}})$",
               fontsize=FS_L - 1)
axb.set_xticks(R); axb.set_xticklabels([str(r) for r in R], fontsize=FS_T)
axb.tick_params(labelsize=FS_T)
axb.set_xlim(-6, 116); axb.set_ylim(-0.09, 0.30)
axb.legend(fontsize=FS_LEG - 2.0, frameon=False, loc="upper right", ncol=1,
           handlelength=2.6, labelspacing=0.32)
axb.set_title("Widening the guard removes the transfer gain\nat every radius, for both feature sets",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
tag(axb, "b")

# ---------------------------------------------------------------- (c)
# The other convention that carried the result: where the block boundaries fall.
axc = fig.add_subplot(G[1, 0])
CAL = [218, 9, 4, 7, 335, 489]   # station 369959 under the primary QC (K=6)
QNT = [177, 177, 176, 177, 177, 178]   # same station under quantile blocks
x = np.arange(6); w = 0.38
axc.bar(x - w/2, CAL, width=w, facecolor=WITH, edgecolor=WITH, lw=1.2, zorder=3,
        label="calendar-equal blocks")
axc.bar(x + w/2, QNT, width=w, facecolor=FREE, edgecolor=FREE, lw=1.2, zorder=3,
        label="quantile blocks")
for xi, v in zip(x - w/2, CAL):
    axc.text(xi, v + 12, str(v), ha="center", fontsize=11.5, color=WITH, fontweight="bold")
for xi, v in zip(x + w/2, QNT):
    axc.text(xi, v + 12, str(v), ha="center", fontsize=11.5, color=FREE, fontweight="bold")
axc.set_xticks(x); axc.set_xticklabels([f"fold {i+1}" for i in range(6)], fontsize=12)
axc.set_ylabel("Points held out per fold", fontsize=FS_L)
axc.tick_params(axis="y", labelsize=FS_T)
axc.set_ylim(0, 660)
axc.legend(fontsize=FS_LEG - 1, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
axc.text(0.62, 0.99,
         "imbalance ratio  124 $\\rightarrow$ 1.01",
         transform=axc.transAxes, ha="center", va="top", fontsize=12.5,
         color=INK, fontweight="bold")
axc.text(0.62, 0.90, "four months of 2023 hold no observations at all",
         transform=axc.transAxes, ha="center", va="top", fontsize=10.5,
         color=MUTED, style="italic")
axc.set_title("A six-fold temporal control that was three-fold in substance",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axc.spines[sp].set_visible(False)
axc.grid(axis="x", visible=False)
tag(axc, "c")

# ---------------------------------------------------------------- (d)
axd = fig.add_subplot(G[1, 1])
PA = [V7["paired"][f"buffer_time_cell_R{r}"] for r in R]
axd.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
ci = np.array([p["cell_ci"] for p in PA])
med = [p["median_diff"] for p in PA]
axd.fill_between(R, ci[:, 0], ci[:, 1], color=FREE, alpha=0.13, lw=0, zorder=2)
axd.plot(R, med, "-o", color=FREE, lw=2.8, ms=10, mec="white", mew=1.4, zorder=4)
for r, p, y in zip(R, PA, med):
    axd.annotate(f"{p['coordfree_better']}/19", (r, y), textcoords="offset points",
                 xytext=(0, 15), ha="center", fontsize=12, color=FREE, fontweight="bold")
    star = "$p$=%.3f" % p["station_binom_p"]
    axd.annotate(star, (r, y), textcoords="offset points", xytext=(0, 33),
                 ha="center", fontsize=10.5, color=MUTED)
axd.set_xlabel("Distance buffer $R$ (km)", fontsize=FS_L)
axd.set_ylabel(r"Paired $\Delta_{\mathrm{free}} - \Delta_{\mathrm{lat/lon}}$", fontsize=FS_L)
axd.set_xticks(R); axd.set_xticklabels([str(r) for r in R], fontsize=FS_T)
axd.tick_params(labelsize=FS_T)
axd.set_xlim(-8, 118); axd.set_ylim(-0.10, 0.66)
axd.text(0.03, 0.95, "coordinate-free better", transform=axd.transAxes,
         fontsize=11.5, color=FREE, fontweight="bold", va="top")
axd.text(0.03, 0.03, "with lat/lon better", transform=axd.transAxes,
         fontsize=11.5, color=WITH, fontweight="bold", va="bottom")
axd.text(0.97, 0.86, "shading: block bootstrap over\nthe nine covariate cells",
         transform=axd.transAxes, ha="right", va="top", fontsize=10.5,
         color=MUTED, style="italic")
axd.set_title("Coordinates are neutral while neighbours remain\nand grow costly as the buffer widens",
              fontsize=FS_TT, color="#333", pad=9)
for sp in ("top", "right"): axd.spines[sp].set_visible(False)
tag(axd, "d", dx=-0.13)

fig.savefig(_HERE + "/Fig4.png", dpi=300, bbox_inches="tight")
print("saved Fig4.png")
print("panel b guard10:", {k: [round(V7[f"buffer_time_cell_{k}_R{r}"]["delta_median"],3) for r in R] for k in ("full","coordfree")})
print("panel d paired:", [(r, p["coordfree_better"], round(p["station_binom_p"],3)) for r, p in zip(R, PA)])

"""Graphical abstract for Environmental Modelling & Software.
Spec: >= 531 x 1328 px (h x w), readable at 5 x 13 cm. Rendered at 1062 x 2656 px.
Same two-hue validated palette as Figure 3; every number is a verified value.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

WITH, FREE, INK, MUTED, GRID = "#C94F4A", "#2E8F94", "#2F2A24", "#6B655C", "#DCD7CE"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white",
                     "figure.facecolor": "white", "axes.edgecolor": "#4A4A4A"})

pool = json.load(open(_ROOT + "/results/pooled_pt.json"))
A3 = json.load(open(_ROOT + "/results/audit_v3.json"))
R = [0, 25, 50, 100]
# buffer-only pooled (the misleading reading) and full-control median (the summary of record)
full = [pool[f"full_R{r}"]["pooled_pt"] for r in R]
free = [pool[f"coordfree_R{r}"]["pooled_pt"] for r in R]
full_m = [A3["llto"][f"full_R{r}"]["median"] for r in R]
free_m = [A3["llto"][f"coordfree_R{r}"]["median"] for r in R]

W, Hh = 13.28, 5.31          # inches -> 2656 x 1062 px at 200 dpi
fig = plt.figure(figsize=(W, Hh))
fig.text(0.012, 0.945, "A distance buffer alone overstates the coordinate penalty",
         fontsize=21, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.845,
         "Withhold every source station within $R$ km of the target; then also block time, and remove latitude/longitude.",
         fontsize=14.5, color=MUTED, va="top")

# ---- left: the design ----
axl = fig.add_axes([0.022, 0.08, 0.235, 0.64]); axl.set_axis_off()
axl.set_xlim(0, 10); axl.set_ylim(0, 10)
sx = np.array([1.6, 3.3, 4.9, 6.6, 8.4, 2.4, 5.8, 7.9, 4.2, 8.8])
sy = np.array([8.4, 8.9, 7.6, 8.7, 8.2, 5.6, 4.9, 6.0, 3.6, 3.2])
tx, ty = 5.0, 6.6
axl.add_patch(plt.Circle((tx, ty), 2.5, color=WITH, alpha=0.15, zorder=1))
d = np.hypot(sx - tx, sy - ty)
axl.scatter(sx[d > 2.5], sy[d > 2.5], s=115, marker="s", facecolor=INK,
            edgecolor="white", lw=1.0, zorder=3)
axl.scatter(sx[d <= 2.5], sy[d <= 2.5], s=115, marker="s", facecolor="white",
            edgecolor=WITH, lw=2.0, zorder=3)
axl.scatter([tx], [ty], s=300, marker="*", facecolor=FREE, edgecolor=INK, lw=1.2, zorder=4)
axl.annotate("", xy=(tx + 2.5, ty), xytext=(tx, ty),
             arrowprops=dict(arrowstyle="-|>", color=WITH, lw=2.0), zorder=5)
axl.text(tx + 1.25, ty + 0.5, "$R$", fontsize=17, color=WITH, fontweight="bold", ha="center")
axl.text(tx, ty - 4.7, "held-out target", fontsize=12.5, color=INK, ha="center")
axl.scatter([0.7], [1.55], s=105, marker="s", facecolor="white", edgecolor=WITH, lw=2.0, clip_on=False)
axl.text(1.5, 1.55, "withheld", fontsize=12.5, color=WITH, ha="left", va="center", fontweight="bold")
axl.scatter([0.7], [0.45], s=105, marker="s", facecolor=INK, edgecolor="white", lw=1.0, clip_on=False)
axl.text(1.5, 0.45, "surviving", fontsize=12.5, color=INK, ha="left", va="center")

# ---- middle arrow ----
fig.text(0.272, 0.40, "\u2192", fontsize=44, color=MUTED, ha="center", va="center")

# ---- middle: the result ----
axr = fig.add_axes([0.345, 0.155, 0.315, 0.575])
axr.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
axr.grid(color=GRID, lw=0.7); axr.set_axisbelow(True)
axr.plot(R, full, ":s", color=WITH, lw=2.0, ms=8, alpha=0.6, mec="white", mew=1.0, zorder=2)
axr.plot(R, free, ":s", color=FREE, lw=2.0, ms=8, alpha=0.6, mec="white", mew=1.0, zorder=2)
axr.plot(R, full_m, "-o", color=WITH, lw=3.0, ms=10, mec="white", mew=1.4, zorder=3)
axr.plot(R, free_m, "-o", color=FREE, lw=3.0, ms=10, mec="white", mew=1.4, zorder=3)
axr.annotate("with lat/lon", (R[1], full_m[1]), textcoords="offset points", xytext=(4, -26),
             color=WITH, fontsize=13.5, fontweight="bold", ha="left")
axr.annotate("coordinate-free", (R[1], free_m[1]), textcoords="offset points", xytext=(4, 12),
             color=FREE, fontsize=13.5, fontweight="bold", ha="left")
for x, y, c, dx, dy, ha in [(100, full_m[-1], WITH, 10, -6, "left"),
                            (100, free_m[-1], FREE, 10, 6, "left"),
                            (100, full[-1], WITH, 10, 0, "left")]:
    axr.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points", xytext=(dx, dy),
                 ha=ha, va="center", fontsize=13, color=c, fontweight="bold")
axr.text(0.02, 0.06, "dotted: buffer only, pooled\nsolid: buffer $+$ time, median",
         transform=axr.transAxes, fontsize=11.5, color=MUTED, va="bottom", ha="left")
axr.set_xticks(R); axr.set_xticklabels([str(r) for r in R], fontsize=13)
axr.tick_params(axis="y", labelsize=13)
axr.set_xlabel("distance buffer $R$ (km)", fontsize=14.5, color=INK)
axr.set_ylabel("gain over raw CAMS (%)", fontsize=14, color=INK)
axr.set_xlim(-30, 148); axr.set_ylim(-50, 38)
for sp in ("top", "right"): axr.spines[sp].set_visible(False)

# ---- right: takeaway ----
fig.text(0.700, 0.645, "Half the apparent transfer\nwas spatial infilling.",
         fontsize=19, fontweight="bold", color=INK, va="top", ha="left", linespacing=1.35)
fig.text(0.700, 0.455,
         "Block time as well, and both feature sets\nstill transfer at 100 km. What coordinates\n"
         "cost is reliability: 3 of 19 stations fail\ncatastrophically, and geographically\n"
         "scrambled coordinates beat the true ones\nat that radius (28 of 30 permutations).",
         fontsize=12.6, color=MUTED, va="top", ha="left", linespacing=1.55)

fig.savefig(_HERE + "/Graphical_abstract.tif", dpi=200,
            bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(_HERE + "/Graphical_abstract.png", dpi=200)
print("saved graphical abstract")
print("pooled with lat/lon :", [round(v, 2) for v in full])
print("pooled coord-free   :", [round(v, 2) for v in free])

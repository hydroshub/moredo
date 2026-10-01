"""
Fig. S7: the recurrent representation retrained on the 362 catchments larger than
1000 km^2 only (lstm_af_largebasins) against the main recurrent model trained on all 1186
catchments, on the shared 362 catchments (strictly held-out estimates of both).
  (a) map of catchment-mean F_add (2001-2020) of the large-catchment model (Fig. 5 colour scale)
  (b) catchment-mean F_add, large-catchment model against main model, by major Koppen-Geiger
      class, dashed 1:1 line, Pearson r (same style as Fig. S2a)
  (c) held-out per-catchment NSE of ET, Q, SWE and TWSA for both models (boxes, whiskers 1.5 IQR,
      no fliers, display floor -1)
Reads figures/largebasins_compare.npz (largebasins_compare.py). Light, no pickle.

Usage: python scripts/analysis/fig_si_largebasins.py
"""
import os
import sys

import numpy as np
import matplotlib.axes as mpl_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, EXP_COLORS, format_map
from basin_attributes import KG_MAJOR_NAMES, KG_MAJOR_COLORS

from project_paths import REPO
FIG = os.path.join(REPO, "results_lateral_experiments", "figures")
NPZ = os.path.join(FIG, "largebasins_compare.npz")
OUT = os.path.join(FIG, "fig_si_largebasins.png")

MODELS = [("main", "All-catchment model (trained on 1186 catchments)", EXP_COLORS["lstm_af"]),
          ("large", "Large-catchment model (trained on 362 catchments > 1000 km$^2$)", "#abd9e9")]
VARS = ["et", "q", "swe", "twsa"]
VTITLE = {"et": "ET", "q": "Q", "swe": "SWE", "twsa": "TWSA"}
FLOOR = -1.0


def _style(ax):
    ax.format(grid=False, xlabelsize=LABELSIZE - 3, ylabelsize=LABELSIZE - 3,
              xticklabelsize=TICKLABELSIZE - 3, yticklabelsize=TICKLABELSIZE - 3)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.0)


def main():
    d = np.load(NPZ, allow_pickle=True)
    lon, lat, kg = d["lon"], d["lat"], d["kg"]
    fa, fb = d["fadd_main"].astype(float), d["fadd_large"].astype(float)

    fig = uplt.figure(figwidth="190mm", figheight="165mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=2, ncols=2, hratios=(1.05, 1), wspace=6, hspace=3)

    # --- a: map of the large-catchment model ---
    ax = fig.add_subplot(gs[0, :], proj=ccrs.PlateCarree(), abc=True, abcloc="l")
    sc = ax.scatter(lon, lat, c=fb, s=13, cmap="Blues", vmin=0, vmax=1, edgecolor="grey",
                    linewidth=0.2, transform=ccrs.PlateCarree(), zorder=2)
    format_map(ax, title="")
    ax.colorbar(sc, loc="r", width=0.14, length=0.8, extend="max",
                label="Catchment-mean $F_{add}$ (mm day$^{-1}$)", labelsize=LABELSIZE - 4,
                ticklabelsize=TICKLABELSIZE - 4)

    # --- b: large-catchment model against main model ---
    ax = fig.add_subplot(gs[1, 0], abc=True, abcloc="l")
    present = [c for c in sorted(KG_MAJOR_NAMES) if np.any(kg == c)]
    for c in present:
        m = kg == c
        ax.scatter(fa[m], fb[m], s=16, color=KG_MAJOR_COLORS[c], edgecolor="k", linewidth=0.2,
                   alpha=0.85, zorder=3)
    hi = float(np.nanpercentile(np.concatenate([fa, fb]), 99.5)) * 1.05
    ax.plot([0, hi], [0, hi], color="k", ls="--", lw=1.0, zorder=2)
    ax.text(0.96, 0.04, f"r = {np.corrcoef(fa, fb)[0, 1]:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=TICKLABELSIZE - 3)
    ax.format(xlim=(0, hi), ylim=(0, hi),
              xlabel="$F_{add}$ (mm day$^{-1}$), all-catchment model",
              ylabel="$F_{add}$ (mm day$^{-1}$), large-catchment model")
    ax.set_box_aspect(1)
    _style(ax)
    handles = [Line2D([], [], marker="o", linestyle="None", markersize=6,
                      markerfacecolor=KG_MAJOR_COLORS[c], markeredgecolor="k", markeredgewidth=0.2,
                      label=KG_MAJOR_NAMES[c]) for c in present]
    ax.legend(handles, loc="b", ncols=4, frame=False, prop={"size": LEGENDSIZE - 4})

    # --- c: held-out per-catchment NSE ---
    ax = fig.add_subplot(gs[1, 1], abc=True, abcloc="l")
    w = 0.36
    for k, (key, _, col) in enumerate(MODELS):
        data = []
        for v in VARS:
            x = d[f"nse_{v}_{key}"].astype(float)
            data.append(np.clip(x[np.isfinite(x)], FLOOR, None))
        pos = np.arange(len(VARS)) + (k - 0.5) * w
        bp = mpl_axes.Axes.boxplot(ax, data, positions=pos, widths=w * 0.85, showfliers=False,
                                   patch_artist=True, medianprops=dict(color="k", linewidth=1.1),
                                   whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7))
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_edgecolor("k"); patch.set_linewidth(0.6)
    ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
    ax.format(xlocator=list(range(len(VARS))), xformatter=[VTITLE[v] for v in VARS],
              xlim=(-0.55, len(VARS) - 0.45), ylim=(FLOOR, 1.05),
              ylabel="Held-out NSE per catchment")
    _style(ax)
    ax.legend([Patch(facecolor=col, edgecolor="k", linewidth=0.6) for _, _, col in MODELS],
              [lab for _, lab, _ in MODELS], loc="b", ncols=1, frame=False,
              prop={"size": LEGENDSIZE - 4})

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT}")
    print(f"  n = {len(fa)}  r = {np.corrcoef(fa, fb)[0, 1]:.3f}  median F_add all {np.median(fa):.3f}  "
          f"large {np.median(fb):.3f} mm/day")
    for v in VARS:
        a, b = d[f"nse_{v}_main"].astype(float), d[f"nse_{v}_large"].astype(float)
        k = np.isfinite(a) & np.isfinite(b)
        print(f"  {VTITLE[v]:4s} median NSE all {np.median(a[k]):+.2f}  large {np.median(b[k]):+.2f}  (n={k.sum()})")


if __name__ == "__main__":
    main()

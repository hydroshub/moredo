"""
Fig. S3: the recurrent representation trained with the cross-catchment pooled loss (Eq. 9) against
the catchment-normalized loss (NSE*), four panels (loss legend below panel a):
  (a) held-out per-catchment NSE of ET, Q, SWE and TWSA (boxes, whiskers 1.5 IQR, no fliers,
      display floor -1)
  (b) catchment-mean F_add (2001-2020), normalized loss against pooled loss, dashed 1:1 line
  (c) map of catchment-mean F_add, pooled loss
  (d) map of catchment-mean F_add, normalized loss
All values are strictly held-out estimates, so the two losses are compared like for like (panel c
therefore differs slightly from Fig. 5, which averages all 25 model instances). Fig. 5 colour scale.
Reads figures/loss_normalization_compare.npz (loss_normalization_compare.py). Light, no pickle.

Usage: python scripts/analysis/fig_si_loss_normalization.py
"""
import os
import sys

import numpy as np
import matplotlib.axes as mpl_axes
from matplotlib.patches import Patch
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, EXP_COLORS, format_map

from project_paths import REPO
FIG = os.path.join(REPO, "results_lateral_experiments", "figures")
NPZ = os.path.join(FIG, "loss_normalization_compare.npz")
OUT = os.path.join(FIG, "fig_si_loss_normalization.png")

RUNS = ["pooled", "normalized"]
LABEL = {"pooled": "Cross-catchment pooled loss", "normalized": "Catchment-normalized loss"}
COLOR = {"pooled": EXP_COLORS["lstm_af"], "normalized": "#e66101"}
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
    lon, lat = d["lon"], d["lat"]
    fp, fn = d["fadd_pooled"].astype(float), d["fadd_normalized"].astype(float)

    fig = uplt.figure(figwidth="190mm", figheight="138mm", tight=True,
                      spanx=False, spany=False, sharex=False, sharey=False)
    gs = uplt.GridSpec(nrows=2, ncols=2, hratios=(1, 0.74), wspace=5, hspace=3)

    # --- a: per-catchment NSE, grouped boxes ---
    ax_a = fig.add_subplot(gs[0, 0], abc=True, abcloc="l")
    w = 0.36
    for k, run in enumerate(RUNS):
        data = []
        for v in VARS:
            x = d[f"nse_{v}_{run}"].astype(float)
            data.append(np.clip(x[np.isfinite(x)], FLOOR, None))
        pos = np.arange(len(VARS)) + (k - 0.5) * w
        bp = mpl_axes.Axes.boxplot(ax_a, data, positions=pos, widths=w * 0.85, showfliers=False,
                                   patch_artist=True, medianprops=dict(color="k", linewidth=1.1),
                                   whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7))
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR[run]); patch.set_edgecolor("k"); patch.set_linewidth(0.6)
    ax_a.axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
    ax_a.format(xlocator=list(range(len(VARS))), xformatter=[VTITLE[v] for v in VARS],
                xlim=(-0.55, len(VARS) - 0.45), ylim=(FLOOR, 1.05),
                ylabel="Held-out NSE per catchment")
    _style(ax_a)

    # --- b: F_add normalized vs pooled ---
    ax = fig.add_subplot(gs[0, 1], abc=True, abcloc="l")
    ax.scatter(fp, fn, s=6, color="0.25", alpha=0.4, linewidth=0, zorder=2)
    lim = (0, 2.6)
    ax.plot(lim, lim, color="0.4", lw=0.9, ls="--", zorder=1)
    ax.text(0.05, 0.95, f"r = {np.corrcoef(fp, fn)[0, 1]:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=TICKLABELSIZE - 3)
    ax.text(2.55, 2.35, "1:1", color="0.4", fontsize=TICKLABELSIZE - 5, ha="right", va="top")
    ax.format(xlim=lim, ylim=lim,
              xlabel="$F_{add}$, pooled loss (mm day$^{-1}$)",
              ylabel="$F_{add}$, normalized loss (mm day$^{-1}$)")
    _style(ax)

    # --- c, d: maps ---
    sc = None
    for j, (run, f) in enumerate([("pooled", fp), ("normalized", fn)]):
        ax = fig.add_subplot(gs[1, j], proj=ccrs.PlateCarree(), abc=True, abcloc="l")
        sc = ax.scatter(lon, lat, c=f, s=7, cmap="Blues", vmin=0, vmax=1, edgecolor="grey",
                        linewidth=0.12, transform=ccrs.PlateCarree(), zorder=2)
        format_map(ax, title=LABEL[run])
        ax.format(titlesize=LABELSIZE - 4)
    fig.colorbar(sc, loc="b", length=0.45, width=0.12, extend="max",
                 label="Catchment-mean $F_{add}$ (mm day$^{-1}$)", labelsize=LABELSIZE - 4,
                 ticklabelsize=TICKLABELSIZE - 4)

    handles = [Patch(facecolor=COLOR[r], edgecolor="k", linewidth=0.6, label=LABEL[r]) for r in RUNS]
    ax_a.legend(handles, loc="b", ncols=1, frame=False, prop={"size": LEGENDSIZE - 3})
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT}")
    print(f"  median F_add pooled {np.median(fp):.2f}, normalized {np.median(fn):.2f} mm/day, "
          f"r = {np.corrcoef(fp, fn)[0, 1]:.3f}")


if __name__ == "__main__":
    main()

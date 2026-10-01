"""
Figure 1 in the exact inset layout of fig5_spatial_fadd_inset: the R_in/P non-closure world map
fills the figure as panel (a), the Budyko diagram (ET/P vs PET/P, coloured by R_in/P) sits as a
small inset in the lower-left as panel (b), and the R_in/P colorbar is a thin vertical inset in the
upper-left. Reads the cached non-closure data (fig_budyko_nonclosure_cache.npz); no pickle load.

Usage: python scripts/analysis/fig1_nonclosure_inset.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, format_map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from project_paths import RESULTS_DIR as RD
CACHE = os.path.join(RD, "figures", "fig_budyko_nonclosure_cache.npz")
OUT = os.path.join(RD, "figures_manuscript_final", "fig_1_nonclosure_inset.png")
FU_OMEGA = 2.6


def main():
    c = dict(np.load(CACHE, allow_pickle=True))
    v = c["valid"].astype(bool)
    rp = c["rin_over_p"]; lon = c["basin_lon"]; lat = c["basin_lat"]
    etp = c["et_over_p"]; petp = c["pet_over_p"]
    m = v & np.isfinite(rp) & np.isfinite(lon) & np.isfinite(lat)
    mb = m & np.isfinite(etp) & np.isfinite(petp)
    vmax = float(np.nanpercentile(np.abs(rp[m]), 95)); vmin = -vmax
    cmap = "RdBu_r"

    # --- fixed layout (matches fig5_spatial_fadd_inset) ---
    fig = uplt.figure(figwidth="250mm", figheight="112mm", tight=False,
                      spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())

    # --- a: world map of R_in/P ---
    ax.add_feature(cfeature.LAND, facecolor="#ebebeb", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    sc = ax.scatter(lon[m], lat[m], c=rp[m], s=14, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor="0.35", linewidth=0.2, transform=ccrs.PlateCarree(), zorder=2)
    format_map(ax, title="")
    ax.set_position([0.12, 0.11, 0.88, 0.88])
    fig.text(0.012, 0.975, "a", fontsize=LABELSIZE, fontweight="bold", va="top", ha="left", zorder=6)

    # --- colorbar: thin vertical, top of the left column (R_in/P, diverging) ---
    cax = fig.add_axes([0.10, 0.62, 0.013, 0.30])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(r"$R_{\mathrm{in}}/P$  (0 = closure)", size=LABELSIZE - 4)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 4)
    cb.outline.set_linewidth(0.6)
    # end labels: red (top) = missing input, blue (bottom) = surplus
    fig.text(0.113, 0.925, "missing\ninput", fontsize=TICKLABELSIZE - 4, color="#b2182b",
             va="bottom", ha="center", linespacing=0.9)
    fig.text(0.113, 0.608, "surplus", fontsize=TICKLABELSIZE - 4, color="#2166ac",
             va="top", ha="center")

    # --- b: Budyko diagram inset (lower-left, over the South Pacific) ---
    bx = fig.add_axes([0.075, 0.16, 0.24, 0.36])
    pr = np.linspace(0.0, 5.0, 500)
    with np.errstate(invalid="ignore"):
        fu = 1 + pr - (1 + pr ** FU_OMEGA) ** (1.0 / FU_OMEGA)
    bx.scatter(petp[mb], etp[mb], c=rp[mb], s=7, cmap=cmap, vmin=vmin, vmax=vmax,
               edgecolor="0.35", linewidth=0.15, alpha=0.9, zorder=1)
    fu_line, = bx.plot(pr, fu, color="k", lw=1.2, zorder=5,
                       label=fr"Budyko ($\omega$ = {FU_OMEGA})")
    lim_line, = bx.plot([0, 1], [0, 1], color="0.45", ls="--", lw=0.9, zorder=4,
                        label="water / energy limit")
    bx.plot([1, 5], [1, 1], color="0.45", ls="--", lw=0.9, zorder=4)
    bx.set_xlim(0, 5); bx.set_ylim(0, 1.5)
    bx.set_xticks([0, 1, 2, 3, 4, 5])
    bx.set_xlabel("PET / P", size=LABELSIZE - 4)
    bx.set_ylabel("ET / P", size=LABELSIZE - 4)
    bx.tick_params(axis="both", labelsize=TICKLABELSIZE - 4, length=2)
    for sp in bx.spines.values():
        sp.set_linewidth(0.8)
    bx.patch.set_facecolor("white"); bx.patch.set_alpha(0.9)
    from matplotlib.lines import Line2D
    pts = Line2D([], [], marker="o", ls="none", mfc="0.6", mec="0.35", mew=0.3, ms=4,
                 label="catchments")
    bx.legend(handles=[pts, fu_line, lim_line], loc="lower right", ncols=1, frameon=True,
              framealpha=0.9, edgecolor="none", prop={"size": LEGENDSIZE - 5}, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.3)
    fig.text(0.012, 0.50, "b", fontsize=LABELSIZE, fontweight="bold", va="bottom", ha="left", zorder=6)

    print(f"n_map={int(m.sum())}  n_budyko={int(mb.sum())}  |R_in/P| p95={vmax:.2f}")
    fig.savefig(OUT, dpi=200)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

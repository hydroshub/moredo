"""
Fig. S17: compound world map of the inferred subsidy expressed as relative fluxes:
  (a) F_add / P     (b) F_add / PET     (c) F_add / ET

These normalized measures make the catchment-mean subsidy comparable across climate zones, where
absolute F_add is hard to compare between wet and dry regions. All three ratios are dimensionless
and non-negative, and they share one sequential colour scale so the panels can be read against each
other. Reads the cached non-closure data (fig_budyko_nonclosure_cache.npz, written by
nonclosure_cache.py): 2001-2020 means of F_add, ERA5-Land P and observed ET, HydroATLAS PET, over
the catchments with a valid R_in (n = 996). No model inference.

Usage: python scripts/analysis/fig_fadd_relative_maps.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, format_map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from project_paths import RESULTS_DIR as RD
CACHE = os.path.join(RD, "figures", "fig_budyko_nonclosure_cache.npz")
OUT = os.path.join(RD, "figures", "fig_fadd_relative_maps.png")
CMAP = "batlowK"


def main():
    c = dict(np.load(CACHE, allow_pickle=True))
    v = c["valid"].astype(bool)
    lon = c["basin_lon"]; lat = c["basin_lat"]
    fadd = c["fadd"].astype(float)
    p = c["p_mean"].astype(float)                 # mm day^-1
    et = c["et_mean"].astype(float)               # mm day^-1
    pet = c["pet_mm_yr"].astype(float) / 365.25   # mm yr^-1 -> mm day^-1

    ratios = [fadd / p, fadd / pet, fadd / et]
    labels = [r"$F_{add}\,/\,P$", r"$F_{add}\,/\,\mathrm{PET}$", r"$F_{add}\,/\,\mathrm{ET}$"]

    base = v & np.isfinite(lon) & np.isfinite(lat)
    masks = [base & np.isfinite(r) & (np.asarray(p if i == 0 else pet if i == 1 else et) > 0)
             for i, r in enumerate(ratios)]
    # shared 0..p95 colour scale across all three ratios for cross-panel comparability
    allvals = np.concatenate([ratios[i][masks[i]] for i in range(3)])
    vmax = float(np.nanpercentile(allvals, 95))

    fig, axs = uplt.subplots(nrows=3, ncols=1, proj="pcarree", refwidth="150mm",
                             refaspect=2.6, hspace=1.0, share=False)
    sc = None
    for ax, r, m, lab, letter in zip(axs, ratios, masks, labels, "abc"):
        ax.add_feature(cfeature.LAND, facecolor="#ebebeb", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        sc = ax.scatter(lon[m], lat[m], c=r[m], s=9, cmap=CMAP, vmin=0, vmax=vmax,
                        edgecolor="0.35", linewidth=0.15, transform=ccrs.PlateCarree(), zorder=2)
        format_map(ax, title=lab)

    axs.format(abc="a", abcloc="ul", abcsize=LABELSIZE, abcweight="bold")
    cb = fig.colorbar(sc, loc="r", length=0.6, width=0.22, extend="max")
    cb.set_label(r"$F_{add}$ relative to flux", size=LABELSIZE - 2)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 1)

    print(f"n = {int(masks[0].sum())}  shared vmax (p95) = {vmax:.3f}")
    for lab, r, m in zip(labels, ratios, masks):
        print(f"  {lab:20s} median={np.nanmedian(r[m]):.3f}  p95={np.nanpercentile(r[m],95):.3f}")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

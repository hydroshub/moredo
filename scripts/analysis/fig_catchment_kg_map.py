"""
Appendix figure: the 1186 study catchments over the Koppen-Geiger major-class background, with an
inset histogram of the number of catchments per class. The KG grid provides the coloured
background; catchments are overlaid as neutral points so they read against it.

Produces: fig_catchment_kg_map.png
"""
import os
import sys
import pickle

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, LABELSIZE, TICKLABELSIZE, LEGENDSIZE, format_map
import cartopy.crs as ccrs
from basin_attributes import build_basin_tables, load_kg_major_grid, KG_MAJOR

from project_paths import REPO
RD = os.path.join(REPO, "results_lateral_experiments")
FIGS = os.path.join(RD, "figures")


def main(with_hist=True, out_name="fig_catchment_kg_map.png", hist_frame=False, legend_size=LEGENDSIZE - 2):
    """Render the study-domain map: catchments over KG major-class shading, with an optional per-class count histogram inset and a class legend."""
    OUT = os.path.join(FIGS, out_name)
    lstm = pickle.load(open(os.path.join(RD, "lstm_af", "averaged_full_output.pickle"), "rb"))
    dfc, _ = build_basin_tables(list(lstm["valid_ids"]))
    lon, lat = dfc["basin_lon"].values, dfc["basin_lat"].values
    n = np.isfinite(lon).sum()

    glon, glat, grid = load_kg_major_grid(downsample=4)

    ids = sorted(KG_MAJOR.keys())
    cmap = ListedColormap([KG_MAJOR[i]["color"] for i in ids])
    norm = BoundaryNorm([i - 0.5 for i in ids] + [ids[-1] + 0.5], cmap.N)

    fig = uplt.figure(figwidth="180mm", tight=True, spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())

    ax.pcolormesh(glon, glat, grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                  shading="auto", alpha=0.65, zorder=0, rasterized=True)
    ax.scatter(lon, lat, s=13, color="#222222", edgecolor="white", linewidth=0.35,
               transform=ccrs.PlateCarree(), zorder=3)
    format_map(ax, title="")

    # --- inset histogram (bottom-left): catchment count per KG major class ---
    if with_hist:
        kg = dfc["kg_major_id"].values
        hist_ids = [i for i in ids if KG_MAJOR[i]["code"] != "Dw"]  # drop Dw (no catchments)
        # order bars by catchment count, largest on the left
        hist_ids = sorted(hist_ids, key=lambda i: np.sum(kg == i), reverse=True)
        counts = [int(np.sum(kg == i)) for i in hist_ids]
        codes = [KG_MAJOR[i]["code"] for i in hist_ids]
        colors = [KG_MAJOR[i]["color"] for i in hist_ids]
        axin = fig.add_axes([0.045, 0.22, 0.24, 0.26])  # figure coords: bottom-left over open ocean
        xpos = np.arange(len(hist_ids))
        axin.bar(xpos, counts, color=colors, edgecolor="k", lw=0.4, zorder=3)
        off = max(counts) * 0.05
        for x, c in zip(xpos, counts):
            if c > 0:
                axin.text(x, c + off, str(c), ha="center", va="bottom", fontsize=8.5, rotation=90, zorder=4)
        axin.set_xticks(xpos)
        axin.set_xticklabels(codes, fontsize=6.5)
        axin.set_ylim(0, max(counts) * 1.28)
        axin.set_yticks([])
        axin.tick_params(axis="x", length=0, pad=1)
        if hist_frame:
            for sp in axin.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.9)
        else:
            for sp in ("top", "right", "left"):
                axin.spines[sp].set_visible(False)
            axin.spines["bottom"].set_linewidth(0.9)
        axin.patch.set_facecolor("white")
        axin.patch.set_alpha(0.85)

    clim_handles = [Patch(facecolor=KG_MAJOR[i]["color"], edgecolor="none",
                          label=KG_MAJOR[i]["code"]) for i in ids]
    catch_handle = Line2D([], [], marker="o", linestyle="None", markersize=5,
                          markerfacecolor="#222222", markeredgecolor="white", markeredgewidth=0.35,
                          label="Catchments")
    n_handle = Line2D([], [], marker="None", linestyle="None", label=f"(n = {n})")
    # 5 columns, row-major: the catchments entry occupies column 1 across both rows, with its two
    # text lines aligned to the two rows of KG classes (A/Cf, B/Ds, Cs/Dw, Cw/Df in columns 2-5).
    ordered = [catch_handle, clim_handles[0], clim_handles[1], clim_handles[2], clim_handles[3],
               n_handle, clim_handles[4], clim_handles[5], clim_handles[6], clim_handles[7]]
    fig.legend(ordered, [h.get_label() for h in ordered],
               loc="b", ncols=5, frame=False, prop={"size": legend_size})

    fig.save(OUT, dpi=250)
    print(f"Saved -> {OUT}  (n={n})")


if __name__ == "__main__":
    main()

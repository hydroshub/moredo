"""
Fig. 10: ET contribution of the recurrent subsidy, from the whole-period F_add=0 counterfactual
(ET_with - ET_no rerun over 2001-2020, ensemble-averaged over the 25 recurrent checkpoints), in the
same compact full-figure-map + inset layout as Fig. 5 (fig5_spatial_fadd_inset): the map of the
catchment delta ET fills the figure, a thin vertical colorbar sits as an inset in the upper-left,
and the distribution of delta ET within each major Koppen-Geiger class sits as an inset panel b in
the lower-left (same classes, colours and box style as the Fig. 5 inset).

Reads fadd_wholeperiod_counterfactual.npz. No pickle, no GPU.

Usage: python scripts/analysis/fig_fadd_contribution_compare.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import uplt, sns, LABELSIZE, TICKLABELSIZE, format_map
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from basin_attributes import build_basin_tables, KG_MAJOR_NAMES, KG_MAJOR_COLORS

from project_paths import RESULTS_DIR as RD
OUT = os.path.join(RD, "figures", "fig_fadd_contribution_compare.png")


def main():
    w = dict(np.load(os.path.join(RD, "figures", "fadd_wholeperiod_counterfactual.npz"), allow_pickle=True))
    ids = [str(g) for g in w["gauge_id"]]
    whole = w["dET_annual"].astype(float)

    dc, _ = build_basin_tables(ids)
    dc = dc.set_index("gauge_id").reindex(ids)
    lon, lat = dc["basin_lon"].values, dc["basin_lat"].values
    cat = dc["kg_major_id"].values
    mp = np.isfinite(whole) & np.isfinite(lon) & np.isfinite(lat)

    # Fixed layout (no tight engine): the map fills the figure, colorbar + boxplot are insets.
    fig = uplt.figure(figwidth="250mm", figheight="112mm", tight=False,
                      spanx=False, spany=False, sharex=False, sharey=False)
    ax = fig.add_subplot(proj=ccrs.PlateCarree())

    # --- a: whole-period delta ET map (no title) ---
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    sc = ax.scatter(lon[mp], lat[mp], c=whole[mp], s=15, cmap="YlOrBr", vmin=0, vmax=200,
                    edgecolor="0.35", linewidth=0.25, transform=ccrs.PlateCarree(), zorder=3)
    format_map(ax, title="")
    ax.set_position([0.12, 0.11, 0.88, 0.88])  # small left column; map fills the rest
    fig.text(0.012, 0.975, "a", fontsize=LABELSIZE, fontweight="bold", va="top", ha="left", zorder=6)

    # --- colorbar: thin vertical inset, upper-left ---
    cax = fig.add_axes([0.10, 0.65, 0.013, 0.28])
    cb = fig.colorbar(sc, cax=cax, extend="max")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("$\\Delta$ET (mm yr$^{-1}$)", size=LABELSIZE - 4)
    cb.ax.tick_params(labelsize=TICKLABELSIZE - 4)
    cb.outline.set_linewidth(0.6)

    # --- b: delta ET per major Koppen-Geiger class, lower-left inset (as the Fig. 5 inset) ---
    ordered_ids = [c for c in sorted(KG_MAJOR_NAMES.keys()) if np.any(cat[mp] == c)]
    order_labels = [KG_MAJOR_NAMES[c] for c in ordered_ids]
    palette = {KG_MAJOR_NAMES[c]: KG_MAJOR_COLORS[c] for c in ordered_ids}
    rows = [{"Cat": KG_MAJOR_NAMES[c], "det": v} for c in ordered_ids
            for v in whole[mp & (cat == c)]]
    df_box = pd.DataFrame(rows)

    bx = fig.add_axes([0.075, 0.16, 0.24, 0.36])
    sns.boxplot(data=df_box, x="Cat", y="det", order=order_labels, hue="Cat", palette=palette,
                hue_order=order_labels, legend=False, width=0.62, linewidth=0.7, fliersize=1.0,
                flierprops=dict(marker="o", markersize=1.0, markerfacecolor="0.4",
                                markeredgecolor="none", alpha=0.4), ax=bx)
    bx.set_xlabel("")
    bx.set_ylabel("$\\Delta$ET (mm yr$^{-1}$)", size=LABELSIZE - 4)
    bx.tick_params(axis="both", labelsize=TICKLABELSIZE - 4)
    bx.tick_params(axis="x", rotation=0, length=2)
    for sp in bx.spines.values():
        sp.set_linewidth(0.8)
    bx.patch.set_facecolor("white")
    bx.patch.set_alpha(0.9)
    fig.text(0.012, 0.585, "b", fontsize=LABELSIZE, fontweight="bold", va="bottom", ha="left", zorder=6)

    fig.savefig(OUT, dpi=200)
    a = whole[mp]
    print(f"  all catchments (n={a.size}): median={np.median(a):6.1f}  "
          f"IQR=[{np.percentile(a, 25):.1f}, {np.percentile(a, 75):.1f}]  mm/yr")
    for c in ordered_ids:
        v = whole[mp & (cat == c)]
        print(f"  {KG_MAJOR_NAMES[c]:3s} (n={v.size:4d}): median={np.median(v):6.1f}  "
              f"IQR=[{np.percentile(v, 25):.1f}, {np.percentile(v, 75):.1f}]  mm/yr")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
